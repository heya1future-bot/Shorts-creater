"""
AI Shorts Creator - Backend
Upload a long video -> get back short, captioned, 9:16 clips.

Flow:
1. POST /upload        -> save video, kick off background processing
2. GET  /status/{job}  -> check progress
3. GET  /download/{job}/{filename} -> get finished clips
"""

import os
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Shorts Creator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "jobs"
WORK_DIR.mkdir(exist_ok=True)

# In-memory job tracking. For real production use, swap this for a database
# (Railway Postgres plugin works well) so progress survives restarts.
JOBS: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str]):
    """Run a shell command and raise if it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def transcribe(video_path: Path) -> dict:
    """Whisper transcription with word-level timestamps."""
    with open(video_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    return transcript.model_dump()


def find_viral_moments(transcript: dict, max_clips: int = 5) -> list[dict]:
    """
    Ask GPT-4o to find the most engaging segments based on the transcript.
    Returns a list of {start, end, title, reason} dicts.
    """
    segments_text = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
        for s in transcript.get("segments", [])
    )

    prompt = f"""You are an expert short-form video editor. Below is a timestamped
transcript of a long video. Identify the {max_clips} most engaging, self-contained
moments that would work as standalone 30-90 second vertical shorts (hooks, strong
claims, emotional peaks, punchlines, surprising facts).

Transcript:
{segments_text}

Respond ONLY with a JSON array, no preamble, no markdown fences. Each item:
{{"start": float, "end": float, "title": "short catchy title", "reason": "why this works as a short"}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def build_captions_ass(transcript: dict, clip_start: float, clip_end: float, out_path: Path):
    """
    Build an .ass subtitle file (animated, styled) for the words that fall
    inside [clip_start, clip_end], re-timed to start at 0.
    """
    words = transcript.get("words", [])
    clip_words = [w for w in words if clip_start <= w["start"] <= clip_end]

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,90,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,6,0,2,40,40,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def fmt_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:01d}:{m:02d}:{s:05.2f}"

    lines = [header]
    # Group words into short caption chunks (~3 words at a time) for the
    # punchy animated-caption style used in shorts.
    chunk = []
    chunk_start = None
    for w in clip_words:
        rel_start = w["start"] - clip_start
        rel_end = w["end"] - clip_start
        if chunk_start is None:
            chunk_start = rel_start
        chunk.append(w["word"])
        if len(chunk) >= 3:
            text = " ".join(chunk).upper()
            lines.append(
                f"Dialogue: 0,{fmt_time(chunk_start)},{fmt_time(rel_end)},Default,,0,0,0,,{{\\fad(80,80)}}{text}"
            )
            chunk = []
            chunk_start = None
    if chunk:
        text = " ".join(chunk).upper()
        lines.append(
            f"Dialogue: 0,{fmt_time(chunk_start)},{fmt_time(rel_end)},Default,,0,0,0,,{{\\fad(80,80)}}{text}"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def cut_clip_with_facetracking(
    source: Path, start: float, end: float, ass_path: Path, out_path: Path
):
    """
    Cuts [start, end] from source, crops/scales to 9:16, burns in captions.

    Face tracking note: true per-frame face tracking requires a separate
    OpenCV/Mediapipe pass that outputs a crop-center path, then feeds that
    into ffmpeg's crop filter with expressions. For a first working version,
    this does a centered 9:16 crop, which works well for single-speaker
    talking-head content. Swap in the face-tracked crop path for v2.
    """
    duration = end - start
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vf = (
        "crop=ih*9/16:ih,scale=1080:1920,"
        f"ass={ass_path.as_posix()}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(source),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        str(out_path),
    ]
    run(cmd)


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------

def process_video(job_id: str, video_path: Path):
    job_dir = video_path.parent
    try:
        JOBS[job_id]["status"] = "transcribing"
        transcript = transcribe(video_path)
        (job_dir / "transcript.json").write_text(json.dumps(transcript))

        JOBS[job_id]["status"] = "finding_viral_moments"
        moments = find_viral_moments(transcript)

        JOBS[job_id]["status"] = "cutting_clips"
        clips = []
        for i, moment in enumerate(moments):
            ass_path = job_dir / f"clip_{i}.ass"
            out_path = job_dir / f"clip_{i}.mp4"
            build_captions_ass(transcript, moment["start"], moment["end"], ass_path)
            cut_clip_with_facetracking(video_path, moment["start"], moment["end"], ass_path, out_path)
            clips.append({
                "filename": out_path.name,
                "title": moment.get("title", f"Clip {i+1}"),
                "reason": moment.get("reason", ""),
                "start": moment["start"],
                "end": moment["end"],
            })

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["clips"] = clips

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / f"source{Path(file.filename).suffix}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {"status": "queued", "clips": []}
    background_tasks.add_task(process_video, job_id, video_path)

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JOBS[job_id]


@app.get("/download/{job_id}/{filename}")
async def download_clip(job_id: str, filename: str):
    file_path = WORK_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@app.get("/")
async def root():
    return {"status": "AI Shorts Creator backend is running"}
