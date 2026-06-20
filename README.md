# AI Shorts Creator

Upload a long video, get back AI-selected, captioned, 9:16 short clips.

## What's included
- `app/main.py` — FastAPI backend: transcription (Whisper) → viral moment
  detection (GPT-4o) → cut + crop to 9:16 + burned-in animated captions (ffmpeg)
- `Dockerfile` — installs ffmpeg + Python deps, runs the API
- `requirements.txt` — Python dependencies
- `index.html` — simple upload/status/download frontend (host this separately,
  e.g. on Netlify/Vercel, or as a Railway static site)

## Required environment variable
Set this in Railway → your service → Variables:

```
OPENAI_API_KEY = sk-xxxxxxxxxxxxxxxxxxxx
```

## Deploying on Railway
1. Push this repo to GitHub (done via web upload or git push).
2. In Railway, New Project → Deploy from GitHub repo → select this repo.
3. Railway detects the Dockerfile automatically and builds it.
4. Add the `OPENAI_API_KEY` environment variable.
5. Once deployed, Railway gives you a public URL like
   `https://shorts-creater.up.railway.app`.
6. Open `index.html`, replace `API_BASE` with that URL, then host `index.html`
   anywhere (Netlify, Vercel, GitHub Pages, or Railway static site).

## Known limitations (v1)
- Face tracking is currently a **centered 9:16 crop**, not true per-frame face
  tracking. Works well for single-speaker talking-head footage. Multi-person
  scenes need a v2 pass (OpenCV/Mediapipe face detection → dynamic crop path).
- Job state is stored in memory — if the server restarts mid-job, progress is
  lost. For production, move `JOBS` dict to a real database (Railway offers a
  one-click Postgres add-on).
- No authentication/payment yet — anyone with the URL can upload. Add an API
  key check or Stripe paywall before sharing publicly.
