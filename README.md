# Real-Time Littering Detection with YOLO and Optical Flow

Full-stack computer vision system for detecting littering behavior from video or IP camera streams. The backend combines YOLO object detection, motion analysis, optical flow tracking, and ownership scoring, then records confirmed violations in SQLite and exposes REST/WebSocket APIs. The frontend provides a React dashboard for live video, alerts, history, and evidence review.

## Features

- YOLO object detection for people and trash candidates.
- Motion, optical-flow, and ownership scoring to confirm violations.
- FastAPI backend with REST endpoints and WebSocket video streaming.
- SQLite violation history with saved evidence images/videos.
- React + Vite dashboard for live monitoring and history review.

## Project Structure

```text
.
├── src/                    # Python backend and detection pipeline
├── frontend/               # React + Vite dashboard
├── docs/                   # Reports and project notes
├── video/                  # Local sample videos (not committed)
├── weights/                # Local model weights (not committed)
├── violations/             # Runtime evidence output (not committed)
├── start_backend.ps1       # Windows backend launcher
├── start_frontend.ps1      # Windows frontend launcher
├── requirements.txt        # Python dependencies
└── .env.example            # Example runtime configuration
```

## Requirements

- Python 3.12 or newer
- Node.js 20 or newer
- Git
- A YOLO model weight file, for example `weights/best.pt`
- A sample video in `video/` when running in file mode

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

Copy `.env.example` to `.env` if you want to customize the video source, camera address, model path, or output directory.

## Run

Start the backend:

```powershell
.\start_backend.ps1 -SourceMode file -VideoFile .\video\di_bo_17.mp4
```

Start the frontend in another terminal:

```powershell
.\start_frontend.ps1
```

Open:

- Frontend: `http://127.0.0.1:5173`
- Backend API docs: `http://127.0.0.1:8000/docs`

## Runtime Data

The following are intentionally ignored by Git:

- Python and Node environments
- model weights in `weights/`
- sample videos in `video/`
- generated evidence in `violations/`
- SQLite databases
- debug outputs, logs, and ZIP archives

Keep large datasets and trained weights outside the repository or publish them separately with a release artifact, cloud storage link, or Git LFS.

## Useful Commands

```powershell
# Frontend quality checks
cd frontend
npm run lint
npm run build

# Backend quick import check
cd ..
python -m compileall src
```
