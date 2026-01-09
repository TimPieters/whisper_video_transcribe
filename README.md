# Whisper Turbo Video Transcriber (Local)

Local-only web app for transcribing video files with Whisper turbo. Upload a video, extract audio with ffmpeg, and download TXT/SRT/VTT/JSON outputs with live progress.

## Requirements

- Python 3.10-3.12 (onnxruntime wheels are not available for 3.13+ yet)
- ffmpeg on PATH (ffprobe recommended for better progress estimates)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Install ffmpeg

Windows (choose one):

- Download from https://www.gyan.dev/ffmpeg/builds/ and add `bin` to PATH
- Or with Chocolatey: `choco install ffmpeg`

macOS:

- `brew install ffmpeg`

Linux (Debian/Ubuntu):

- `sudo apt-get install ffmpeg`

## Run

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Whisper turbo model

This app uses `faster-whisper` for reliable local inference and segment-level timestamps. Default model:

- `large-v3-turbo`

You can override with:

- `WHISPER_MODEL=large-v3-turbo`
- `WHISPER_DEVICE=auto` (auto-pick CUDA if available, else CPU; DirectML is detected but faster-whisper falls back to CPU)
- `WHISPER_COMPUTE_TYPE=int8` (use `float16` for GPU)
- `WHISPER_BEAM_SIZE=5`

Models are downloaded to `data/models` on first use.

## Configuration

- `MAX_UPLOAD_MB` (default 2048)
- `JOB_TTL_HOURS` (default 24) for cleaning old job folders

## AMD GPU (WhisperPS)

- WhisperPS module is vendored at `third_party/WhisperPS/WhisperPS.psd1`; the app imports it by absolute path (no external install needed).
- AMD WhisperPS models (GGML `.bin`, default `ggml-medium.bin`) are stored under `data/models/whisperps` and auto-download from the Whisper.cpp Hugging Face repo on first use.
- Environment:
  - `WHISPERPS_MODEL_FILE` to choose the GGML file name (defaults to `ggml-medium.bin`)
  - `WHISPERPS_ADAPTER_CONTAINS` to force a substring match for adapter selection from `Get-Adapters`
- Select the "AMD GPU (WhisperPS)" device in the UI (visible on Windows when an AMD GPU is detected). Transcription runs locally via the PowerShell script in `scripts/whisperps_transcribe.ps1`.
- When AMD is selected, the UI exposes an AMD model dropdown populated from `data/models/whisperps` (falls back to the default name if missing and auto-downloads).

## Notes

- Audio is converted to WAV 16kHz mono for consistent transcription.
- Progress uses SSE. Upload progress is reported directly by the browser.
- If ffmpeg or ffprobe is missing, the job will fail with a clear error.
- The UI shows environment checks and can install Whisper locally if missing.
- Cancel is supported for queued or running jobs. Some audio decode work may still finish after cancellation.
- If you install Whisper from the UI, it runs `pip install` in the active venv.
- Use the Debug panel for ffmpeg command, elapsed time, and stderr tail.

## Project layout

- `app.py` FastAPI server
- `static/index.html` UI
- `static/app.js` UI logic
- `static/styles.css` styling
- `requirements.txt`
