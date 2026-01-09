import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from whisperps_runner import (
    detect_amd_gpu,
    ensure_whisperps_model,
    run_whisperps_transcribe,
    whisperps_can_import,
    whisperps_get_adapters,
    whisperps_import_check,
)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

WhisperModel = None


APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = APP_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = DATA_DIR / "models"
WHISPERPS_MODELS_DIR = MODELS_DIR / "whisperps"
WHISPERPS_MANIFEST = APP_DIR / "third_party" / "WhisperPS" / "WhisperPS.psd1"
WHISPERPS_SCRIPT = APP_DIR / "scripts" / "whisperps_transcribe.ps1"
JOB_META_FILENAME = "job_meta.json"

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".flv",
    ".mpeg",
    ".mpg",
    ".m4v",
}

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
MODEL_OPTIONS = [
    "large-v3-turbo",
    "large-v3",
    "large-v2",
    "distil-large-v3",
    "medium",
    "small",
    "base",
    "tiny",
]
MODEL_SPECS: Dict[str, Dict[str, int]] = {
    # Approximate disk and VRAM footprints (MB) for float16 GPU use.
    "large-v3-turbo": {"disk_mb": 5400, "vram_mb": 7000},
    "large-v3": {"disk_mb": 5200, "vram_mb": 6500},
    "large-v2": {"disk_mb": 5200, "vram_mb": 6500},
    "distil-large-v3": {"disk_mb": 3000, "vram_mb": 4000},
    "medium": {"disk_mb": 1500, "vram_mb": 2500},
    "small": {"disk_mb": 600, "vram_mb": 1200},
    "base": {"disk_mb": 150, "vram_mb": 800},
    "tiny": {"disk_mb": 70, "vram_mb": 512},
}
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
DEFAULT_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
JOB_TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "24"))
WHISPERPS_MODEL_FILE = os.getenv("WHISPERPS_MODEL_FILE", "ggml-medium.bin")
WHISPERPS_MODEL_FALLBACK = "ggml-large-v3-turbo.bin"
WHISPERPS_ADAPTER_CONTAINS = os.getenv("WHISPERPS_ADAPTER_CONTAINS")
AMD_NOTE = "DirectML (AMD/Intel) requires onnxruntime-directml; falls back to CPU if unavailable"
FINAL_STATUSES = {"done", "error", "cancelled"}

logger = logging.getLogger("whisper_app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

_cuda_path_added = False


def _candidate_cuda_dirs() -> List[Path]:
    dirs: List[Path] = []
    for key in ("CUDA_PATH", "CUDA_HOME"):
        val = os.environ.get(key)
        if val:
            dirs.append(Path(val))
    dirs.extend(
        [
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"),
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"),
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"),
        ]
    )
    return dirs


def ensure_cuda_dll_path(job_id: Optional[str]) -> None:
    """On Windows, add CUDA bin to PATH/DLL search if cublas is installed."""
    global _cuda_path_added
    if _cuda_path_added or not sys.platform.startswith("win"):
        return
    for base in _candidate_cuda_dirs():
        bin_dir = base / "bin"
        cublas = bin_dir / "cublas64_12.dll"
        if cublas.exists():
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(bin_dir))
            except Exception:
                # os.add_dll_directory is only on Python 3.8+; PATH change still helps.
                pass
            _cuda_path_added = True
            log(job_id or "n/a", f"Using CUDA from {bin_dir}")
            return
    log(job_id or "n/a", "CUDA cublas64_12.dll not found in default locations; CUDA load may fail.")


@dataclass
class Job:
    job_id: str
    created_at: float
    updated_at: float
    status: str
    progress: float
    message: str
    progress_mode: str = "determinate"
    error: Optional[str] = None
    queue_position: Optional[int] = None
    ffmpeg_cmd: Optional[str] = None
    ffmpeg_started_at: Optional[float] = None
    ffmpeg_pid: Optional[int] = None
    ffmpeg_last_progress: Optional[str] = None
    ffmpeg_stderr_tail: Optional[str] = None
    ffmpeg_stdout_tail: Optional[str] = None
    ffmpeg_exit_code: Optional[int] = None
    ffmpeg_error: Optional[str] = None
    ffmpeg_wav_size_kb: Optional[int] = None
    ffmpeg_last_activity: Optional[float] = None
    input_path: Optional[str] = None
    audio_path: Optional[str] = None
    audio_info: Optional[str] = None
    duration_sec: Optional[float] = None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    text: Optional[str] = None
    outputs: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    listeners: List[asyncio.Queue] = field(default_factory=list)
    transcribe_stdout_tail: Optional[str] = None
    transcribe_stderr_tail: Optional[str] = None
    transcribe_pid: Optional[int] = None
    transcribe_last_activity: Optional[float] = None


jobs: Dict[str, Job] = {}
jobs_lock = asyncio.Lock()
job_queue: asyncio.Queue[str]
main_loop: asyncio.AbstractEventLoop
worker_task: Optional[asyncio.Task] = None
job_queue_order: List[str] = []

model_cache: Dict[str, Any] = {}
model_lock = threading.Lock()


def log(job_id: str, message: str) -> None:
    logger.info("[job %s] %s", job_id, message)


def ensure_dirs() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WHISPERPS_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    base = os.path.basename(name)
    base = base.replace(" ", "_")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or f"upload_{uuid.uuid4().hex}"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_ffmpeg_path() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found on PATH")
    return ffmpeg_path


def get_ffprobe_path() -> str:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RuntimeError("ffprobe not found on PATH")
    return ffprobe_path


def try_import_whisper() -> tuple[bool, Optional[str]]:
    global WhisperModel
    if WhisperModel is not None:
        return True, None
    try:
        from faster_whisper import WhisperModel as FW

        WhisperModel = FW
        return True, None
    except Exception as exc:
        return False, str(exc)


def probe_duration(path: str) -> Optional[float]:
    try:
        ffprobe = get_ffprobe_path()
    except RuntimeError:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        return float(output) if output else None
    except Exception:
        return None


def wav_duration(path: str) -> Optional[float]:
    try:
        with wave.open(path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)
    except Exception:
        return None


def format_timestamp_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000.0))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000.0))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_ffmpeg_timestamp(value: str) -> Optional[float]:
    if not value:
        return None
    if value.isdigit():
        return float(value) / 1_000_000.0
    if "." in value and value.replace(".", "").isdigit():
        return float(value) / 1_000_000.0
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return None


def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, segment in enumerate(segments, start=1):
        start = format_timestamp_srt(segment["start"])
        end = format_timestamp_srt(segment["end"])
        text = segment["text"].strip()
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        start = format_timestamp_vtt(segment["start"])
        end = format_timestamp_vtt(segment["end"])
        text = segment["text"].strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def delete_file_safely(path: Optional[str]) -> None:
    if not path:
        return
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception as exc:
        logger.warning("Failed to delete file %s: %s", path, exc)


def list_whisperps_models() -> List[str]:
    files = []
    if WHISPERPS_MODELS_DIR.exists():
        for item in WHISPERPS_MODELS_DIR.glob("*.bin"):
            files.append(item.name)
    if WHISPERPS_MODEL_FILE not in files:
        files.append(WHISPERPS_MODEL_FILE)
    return sorted(set(files))


def pick_whisperps_adapter(adapters: List[str]) -> Optional[str]:
    if WHISPERPS_ADAPTER_CONTAINS:
        for name in adapters:
            if WHISPERPS_ADAPTER_CONTAINS.lower() in name.lower():
                return name
    for name in adapters:
        if re.search(r"(amd|radeon)", name, re.IGNORECASE):
            return name
    return adapters[0] if adapters else None


def resolve_device_choice(requested: str, preflight: Dict[str, Any]) -> str:
    device = requested or "auto"
    if device != "auto":
        return device
    if preflight.get("cuda_available"):
        return "cuda"
    if preflight.get("has_amd_gpu") and preflight.get("whisperps_available"):
        return "amd"
    if preflight.get("dml_available"):
        return "dml"
    return "cpu"


def job_public_state(job: Job) -> Dict[str, Any]:
    input_name = Path(job.input_path).name if job.input_path else None
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "progress_mode": job.progress_mode,
        "error": job.error,
        "queue_position": job.queue_position,
        "audio_info": job.audio_info,
        "duration_sec": job.duration_sec,
        "updated_at": job.updated_at,
        "created_at": job.created_at,
        "input_filename": input_name,
        # debug fields
        "ffmpeg_cmd": job.ffmpeg_cmd,
        "ffmpeg_started_at": job.ffmpeg_started_at,
        "ffmpeg_pid": job.ffmpeg_pid,
        "ffmpeg_last_progress": job.ffmpeg_last_progress,
        "ffmpeg_stderr_tail": job.ffmpeg_stderr_tail,
        "ffmpeg_stdout_tail": job.ffmpeg_stdout_tail,
        "ffmpeg_exit_code": job.ffmpeg_exit_code,
        "ffmpeg_error": job.ffmpeg_error,
        "ffmpeg_wav_size_kb": job.ffmpeg_wav_size_kb,
        "ffmpeg_last_activity": job.ffmpeg_last_activity,
        "device": (job.config.get("resolved_device") or job.config.get("device")) if job.config else None,
        "preview": job.text,
        "available_formats": list(job.outputs.keys()) if job.outputs else [],
    }


def persist_job_metadata(job: Job) -> None:
    if not job.input_path:
        return
    try:
        job_dir = Path(job.input_path).parent
        payload = {
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "error": job.error,
            "progress": job.progress,
            "progress_mode": job.progress_mode,
            "queue_position": job.queue_position,
            "audio_info": job.audio_info,
            "duration_sec": job.duration_sec,
            "updated_at": job.updated_at,
            "created_at": job.created_at,
            "input_filename": Path(job.input_path).name if job.input_path else None,
            "outputs": job.outputs,
            "config": job.config,
            "preview": job.text,
        }
        meta_path = job_dir / JOB_META_FILENAME
        job_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to persist job metadata for %s: %s", job.job_id, exc)


def load_job_metadata(job_id: str) -> Optional[Dict[str, Any]]:
    meta_path = JOBS_DIR / job_id / JOB_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    job_dir = meta_path.parent.resolve()
    outputs: Dict[str, str] = {}
    for fmt, path in (data.get("outputs") or {}).items():
        try:
            resolved = Path(path).resolve()
            if resolved.exists():
                if resolved.is_relative_to(job_dir):
                    outputs[fmt] = str(resolved)
                elif job_dir in resolved.parents:
                    outputs[fmt] = str(resolved)
        except Exception:
            continue
    data["outputs"] = outputs
    data["available_formats"] = list(outputs.keys())
    return data


def publish_event(job: Job) -> None:
    if not job.listeners:
        return
    data = job_public_state(job)
    for queue in list(job.listeners):
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            continue


def update_job(job_id: str, **kwargs: Any) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    for key, value in kwargs.items():
        setattr(job, key, value)
    job.updated_at = time.time()
    publish_event(job)
    if job.status in FINAL_STATUSES or ("outputs" in kwargs):
        persist_job_metadata(job)


def schedule_job_update(job_id: str, **kwargs: Any) -> None:
    try:
        loop = main_loop
    except Exception:
        loop = None
    if loop:
        try:
            loop.call_soon_threadsafe(lambda: update_job(job_id, **kwargs))
            return
        except Exception as exc:
            logger.error("call_soon_threadsafe failed: %s", exc)
    update_job(job_id, **kwargs)


def run_ffmpeg_extract(
    job_id: str,
    input_path: str,
    output_path: str,
    duration_sec: Optional[float],
    cancel_event: threading.Event,
) -> None:
    log(job_id, f"run_ffmpeg_extract invoked with input={input_path} output={output_path}")
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "debug",
        "-nostdin",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        output_path,
    ]
    log(job_id, f"Running ffmpeg to extract audio: {json.dumps(cmd)}")
    start_time = time.monotonic()
    last_emit = 0.0
    last_progress_log = 0.0
    ticker_stop = threading.Event()
    progress_seen = threading.Event()
    stalled_logged = threading.Event()
    last_activity = time.monotonic()
    stderr_tail: List[str] = []
    last_stderr_emit = 0.0
    process_ref: Dict[str, Optional[subprocess.Popen]] = {"p": None}

    schedule_job_update(
        job_id,
        ffmpeg_cmd=" ".join(cmd),
        ffmpeg_started_at=time.time(),
        ffmpeg_pid=None,
        ffmpeg_last_progress=None,
        ffmpeg_stderr_tail="",
        ffmpeg_stdout_tail="",
        ffmpeg_exit_code=None,
        ffmpeg_error=None,
        ffmpeg_wav_size_kb=0,
        ffmpeg_last_activity=time.time(),
    )

    def tick_elapsed() -> None:
        while not ticker_stop.is_set():
            elapsed = time.monotonic() - start_time
            if elapsed >= 20 and not progress_seen.is_set() and not stalled_logged.is_set():
                stalled_logged.set()
                log(job_id, "ffmpeg running without progress output yet")
            size_kb = Path(output_path).stat().st_size // 1024 if Path(output_path).exists() else 0
            schedule_job_update(
                job_id,
                status="extracting",
                progress=2 if not progress_seen.is_set() else None,
                progress_mode="indeterminate" if not progress_seen.is_set() else "determinate",
                message=f"Extracting audio (elapsed {format_duration(elapsed)}, wav size {size_kb}KB)",
                ffmpeg_wav_size_kb=size_kb,
                ffmpeg_last_activity=time.time(),
            )
            # kill if no activity for 30s
            if not progress_seen.is_set() and (time.monotonic() - last_activity) > 30:
                p = process_ref.get("p")
                if p and p.poll() is None:
                    log(job_id, "ffmpeg produced no progress for 30s; terminating")
                    schedule_job_update(
                        job_id,
                        status="error",
                        progress=0,
                        progress_mode="determinate",
                        message="ffmpeg stalled (no progress for 30s)",
                        error="ffmpeg produced no progress for 30s; check input file and ffmpeg permissions",
                    )
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    ticker_stop.set()
                    return
            ticker_stop.wait(1.0)

    threading.Thread(target=tick_elapsed, daemon=True).start()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    process_ref["p"] = process
    log(job_id, f"ffmpeg started with pid={process.pid}")
    schedule_job_update(job_id, ffmpeg_pid=process.pid)
    try:
        # stderr is merged into stdout (stderr=STDOUT) so no separate reader

        stdout_tail: List[str] = []

        def read_stdout_progress() -> None:
            nonlocal last_emit, last_progress_log, last_activity, stdout_tail
            if process.stdout is None:
                return
            for line in process.stdout:
                if cancel_event.is_set():
                    break
                line = line.rstrip("\n")
                if not line:
                    continue
                logger.debug("[job %s] ffmpeg stdout: %s", job_id, line)
                stdout_tail.append(line)
                if len(stdout_tail) > 200:
                    stdout_tail = stdout_tail[-200:]
                # progress parsing
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key in {"out_time_ms", "out_time_us", "out_time"}:
                        current_time = parse_ffmpeg_timestamp(value)
                        now = time.monotonic()
                        if now - last_emit < 0.5:
                            continue
                        last_emit = now
                        last_activity = now
                        progress_seen.set()
                        schedule_job_update(
                            job_id,
                            ffmpeg_last_progress=f"{key}={value}",
                            ffmpeg_last_activity=time.time(),
                        )
                        if now - last_progress_log >= 15.0 and current_time is not None:
                            last_progress_log = now
                            log(
                                job_id,
                                f"ffmpeg progress {format_duration(current_time)} / "
                                f"{format_duration(duration_sec) if duration_sec else 'unknown'}",
                            )
                        if duration_sec and current_time is not None:
                            progress = min(99.0, (current_time / duration_sec) * 100.0)
                            schedule_job_update(
                                job_id,
                                status="extracting",
                                progress=progress,
                                progress_mode="determinate",
                                message="Extracting audio",
                            )
                        else:
                            readable = format_duration(current_time or (now - start_time))
                            schedule_job_update(
                                job_id,
                                status="extracting",
                                progress=2,
                                progress_mode="indeterminate",
                                message=f"Extracting audio (processed {readable})",
                            )
                # store stdout tail (last 200 lines) and mirror to stderr tail for visibility
                tail_text = "\n".join(stdout_tail)
                schedule_job_update(
                    job_id,
                    ffmpeg_stdout_tail=tail_text,
                    ffmpeg_stderr_tail=tail_text,
                    ffmpeg_last_activity=time.time(),
                )

        threading.Thread(target=read_stdout_progress, daemon=True).start()

        return_code = process.wait()
        stderr_text = "\n".join(stderr_tail) if stderr_tail else ""
        stdout_text = "\n".join(stdout_tail) if stdout_tail else ""
        schedule_job_update(
            job_id,
            ffmpeg_exit_code=return_code,
            ffmpeg_stderr_tail=stderr_text,
            ffmpeg_stdout_tail=stdout_text,
        )
        if return_code != 0:
            schedule_job_update(
                job_id,
                ffmpeg_error=stderr_text.strip() or f"ffmpeg exited with code {return_code}",
            )
            raise RuntimeError(f"ffmpeg failed: {stderr_text.strip()}")
        last_activity = time.monotonic()
    finally:
        ticker_stop.set()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


def load_model(job_id: str, model_name: str) -> Any:
    ok, err = try_import_whisper()
    if not ok:
        raise RuntimeError(f"faster-whisper is not installed ({err})")
    # pick device from job config or auto
    job = jobs.get(job_id)
    chosen_device = DEFAULT_DEVICE
    if job and job.config.get("resolved_device"):
        chosen_device = job.config.get("resolved_device") or DEFAULT_DEVICE
    elif job and job.config.get("device"):
        chosen_device = job.config.get("device") or DEFAULT_DEVICE
    device = chosen_device
    if device == "auto":
        device = "cuda" if shutil.which("nvidia-smi") else ("dml" if shutil.which("dxdiag") else "cpu")
    if device == "dml":
        # faster-whisper does not natively support DirectML; fall back to CPU
        log(job_id, "Requested DirectML/AMD path; faster-whisper lacks native support. Falling back to CPU.")
        device = "cpu"
    with model_lock:
        if model_name in model_cache:
            return model_cache[model_name]
        compute_type = DEFAULT_COMPUTE_TYPE
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        if device == "cuda":
            ensure_cuda_dll_path(job_id)
        log(job_id, f"Loading model {model_name} on {device} ({compute_type})")
        model_cache[model_name] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(MODELS_DIR),
        )
        return model_cache[model_name]


def write_outputs(
    job_dir: Path,
    text: str,
    segments: List[Dict[str, Any]],
    language: Optional[str],
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    txt_path = job_dir / "transcript.txt"
    txt_path.write_text(text, encoding="utf-8")
    outputs["txt"] = str(txt_path)

    srt_path = job_dir / "transcript.srt"
    srt_path.write_text(segments_to_srt(segments), encoding="utf-8")
    outputs["srt"] = str(srt_path)

    vtt_path = job_dir / "transcript.vtt"
    vtt_path.write_text(segments_to_vtt(segments), encoding="utf-8")
    outputs["vtt"] = str(vtt_path)

    json_path = job_dir / "transcript.json"
    payload = {
        "language": language or "auto",
        "segments": segments,
        "text": text,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["json"] = str(json_path)
    return outputs


def process_job_whisperps(
    job_id: str,
    job: Job,
    audio_path: str,
    actual_duration: Optional[float],
) -> None:
    language = job.config.get("language")
    if language == "auto":
        language = None
    if not language:
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message="Language is required when using WhisperPS (AMD). Set a language code (e.g. nl).",
            error="WhisperPS requires an explicit language.",
        )
        return
    model_name = job.config.get("whisperps_model") or WHISPERPS_MODEL_FILE
    model_path = WHISPERPS_MODELS_DIR / model_name
    fallback_model_name = WHISPERPS_MODEL_FALLBACK
    fallback_model_path = WHISPERPS_MODELS_DIR / fallback_model_name
    job_dir = Path(audio_path).parent

    def status_cb(message: str) -> None:
        schedule_job_update(
            job_id,
            status="loading_model",
            progress=None,
            progress_mode="indeterminate",
            message=message,
        )

    try:
        ensure_whisperps_model(model_name, model_path, status_cb=status_cb)
    except Exception as exc:
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message=str(exc),
            error=str(exc),
        )
        return

    if job.cancel_event.is_set():
        schedule_job_update(
            job_id,
            status="cancelled",
            progress=0,
            progress_mode="determinate",
            message="Cancelled",
        )
        return

    adapters = whisperps_get_adapters(WHISPERPS_MANIFEST) if whisperps_can_import(WHISPERPS_MANIFEST) else []
    adapter_name = pick_whisperps_adapter(adapters)
    log(job_id, f"WhisperPS adapters found: {adapters} selected={adapter_name or 'auto'} model={model_path}")
    schedule_job_update(
        job_id,
        status="loading_model",
        progress=None,
        progress_mode="indeterminate",
        message=f"Loading WhisperPS model ({adapter_name or 'auto adapter'})",
    )

    tail: List[str] = []

    def log_callback(line: str) -> None:
        tail.append(line)
        if len(tail) > 200:
            tail[:] = tail[-200:]
        schedule_job_update(
            job_id,
            transcribe_stdout_tail="\n".join(tail),
            transcribe_last_activity=time.time(),
            status="transcribing",
            progress=None,
            progress_mode="indeterminate",
            message="Transcribing (AMD)",
        )

    schedule_job_update(
        job_id,
        status="transcribing",
        progress=None,
        progress_mode="indeterminate",
        message="Transcribing (AMD)",
    )
    return_code, tail_text, ps_pid = run_whisperps_transcribe(
        WHISPERPS_SCRIPT,
        WHISPERPS_MANIFEST,
        model_path,
        Path(audio_path),
        job_dir,
        language,
        job.config.get("output_format") or "json",
        adapter_name,
        log_callback=log_callback,
        cancel_event=job.cancel_event,
    )
    schedule_job_update(
        job_id,
        transcribe_stdout_tail=tail_text,
        transcribe_stderr_tail=tail_text,
        transcribe_last_activity=time.time(),
        transcribe_pid=ps_pid,
    )

    if return_code != 0 and ("Error loading the model" in (tail_text or "")) and model_name != fallback_model_name:
        log(job_id, f"Primary AMD model failed to load; retrying with fallback {fallback_model_name}")
        try:
            ensure_whisperps_model(fallback_model_name, fallback_model_path, status_cb=status_cb)
            job.config["whisperps_model"] = fallback_model_name
            return_code, tail_text, ps_pid = run_whisperps_transcribe(
                WHISPERPS_SCRIPT,
                WHISPERPS_MANIFEST,
                fallback_model_path,
                Path(audio_path),
                job_dir,
                language,
                job.config.get("output_format") or "json",
                adapter_name,
                log_callback=log_callback,
                cancel_event=job.cancel_event,
            )
            schedule_job_update(
                job_id,
                transcribe_stdout_tail=tail_text,
                transcribe_stderr_tail=tail_text,
                transcribe_last_activity=time.time(),
                transcribe_pid=ps_pid,
            )
            model_path = fallback_model_path
            model_name = fallback_model_name
        except Exception as exc:
            log(job_id, f"Fallback model load failed: {exc}")

    if job.cancel_event.is_set():
        schedule_job_update(
            job_id,
            status="cancelled",
            progress=0,
            progress_mode="determinate",
            message="Cancelled",
        )
        return

    if return_code != 0:
        err_text = tail_text.strip() if tail_text else f"WhisperPS exited with code {return_code}"
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message="WhisperPS failed. See debug output.",
            error=err_text,
        )
        return

    json_path = job_dir / "transcript.json"
    txt_path = job_dir / "transcript.txt"
    srt_path = job_dir / "transcript.srt"
    vtt_path = job_dir / "transcript.vtt"

    segments_data: List[Dict[str, Any]] = []
    text = ""
    detected_language: Optional[str] = language
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            detected_language = payload.get("language") or detected_language
            segments_raw = payload.get("segments") or []
            for seg in segments_raw:
                segments_data.append(
                    {
                        "start": float(seg.get("start") or seg.get("Start") or 0.0),
                        "end": float(seg.get("end") or seg.get("End") or 0.0),
                        "text": (seg.get("text") or seg.get("Text") or "").strip(),
                    }
                )
            text = payload.get("text") or text
        except Exception:
            segments_data = []
            text = ""
    if not text and txt_path.exists():
        try:
            text = txt_path.read_text(encoding="utf-8")
        except Exception:
            text = ""

    outputs: Dict[str, str] = {}
    for fmt, path in {
        "txt": txt_path,
        "srt": srt_path,
        "vtt": vtt_path,
        "json": json_path,
    }.items():
        if path.exists():
            outputs[fmt] = str(path)

    if not outputs:
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message="WhisperPS did not produce outputs.",
            error="WhisperPS outputs missing",
        )
        return

    schedule_job_update(
        job_id,
        status="packaging",
        progress=0,
        progress_mode="indeterminate",
        message="Packaging output",
    )
    delete_file_safely(job.input_path)
    preview = text[:2000] if text else ""
    schedule_job_update(
        job_id,
        status="done",
        progress=100,
        progress_mode="determinate",
        message="Done",
        text=preview,
        segments=segments_data,
        outputs=outputs,
    )


def process_job(job_id: str) -> None:
    job = jobs[job_id]
    if job.cancel_event.is_set():
        schedule_job_update(
            job_id,
            status="cancelled",
            progress=0,
            progress_mode="determinate",
            message="Cancelled",
        )
        return

    input_path = job.input_path
    if not input_path:
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message="Missing input file",
            error="Missing input file",
        )
        return

    try:
        get_ffmpeg_path()
    except RuntimeError as exc:
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message=str(exc),
            error=str(exc),
        )
        return

    duration_sec = probe_duration(input_path)
    schedule_job_update(
        job_id,
        status="extracting",
        progress=2,
        progress_mode="indeterminate",
        message="Preparing audio extraction",
    )
    if not Path(input_path).exists():
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message=f"Input file missing at {input_path}",
            error=f"Input file missing at {input_path}",
        )
        return

    audio_path = str((Path(input_path).parent / "audio.wav"))
    log(job_id, f"Starting ffmpeg extraction from {input_path} to {audio_path}")
    try:
        run_ffmpeg_extract(job_id, input_path, audio_path, duration_sec, job.cancel_event)
    except Exception as exc:
        log(job_id, f"ffmpeg extraction failed: {exc}")
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message=str(exc),
            error=str(exc),
        )
        return
    log(job_id, "ffmpeg extraction finished")

    if job.cancel_event.is_set():
        schedule_job_update(
            job_id,
            status="cancelled",
            progress=0,
            progress_mode="determinate",
            message="Cancelled",
        )
        return

    actual_duration = wav_duration(audio_path) or duration_sec
    audio_info = (
        f"Converted to WAV 16kHz mono as audio.wav "
        f"(duration {format_duration(actual_duration)})"
    )
    schedule_job_update(
        job_id,
        audio_path=audio_path,
        duration_sec=actual_duration,
        audio_info=audio_info,
    )

    preflight_info = get_preflight_status()
    resolved_device = resolve_device_choice(job.config.get("device") or "auto", preflight_info)
    job.config["resolved_device"] = resolved_device
    if resolved_device == "amd":
        available, import_err = whisperps_import_check(WHISPERPS_MANIFEST)
        if not available:
            log(job_id, f"WhisperPS import failed: {import_err}")
            schedule_job_update(
                job_id,
                status="error",
                progress=0,
                progress_mode="determinate",
                message="WhisperPS module unavailable on this system.",
                error=f"WhisperPS import failed: {import_err}",
            )
            return
        if not WHISPERPS_SCRIPT.exists():
            schedule_job_update(
                job_id,
                status="error",
                progress=0,
                progress_mode="determinate",
                message=f"WhisperPS script missing at {WHISPERPS_SCRIPT}",
                error=f"WhisperPS script missing at {WHISPERPS_SCRIPT}",
            )
            return
        schedule_job_update(
            job_id,
            status="loading_model",
            progress=None,
            progress_mode="indeterminate",
            message="Preparing AMD model",
        )
        process_job_whisperps(job_id, job, audio_path, actual_duration)
        return

    schedule_job_update(
        job_id,
        status="loading_model",
        progress=0,
        progress_mode="indeterminate",
        message="Loading model",
    )
    try:
        model = load_model(job_id, job.config["model"])
    except Exception as exc:
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message=str(exc),
            error=str(exc),
        )
        return

    if job.cancel_event.is_set():
        schedule_job_update(
            job_id,
            status="cancelled",
            progress=0,
            progress_mode="determinate",
            message="Cancelled",
        )
        return

    schedule_job_update(
        job_id,
        status="transcribing",
        progress=0,
        progress_mode="indeterminate" if actual_duration is None else "determinate",
        message="Transcribing",
    )

    language = job.config.get("language")
    if language == "auto":
        language = None
    word_timestamps = bool(job.config.get("word_timestamps"))

    segments_data: List[Dict[str, Any]] = []
    text_chunks: List[str] = []

    try:
        segments, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            beam_size=DEFAULT_BEAM_SIZE,
        )
        detected_language = info.language if info and getattr(info, "language", None) else language
        for segment in segments:
            if job.cancel_event.is_set():
                raise RuntimeError("Job cancelled during transcription")
            seg_text = segment.text or ""
            text_chunks.append(seg_text)
            seg = {
                "id": segment.id,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": seg_text.strip(),
            }
            if word_timestamps and getattr(segment, "words", None):
                seg["words"] = [
                    {
                        "word": word.word,
                        "start": float(word.start),
                        "end": float(word.end),
                    }
                    for word in segment.words
                ]
            segments_data.append(seg)
            if actual_duration:
                progress = min(99.0, (segment.end / actual_duration) * 100.0)
                message = (
                    f"Transcribing {format_duration(segment.end)} / "
                    f"{format_duration(actual_duration)}"
                )
            else:
                progress = 0.0
                message = "Transcribing (estimating...)"
            schedule_job_update(
                job_id,
                status="transcribing",
                progress=progress,
                progress_mode="determinate" if actual_duration else "indeterminate",
                message=message,
            )
    except Exception as exc:
        schedule_job_update(
            job_id,
            status="error",
            progress=0,
            progress_mode="determinate",
            message=str(exc),
            error=str(exc),
        )
        return

    if job.cancel_event.is_set():
        schedule_job_update(
            job_id,
            status="cancelled",
            progress=0,
            progress_mode="determinate",
            message="Cancelled",
        )
        return

    text = "".join(text_chunks).strip()
    schedule_job_update(
        job_id,
        status="packaging",
        progress=0,
        progress_mode="indeterminate",
        message="Packaging output",
    )

    outputs = write_outputs(Path(input_path).parent, text, segments_data, detected_language)
    delete_file_safely(input_path)

    preview = text[:2000]
    schedule_job_update(
        job_id,
        status="done",
        progress=100,
        progress_mode="determinate",
        message="Done",
        text=preview,
        segments=segments_data,
        outputs=outputs,
    )


async def job_worker() -> None:
    while True:
        job_id = await job_queue.get()
        if job_id in job_queue_order:
            job_queue_order.remove(job_id)
            refresh_queue_positions()
        job = jobs.get(job_id)
        if not job:
            job_queue.task_done()
            continue
        if job.cancel_event.is_set():
            update_job(
                job_id,
                status="cancelled",
                progress=0,
                progress_mode="determinate",
                message="Cancelled",
            )
            job_queue.task_done()
            continue
        update_job(
            job_id,
            status="extracting",
            progress=1,
            progress_mode="indeterminate",
            message="Starting audio extraction",
            queue_position=None,
        )
        log(job_id, "Starting job")
        await asyncio.to_thread(process_job, job_id)
        log(job_id, f"Job finished with status {jobs[job_id].status}")
        job_queue.task_done()


def cleanup_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    for job_id, job in list(jobs.items()):
        if job.updated_at < cutoff and job.status in {"done", "error", "cancelled"}:
            try:
                if job.input_path:
                    shutil.rmtree(Path(job.input_path).parent, ignore_errors=True)
            except Exception:
                pass
            jobs.pop(job_id, None)


def refresh_queue_positions() -> None:
    for index, queued_id in enumerate(job_queue_order, start=1):
        queued_job = jobs.get(queued_id)
        if not queued_job or queued_job.status != "queued":
            continue
        update_job(
            queued_id,
            queue_position=index,
            progress_mode="indeterminate",
            message=f"Queued (position {index}) - waiting for worker",
        )


def get_preflight_status() -> Dict[str, Any]:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    whisper_ok, whisper_error = try_import_whisper()
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    cuda_available = shutil.which("nvidia-smi") is not None
    gpu_name = None
    gpu_vram_mb: Optional[int] = None
    if cuda_available:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                line = result.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if parts:
                    gpu_name = parts[0]
                if len(parts) > 1 and parts[1].isdigit():
                    gpu_vram_mb = int(parts[1])
        except Exception:
            gpu_name = None
    dml_available = False
    dml_note = AMD_NOTE
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        dml_available = any("Dml" in p for p in providers)
        if dml_available:
            dml_note = "DirectML available via onnxruntime-directml"
    except Exception as exc:  # pragma: no cover
        dml_note = f"DirectML not available ({exc})"
    has_amd_gpu = False
    amd_names: List[str] = []
    whisperps_available = False
    whisperps_error: Optional[str] = None
    whisperps_adapters: List[str] = []
    if sys.platform.startswith("win"):
        has_amd_gpu, amd_names = detect_amd_gpu()
        logger.info("AMD detection: has_amd_gpu=%s names=%s", has_amd_gpu, amd_names)
        whisperps_available, whisperps_error = whisperps_import_check(WHISPERPS_MANIFEST)
        if whisperps_available:
            whisperps_adapters = whisperps_get_adapters(WHISPERPS_MANIFEST)
            logger.info("WhisperPS adapters: %s", whisperps_adapters)
        else:
            logger.warning("WhisperPS import check failed: %s", whisperps_error)
    device_options = ["auto", "cpu"]
    if cuda_available:
        device_options.append("cuda")
    if dml_available:
        device_options.append("dml")
    if has_amd_gpu:
        device_options.append("amd")
    whisperps_model_path = (WHISPERPS_MODELS_DIR / WHISPERPS_MODEL_FILE).resolve()
    return {
        "python_version": python_version,
        "python_supported": sys.version_info < (3, 13),
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "whisper": whisper_ok,
        "whisper_error": whisper_error,
        "model": DEFAULT_MODEL,
        "model_options": MODEL_OPTIONS,
        "model_specs": MODEL_SPECS,
        "device": DEFAULT_DEVICE,
        "compute_type": DEFAULT_COMPUTE_TYPE,
        "cuda_available": cuda_available,
        "cuda_gpu_name": gpu_name,
        "cuda_vram_mb": gpu_vram_mb,
        "dml_available": dml_available,
        "dml_note": dml_note,
        "has_amd_gpu": has_amd_gpu,
        "amd_gpu_names": amd_names,
        "whisperps_manifest_path": str(WHISPERPS_MANIFEST.resolve()),
        "whisperps_available": whisperps_available,
        "whisperps_error": whisperps_error,
        "whisperps_model_present": whisperps_model_path.exists(),
        "whisperps_model_default": WHISPERPS_MODEL_FILE,
        "whisperps_models": list_whisperps_models(),
        "whisperps_adapters": whisperps_adapters,
        "device_options": device_options,
    }


def list_persisted_jobs() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not JOBS_DIR.exists():
        return records
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        meta = load_job_metadata(job_id)
        if meta:
            records.append(
                {
                    "job_id": job_id,
                    "status": meta.get("status"),
                    "message": meta.get("message"),
                    "error": meta.get("error"),
                    "created_at": meta.get("created_at"),
                    "updated_at": meta.get("updated_at"),
                    "input_filename": meta.get("input_filename"),
                    "available_formats": meta.get("available_formats", []),
                    "outputs": meta.get("outputs", {}),
                    "preview": meta.get("preview"),
                    "duration_sec": meta.get("duration_sec"),
                }
            )
            continue
        outputs: Dict[str, str] = {}
        transcript_map = {
            "txt": job_dir / "transcript.txt",
            "srt": job_dir / "transcript.srt",
            "vtt": job_dir / "transcript.vtt",
            "json": job_dir / "transcript.json",
        }
        for fmt, path in transcript_map.items():
            if path.exists():
                outputs[fmt] = str(path)
        if not outputs:
            continue
        preview_text = ""
        txt_path = transcript_map["txt"]
        if txt_path.exists():
            try:
                preview_text = txt_path.read_text(encoding="utf-8")[:2000]
            except Exception:
                preview_text = ""
        timestamp = job_dir.stat().st_mtime
        records.append(
            {
                "job_id": job_id,
                "status": "done",
                "message": "Done",
                "error": None,
                "created_at": timestamp,
                "updated_at": timestamp,
                "input_filename": None,
                "available_formats": list(outputs.keys()),
                "outputs": outputs,
                "preview": preview_text,
                "duration_sec": None,
            }
        )
    return records


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup() -> None:
    ensure_dirs()
    global main_loop, job_queue, worker_task
    main_loop = asyncio.get_running_loop()
    job_queue = asyncio.Queue()
    worker_task = asyncio.create_task(job_worker())
    cleanup_old_jobs()


@app.get("/")
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/preflight")
async def preflight() -> JSONResponse:
    then = get_preflight_status()
    return JSONResponse(then)


@app.get("/jobs")
async def jobs_history() -> JSONResponse:
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for job_id, job in jobs.items():
        record = job_public_state(job)
        record["input_filename"] = Path(job.input_path).name if job.input_path else None
        record["created_at"] = job.created_at
        items.append(record)
        seen.add(job_id)
    for record in list_persisted_jobs():
        if record["job_id"] in seen:
            continue
        items.append(record)
    items.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
    return JSONResponse({"jobs": items})


@app.post("/install")
async def install_whisper() -> JSONResponse:
    status = get_preflight_status()
    if not status["python_supported"]:
        return JSONResponse(
            {
                "status": "error",
                "message": "Python 3.10-3.12 required for onnxruntime wheels.",
                "output": "",
            },
            status_code=400,
        )
    if status["whisper"]:
        return JSONResponse({"status": "ok", "message": "Whisper already installed.", "output": ""})

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "faster-whisper",
        "onnxruntime>=1.14,<2",
    ]
    result = await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return JSONResponse(
            {"status": "error", "message": "Installation failed.", "output": output},
            status_code=500,
        )
    ok, err = try_import_whisper()
    if not ok:
        return JSONResponse(
            {
                "status": "error",
                "message": "Installation completed, but import failed.",
                "output": output + f"\n{err}",
            },
            status_code=500,
        )
    return JSONResponse({"status": "ok", "message": "Whisper installed.", "output": output})

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    word_timestamps: bool = Form(False),
    output_format: str = Form("txt"),
    device: str = Form("auto"),
    model: str = Form(None),
    whisperps_model: str = Form(None),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    if output_format not in {"txt", "srt", "vtt", "json"}:
        raise HTTPException(status_code=400, detail="Unsupported output format")
    chosen_model = model.strip() if model else DEFAULT_MODEL
    if chosen_model not in MODEL_OPTIONS:
        chosen_model = DEFAULT_MODEL

    job_id = uuid.uuid4().hex
    safe_name = safe_filename(file.filename)
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input_{safe_name}"

    size = 0
    try:
        with input_path.open("wb") as out_file:
            while True:
                chunk = await file.read(4 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large")
                out_file.write(chunk)
    finally:
        await file.close()

    now = time.time()
    job = Job(
        job_id=job_id,
        created_at=now,
        updated_at=now,
        status="queued",
        progress=0,
        progress_mode="indeterminate",
        message="Queued - waiting for worker",
        input_path=str(input_path),
        config={
            "language": language.strip().lower() if language else "auto",
            "word_timestamps": word_timestamps,
            "output_format": output_format,
            "device": device.strip().lower() if device else "auto",
            "model": chosen_model,
            "whisperps_model": whisperps_model.strip() if whisperps_model else WHISPERPS_MODEL_FILE,
        },
    )
    async with jobs_lock:
        jobs[job_id] = job

    log(job_id, f"Upload complete: {input_path} ({size} bytes)")
    job_queue_order.append(job_id)
    refresh_queue_positions()
    await job_queue.put(job_id)
    return JSONResponse({"job_id": job_id})


@app.get("/events/{job_id}")
async def events(job_id: str) -> StreamingResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    queue: asyncio.Queue = asyncio.Queue(maxsize=20)
    job.listeners.append(queue)

    async def event_generator():
        yield f"data: {json.dumps(job_public_state(job))}\n\n"
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        finally:
            if queue in job.listeners:
                job.listeners.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/result/{job_id}")
async def result(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if job:
        payload = {
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "error": job.error,
            "preview": job.text,
            "audio_info": job.audio_info,
            "duration_sec": job.duration_sec,
            "available_formats": list(job.outputs.keys()),
        }
        return JSONResponse(payload)
    meta = load_job_metadata(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Job not found")
    payload = {
        "job_id": job_id,
        "status": meta.get("status"),
        "message": meta.get("message"),
        "error": meta.get("error"),
        "preview": meta.get("preview"),
        "audio_info": meta.get("audio_info"),
        "duration_sec": meta.get("duration_sec"),
        "available_formats": meta.get("available_formats", []),
    }
    return JSONResponse(payload)


@app.get("/debug/{job_id}")
async def debug_job(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    elapsed = None
    if job.ffmpeg_started_at:
        elapsed = time.time() - job.ffmpeg_started_at
    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "progress": job.progress,
        "progress_mode": job.progress_mode,
        "error": job.error,
        "ffmpeg_cmd": job.ffmpeg_cmd,
        "ffmpeg_started_at": job.ffmpeg_started_at,
        "ffmpeg_pid": job.ffmpeg_pid,
        "ffmpeg_elapsed_sec": elapsed,
        "ffmpeg_last_progress": job.ffmpeg_last_progress,
        "ffmpeg_exit_code": job.ffmpeg_exit_code,
        "ffmpeg_error": job.ffmpeg_error,
        "ffmpeg_stderr_tail": job.ffmpeg_stderr_tail,
        "ffmpeg_wav_size_kb": job.ffmpeg_wav_size_kb,
        "ffmpeg_last_activity": job.ffmpeg_last_activity,
        "ffmpeg_stdout_tail": job.ffmpeg_stdout_tail,
        "wav_exists": Path(job.audio_path).exists() if job.audio_path else False,
        "input_path": job.input_path,
        "audio_path": job.audio_path,
        "transcribe_pid": job.transcribe_pid,
        "transcribe_last_activity": job.transcribe_last_activity,
        "transcribe_stdout_tail": job.transcribe_stdout_tail,
        "transcribe_stderr_tail": job.transcribe_stderr_tail,
    }
    return JSONResponse(payload)


@app.get("/download/{job_id}")
async def download(job_id: str, format: str = "txt") -> FileResponse:
    job = jobs.get(job_id)
    path = None
    if job and format in job.outputs:
        path = job.outputs[format]
    else:
        meta = load_job_metadata(job_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Job not found")
        outputs = meta.get("outputs") or {}
        path = outputs.get(format)
    if not path:
        raise HTTPException(status_code=400, detail="Format not available")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output not found")
    filename = f"transcript_{job_id}.{format}"
    return FileResponse(path, filename=filename)


@app.post("/cancel/{job_id}")
async def cancel(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancel_event.set()
    if job_id in job_queue_order:
        job_queue_order.remove(job_id)
        refresh_queue_positions()
    update_job(
        job_id,
        status="cancelled",
        progress=0,
        progress_mode="determinate",
        message="Cancelled",
        queue_position=None,
    )
    return JSONResponse({"status": "cancelled"})
