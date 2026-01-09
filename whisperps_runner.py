import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("whisperps_runner")


def is_windows() -> bool:
    return sys.platform.startswith("win")


def detect_amd_gpu() -> Tuple[bool, List[str]]:
    if not is_windows():
        return False, []
    names: List[str] = []
    try:
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        names = []
    amd_names = [name for name in names if re.search(r"(amd|radeon)", name, re.IGNORECASE)]
    return bool(amd_names), amd_names


def whisperps_import_check(psd1_path: Path) -> Tuple[bool, Optional[str]]:
    if not is_windows():
        return False, "Not running on Windows."
    if not psd1_path.exists():
        return False, f"Module manifest missing at {psd1_path}"
    module_dir = psd1_path.parent
    try:
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            (
                f"Unblock-File -Path '{psd1_path}' -ErrorAction SilentlyContinue; "
                f"Get-ChildItem -Path '{module_dir}' -Filter *.dll | Unblock-File -ErrorAction SilentlyContinue; "
                f"Import-Module '{psd1_path}' -Force; Get-Command Transcribe-File | Out-Null"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("WhisperPS import succeeded from %s", psd1_path)
            return True, None
        msg = (result.stderr or "").strip() or (result.stdout or "").strip()
        logger.warning("WhisperPS import failed: %s", msg)
        return False, msg or "Unknown WhisperPS import failure"
    except Exception as exc:
        logger.exception("WhisperPS import raised an exception")
        return False, str(exc)


def whisperps_can_import(psd1_path: Path) -> bool:
    ok, _ = whisperps_import_check(psd1_path)
    return ok


def _clean_adapter_lines(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("WARNING:"):
            continue
        cleaned.append(clean)
    return cleaned


def whisperps_get_adapters(psd1_path: Path) -> List[str]:
    if not is_windows():
        return []
    if not psd1_path.exists():
        return []
    try:
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"Import-Module '{psd1_path}' -Force; Get-Adapters",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return []
        return _clean_adapter_lines(result.stdout.splitlines())
    except Exception:
        return []


def ensure_whisperps_model(filename: str, target_path: Path, status_cb: Optional[Callable[[str], None]] = None) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        logger.info("WhisperPS model already present at %s", target_path)
        return

    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}"
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    if status_cb:
        status_cb(f"Downloading AMD model {filename}...")
    logger.info("Downloading WhisperPS model %s from %s", filename, url)
    downloaded = 0
    total_size = 0
    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as handle:
            length_header = response.headers.get("Content-Length")
            if length_header and length_header.isdigit():
                total_size = int(length_header)
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if status_cb:
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024) if total_size else 0
                    if total_mb:
                        status_cb(
                            f"Downloading AMD model {filename} "
                            f"({downloaded_mb:.1f}/{total_mb:.1f} MB)"
                        )
                    else:
                        status_cb(f"Downloading AMD model {filename} ({downloaded_mb:.1f} MB)")
        os.replace(tmp_path, target_path)
        logger.info("WhisperPS model download finished: %s", target_path)
        if status_cb:
            status_cb(f"AMD model ready: {filename}")
    finally:
        if tmp_path.exists() and not target_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def run_whisperps_transcribe(
    script_path: Path,
    module_manifest_path: Path,
    model_path: Path,
    input_wav_path: Path,
    out_dir: Path,
    language: Optional[str],
    out_format: str,
    adapter_name: Optional[str],
    log_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[int, str, Optional[int]]:
    if not is_windows():
        raise RuntimeError("WhisperPS transcription is only available on Windows")
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-ModuleManifestPath",
        str(module_manifest_path),
        "-ModelPath",
        str(model_path),
        "-InputWavPath",
        str(input_wav_path),
        "-OutDir",
        str(out_dir),
        "-OutFormat",
        out_format,
    ]
    if language:
        cmd.extend(["-Language", language])
    if adapter_name:
        cmd.extend(["-AdapterName", adapter_name])

    logger.info(
        "Launching WhisperPS transcription: script=%s model=%s adapter=%s",
        script_path,
        model_path,
        adapter_name or "auto",
    )
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if log_callback:
        log_callback(f"Started WhisperPS process pid={process.pid} (adapter={adapter_name or 'auto'})")
    tail: List[str] = []

    def stop_process() -> None:
        try:
            if process.poll() is None:
                process.terminate()
                time.sleep(0.5)
                if process.poll() is None:
                    process.kill()
        except Exception:
            pass

    try:
        if process.stdout:
            for line in process.stdout:
                if cancel_event and cancel_event.is_set():
                    stop_process()
                    break
                clean = line.rstrip("\r\n")
                if clean:
                    tail.append(clean)
                    if len(tail) > 200:
                        tail[:] = tail[-200:]
                    if log_callback:
                        log_callback(clean)
        return_code = process.wait()
        logger.info("WhisperPS process exited with code %s", return_code)
        return return_code, "\n".join(tail), process.pid
    finally:
        if cancel_event and cancel_event.is_set():
            stop_process()
        if process.stdout:
            process.stdout.close()
