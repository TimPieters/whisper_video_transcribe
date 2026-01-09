const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileMeta = document.getElementById("fileMeta");
const outputFormat = document.getElementById("outputFormat");
const languageInput = document.getElementById("language");
const deviceSelect = document.getElementById("deviceSelect");
const whisperpsModelSelect = document.getElementById("whisperpsModelSelect");
const whisperpsModelField = document.getElementById("whisperpsModelField");
const wordTimestamps = document.getElementById("wordTimestamps");
const startButton = document.getElementById("startButton");
const cancelButton = document.getElementById("cancelButton");
const statusBadge = document.getElementById("statusBadge");
const statusMessage = document.getElementById("statusMessage");
const audioInfo = document.getElementById("audioInfo");
const errorMessage = document.getElementById("errorMessage");
const progressFill = document.getElementById("progressFill");
const progressLabel = document.getElementById("progressLabel");
const deviceBadge = document.getElementById("deviceBadge");
const preview = document.getElementById("preview");
const downloadButton = document.getElementById("downloadButton");
const debugToggle = document.getElementById("debugToggle");
const debugContent = document.getElementById("debugContent");
const debugMeta = document.getElementById("debugMeta");
const debugLog = document.getElementById("debugLog");
const envBadge = document.getElementById("envBadge");
const envPython = document.getElementById("envPython");
const envWhisper = document.getElementById("envWhisper");
const envFfmpeg = document.getElementById("envFfmpeg");
const envFfprobe = document.getElementById("envFfprobe");
const envAmd = document.getElementById("envAmd");
const installButton = document.getElementById("installButton");
const installHint = document.getElementById("installHint");
const installLog = document.getElementById("installLog");
const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
const historyList = document.getElementById("historyList");
const refreshHistory = document.getElementById("refreshHistory");

const steps = Array.from(document.querySelectorAll(".step"));
const stepOrder = ["uploading", "extracting", "loading_model", "transcribing", "packaging", "done"];
const stageMap = { queued: "extracting", cancelled: "extracting", error: "extracting" };

let currentFile = null;
let jobId = null;
let eventSource = null;
let uploadRequest = null;
let lastServerStage = null;
let debugTimer = null;
let historyTimer = null;

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function formatElapsed(seconds) {
  if (seconds === null || seconds === undefined) return "unknown";
  const total = Math.max(0, Math.floor(seconds));
  const hours = String(Math.floor(total / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const secs = String(total % 60).padStart(2, "0");
  return `${hours}:${minutes}:${secs}`;
}

function formatDateTime(seconds) {
  if (!seconds) return "unknown";
  const date = new Date(seconds * 1000);
  return date.toLocaleString();
}

function statusVariant(status) {
  if (status === "done") return "success";
  if (status === "error" || status === "cancelled") return "error";
  if (status === "queued") return "warn";
  return "warn";
}

function setFile(file) {
  currentFile = file;
  if (file) {
    fileMeta.textContent = `${file.name} (${formatBytes(file.size)})`;
  } else {
    fileMeta.textContent = "No file selected";
  }
}

function setProgress(percent, label) {
  const safe = Math.max(0, Math.min(100, percent));
  progressFill.classList.remove("indeterminate");
  progressFill.style.width = `${safe}%`;
  progressLabel.textContent = label || "";
}

function setIndeterminate(label) {
  progressFill.classList.add("indeterminate");
  progressFill.style.width = "30%";
  progressLabel.textContent = label || "Working...";
}

function setBadge(text, variant) {
  statusBadge.textContent = text;
  statusBadge.className = "badge";
  if (variant === "error") {
    statusBadge.style.background = "rgba(181, 101, 118, 0.2)";
    statusBadge.style.color = "#6b1b2d";
  } else {
    statusBadge.style.background = "rgba(15, 76, 92, 0.1)";
    statusBadge.style.color = "var(--accent)";
  }
}

function setDeviceBadge(device) {
  if (!deviceBadge) return;
  if (device === "amd") {
    deviceBadge.textContent = "AMD via WhisperPS";
    deviceBadge.classList.remove("hidden");
    deviceBadge.classList.add("amd", "chip");
  } else {
    deviceBadge.classList.add("hidden");
    deviceBadge.textContent = "";
  }
}

function toggleWhisperpsModel(show) {
  if (!whisperpsModelField) return;
  whisperpsModelField.classList.toggle("hidden", !show);
}

function resetSteps() {
  steps.forEach((step) => {
    step.classList.remove("active", "done", "error");
  });
}

function updateSteps(currentStep, isError) {
  const currentIndex = stepOrder.indexOf(currentStep);
  steps.forEach((step, index) => {
    step.classList.remove("active", "done", "error");
    if (currentIndex >= 0 && index < currentIndex) {
      step.classList.add("done");
    } else if (index === currentIndex) {
      step.classList.add(isError ? "error" : "active");
    }
  });
}

function resetUI() {
  setBadge("Idle");
  setDeviceBadge(null);
  resetSteps();
  setProgress(0, "Waiting for upload");
  statusMessage.textContent = "Idle";
  audioInfo.textContent = "";
  errorMessage.textContent = "";
  preview.textContent = "Waiting for a finished transcript...";
  downloadButton.disabled = true;
  toggleWhisperpsModel(false);
}

function closeEventSource() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

function connectEvents(id) {
  closeEventSource();
  eventSource = new EventSource(`/events/${id}`);
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleServerUpdate(data);
  };
  eventSource.onerror = () => {
    setBadge("Disconnected", "error");
  };
}

function stopDebugTimer() {
  if (debugTimer) {
    clearInterval(debugTimer);
    debugTimer = null;
  }
}

function fetchDebug() {
  if (!jobId) return;
  fetch(`/debug/${jobId}`)
    .then((resp) => resp.json())
    .then((data) => {
      const lines = [];
      if (data.ffmpeg_cmd) {
        lines.push(`ffmpeg cmd: ${data.ffmpeg_cmd}`);
      }
      if (data.ffmpeg_elapsed_sec !== null && data.ffmpeg_elapsed_sec !== undefined) {
        lines.push(`ffmpeg elapsed: ${formatElapsed(data.ffmpeg_elapsed_sec)}`);
      }
      if (data.ffmpeg_last_progress) {
        lines.push(`ffmpeg progress: ${data.ffmpeg_last_progress}`);
      }
      if (data.ffmpeg_error) {
        lines.push(`ffmpeg error: ${data.ffmpeg_error}`);
      }
      if (data.ffmpeg_exit_code !== null && data.ffmpeg_exit_code !== undefined) {
        lines.push(`ffmpeg exit code: ${data.ffmpeg_exit_code}`);
      }
      if (data.ffmpeg_wav_size_kb !== null && data.ffmpeg_wav_size_kb !== undefined) {
        lines.push(`wav size: ${data.ffmpeg_wav_size_kb} KB`);
      }
      if (data.ffmpeg_last_activity) {
        lines.push(`last activity: ${new Date(data.ffmpeg_last_activity * 1000).toLocaleTimeString()}`);
      }
      if (data.wav_exists !== undefined) {
        lines.push(`wav exists: ${data.wav_exists}`);
      }
      if (data.transcribe_pid) {
        lines.push(`whisperps pid: ${data.transcribe_pid}`);
      }
      if (data.transcribe_last_activity) {
        lines.push(`whisperps last activity: ${new Date(data.transcribe_last_activity * 1000).toLocaleTimeString()}`);
      }
      if (data.ffmpeg_stdout_tail) {
        lines.push("--- stdout tail ---");
      }
      debugMeta.textContent = lines.join("\n") || "No debug metadata yet.";
      const stderrText = data.ffmpeg_stderr_tail || "No ffmpeg stderr output yet.";
      const stdoutText = data.ffmpeg_stdout_tail || "";
      const whisperpsText = data.transcribe_stdout_tail ? `--- WhisperPS ---\n${data.transcribe_stdout_tail}` : "";
      debugLog.textContent = [stderrText, stdoutText, whisperpsText].filter(Boolean).join("\n");
    })
    .catch(() => {
      debugMeta.textContent = "Failed to load debug info.";
    });
}

function handleServerUpdate(data) {
  if (data.status && data.status !== "error" && data.status !== "cancelled") {
    lastServerStage = data.status;
  }

  setDeviceBadge(data.device);

  const rawStage = data.status || "done";
  const stage = stageMap[rawStage] || rawStage;

  updateSteps(stage, data.status === "error");
  let badgeText = data.status ? data.status.toUpperCase() : "RUNNING";
  if (data.status === "queued" && data.queue_position) {
    badgeText = `QUEUED #${data.queue_position}`;
  }
  setBadge(badgeText, data.status === "error" ? "error" : null);

  if (data.message) {
    statusMessage.textContent = data.message;
  }
  if (data.audio_info) {
    audioInfo.textContent = data.audio_info;
  }
  if (data.error) {
    errorMessage.textContent = data.error;
  } else if (data.status && data.status !== "error") {
    errorMessage.textContent = "";
  }

  if (data.progress_mode === "indeterminate" || data.progress === null) {
    const baseLabel = data.message || "Working...";
    setIndeterminate(baseLabel);
  } else if (typeof data.progress === "number") {
    setProgress(data.progress, `${data.message || "Working"} (${data.progress.toFixed(0)}%)`);
  }

  if (data.status === "done") {
    setProgress(100, "Complete");
    fetchResult();
  }

  if (data.status === "error" || data.status === "cancelled") {
    startButton.disabled = false;
    cancelButton.disabled = true;
    closeEventSource();
    stopDebugTimer();
  }
}

function fetchResult() {
  fetch(`/result/${jobId}`)
    .then((resp) => resp.json())
    .then((data) => {
      if (data.preview) {
        preview.textContent = data.preview;
      } else {
        preview.textContent = "Transcript is ready. Download to view full output.";
      }
      downloadButton.disabled = false;
      startButton.disabled = false;
      cancelButton.disabled = true;
      closeEventSource();
    })
    .catch(() => {
      preview.textContent = "Transcript finished, but preview failed to load.";
      downloadButton.disabled = false;
      startButton.disabled = false;
      cancelButton.disabled = true;
      closeEventSource();
    });
}

function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("output_format", outputFormat.value);
    const languageVal = languageInput.value.trim() || "auto";
    formData.append("language", languageVal);
    formData.append("device", deviceSelect.value);
    if (deviceSelect.value === "amd" && whisperpsModelSelect && whisperpsModelSelect.value) {
      formData.append("whisperps_model", whisperpsModelSelect.value);
    }
    formData.append("word_timestamps", wordTimestamps.checked ? "true" : "false");

    const xhr = new XMLHttpRequest();
    uploadRequest = xhr;
    xhr.open("POST", "/upload");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = (event.loaded / event.total) * 100;
        updateSteps("uploading", false);
        setBadge("UPLOADING");
        setProgress(percent, `Uploading ${percent.toFixed(0)}%`);
        statusMessage.textContent = "Uploading video";
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const response = JSON.parse(xhr.responseText);
        resolve(response.job_id);
      } else {
        reject(new Error(xhr.responseText || "Upload failed"));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed"));
    xhr.send(formData);
  });
}

function startJob() {
  if (!currentFile) {
    errorMessage.textContent = "Please select a video file.";
    return;
  }
  if (deviceSelect.value === "amd") {
    const langVal = languageInput.value.trim();
    if (!langVal || langVal.toLowerCase() === "auto") {
      errorMessage.textContent = "Set a language (e.g. nl) when using AMD/WhisperPS.";
      return;
    }
  }
  resetUI();
  startButton.disabled = true;
  cancelButton.disabled = false;
  uploadFile(currentFile)
    .then((id) => {
      jobId = id;
      setBadge("QUEUED");
      statusMessage.textContent = "Queued for processing";
      updateSteps("extracting", false);
      setIndeterminate("Queued - waiting for worker");
      connectEvents(jobId);
      if (!debugContent.classList.contains("hidden")) {
        fetchDebug();
        stopDebugTimer();
        debugTimer = setInterval(fetchDebug, 2000);
      }
    })
    .catch((err) => {
      errorMessage.textContent = err.message || "Upload failed";
      startButton.disabled = false;
      cancelButton.disabled = true;
    });
}

function cancelJob() {
  if (uploadRequest && !jobId) {
    uploadRequest.abort();
    setBadge("CANCELLED", "error");
    startButton.disabled = false;
    cancelButton.disabled = true;
    return;
  }
  if (!jobId) return;
  fetch(`/cancel/${jobId}`, { method: "POST" })
    .then(() => {
      setBadge("CANCELLED", "error");
      statusMessage.textContent = "Cancelled";
      updateSteps(lastServerStage || "extracting", true);
      setProgress(0, "Cancelled");
      startButton.disabled = false;
      cancelButton.disabled = true;
      closeEventSource();
      stopDebugTimer();
    })
    .catch(() => {
      errorMessage.textContent = "Failed to cancel job.";
    });
}

function clipText(text, maxLength = 180) {
  if (!text) return "";
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= maxLength) return clean;
  return `${clean.slice(0, maxLength)}…`;
}

function renderHistory(jobs) {
  if (!historyList) return;
  historyList.innerHTML = "";
  if (!jobs || jobs.length === 0) {
    historyList.innerHTML = `
      <div class="empty-state">
        <div class="empty-title">No jobs yet</div>
        <div class="empty-subtitle">Start a transcription to see it appear here.</div>
      </div>
    `;
    return;
  }
  jobs.forEach((job) => {
    const row = document.createElement("div");
    row.className = "history-row";

    const main = document.createElement("div");
    main.className = "history-main";

    const title = document.createElement("div");
    title.className = "history-title";
    title.textContent = job.input_filename || `Job ${job.job_id}`;
    main.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "history-subtitle";
    const updatedLabel = formatDateTime(job.updated_at);
    const durationLabel = job.duration_sec ? ` • ${formatElapsed(job.duration_sec)} audio` : "";
    meta.textContent = `Job ${job.job_id?.slice(0, 8) || "?"} • Updated ${updatedLabel}${durationLabel}`;
    main.appendChild(meta);

    const tags = document.createElement("div");
    tags.className = "history-tags";
    const statusChip = document.createElement("span");
    statusChip.className = `chip ${statusVariant(job.status)}`;
    statusChip.textContent = (job.status || "unknown").toUpperCase();
    tags.appendChild(statusChip);
    (job.available_formats || []).forEach((fmt) => {
      const fmtChip = document.createElement("span");
      fmtChip.className = "chip format";
      fmtChip.textContent = fmt.toUpperCase();
      tags.appendChild(fmtChip);
    });
    main.appendChild(tags);

    const snippetText = job.preview || job.message || job.error;
    if (snippetText) {
      const snippet = document.createElement("div");
      snippet.className = "history-subtitle";
      snippet.textContent = clipText(snippetText);
      main.appendChild(snippet);
    }

    const downloads = document.createElement("div");
    downloads.className = "history-downloads";
    if (job.available_formats && job.available_formats.length > 0) {
      job.available_formats.forEach((fmt) => {
        const btn = document.createElement("button");
        btn.className = "download-pill";
        btn.textContent = `Download ${fmt.toUpperCase()}`;
        btn.addEventListener("click", () => {
          window.location.href = `/download/${job.job_id}?format=${encodeURIComponent(fmt)}`;
        });
        downloads.appendChild(btn);
      });
    } else {
      const pending = document.createElement("div");
      pending.className = "history-subtitle";
      pending.textContent = "Outputs not available yet.";
      downloads.appendChild(pending);
    }

    row.appendChild(main);
    row.appendChild(downloads);
    historyList.appendChild(row);
  });
}

function fetchHistory() {
  fetch("/jobs")
    .then((resp) => resp.json())
    .then((data) => renderHistory(data.jobs || []))
    .catch(() => {
      if (!historyList) return;
      historyList.innerHTML = `
        <div class="empty-state">
          <div class="empty-title">Unable to load history</div>
          <div class="empty-subtitle">Check the server and try again.</div>
        </div>
      `;
    });
}

function stopHistoryTimer() {
  if (historyTimer) {
    clearInterval(historyTimer);
    historyTimer = null;
  }
}

function switchTab(target) {
  tabButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === target);
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.tab === target);
  });
  if (target === "history") {
    fetchHistory();
    stopHistoryTimer();
    historyTimer = setInterval(fetchHistory, 6000);
  } else {
    stopHistoryTimer();
  }
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const files = event.dataTransfer.files;
  if (files && files[0]) {
    fileInput.files = files;
    setFile(files[0]);
  }
});

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  setFile(file);
});

startButton.addEventListener("click", startJob);
cancelButton.addEventListener("click", cancelJob);
downloadButton.addEventListener("click", () => {
  if (!jobId) return;
  const format = outputFormat.value;
  window.location.href = `/download/${jobId}?format=${encodeURIComponent(format)}`;
});

function updateEnvRow(element, ok, label, detail) {
  element.textContent = detail ? `${label}: ${detail}` : label;
  element.classList.remove("ok", "warn");
  element.classList.add(ok ? "ok" : "warn");
}

function updateEnvBadge(ok) {
  envBadge.textContent = ok ? "Ready" : "Needs setup";
  envBadge.style.background = ok ? "rgba(15, 76, 92, 0.1)" : "rgba(181, 101, 118, 0.2)";
  envBadge.style.color = ok ? "var(--accent)" : "#6b1b2d";
}

function populateDevices(data) {
  const opts = data.device_options || ["auto", "cpu"];
  deviceSelect.innerHTML = "";
  opts.forEach((opt) => {
    const option = document.createElement("option");
    option.value = opt;
    if (opt === "cuda") {
      option.textContent = "CUDA GPU";
    } else if (opt === "dml") {
      option.textContent = "DirectML";
    } else if (opt === "amd") {
      option.textContent = "AMD GPU (WhisperPS)";
      if (!data.whisperps_available) {
        option.disabled = true;
        option.title = "WhisperPS import failed (vendored module missing or broken).";
      }
    } else {
      option.textContent = opt.toUpperCase();
    }
    deviceSelect.appendChild(option);
  });
  if (data.cuda_available) {
    deviceSelect.value = "cuda";
  } else if (data.has_amd_gpu && data.whisperps_available && opts.includes("amd")) {
    deviceSelect.value = "amd";
  } else if (opts.includes("dml") && data.dml_available) {
    deviceSelect.value = "dml";
  } else {
    deviceSelect.value = "auto";
  }

  if (whisperpsModelSelect) {
    whisperpsModelSelect.innerHTML = "";
    const models = data.whisperps_models || [];
    models.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      whisperpsModelSelect.appendChild(option);
    });
    if (data.whisperps_model_default && models.includes(data.whisperps_model_default)) {
      whisperpsModelSelect.value = data.whisperps_model_default;
    }
    toggleWhisperpsModel(deviceSelect.value === "amd");
  }
}

function fetchPreflight() {
  fetch("/preflight")
    .then((resp) => resp.json())
    .then((data) => {
      populateDevices(data);
      updateEnvRow(envPython, data.python_supported, "Python", data.python_version);
      const deviceLabel = data.cuda_available && data.cuda_gpu_name
        ? `CUDA available (${data.cuda_gpu_name})`
        : data.cuda_available
          ? "CUDA available"
          : data.has_amd_gpu
            ? "AMD GPU detected"
            : data.dml_available
              ? "DirectML available (AMD/Intel) - experimental fallback to CPU in faster-whisper"
              : "GPU not detected (using CPU)";
      updateEnvRow(envWhisper, data.whisper, "Whisper", data.whisper ? "installed" : "missing");
      envPython.textContent = `Python: ${data.python_version} - ${deviceLabel}`;
      updateEnvRow(envFfmpeg, data.ffmpeg, "ffmpeg", data.ffmpeg ? "available" : "missing");
      updateEnvRow(envFfprobe, data.ffprobe, "ffprobe", data.ffprobe ? "available" : "missing");
      if (envAmd) {
        const amdLabel = data.has_amd_gpu
          ? data.whisperps_available
            ? "AMD GPU detected (WhisperPS ready)"
            : "AMD GPU detected (WhisperPS unavailable)"
          : "AMD GPU not detected";
        updateEnvRow(envAmd, data.has_amd_gpu && data.whisperps_available, "AMD GPU", amdLabel);
      }
      if (whisperpsModelSelect) {
        whisperpsModelSelect.disabled = !(data.has_amd_gpu && data.whisperps_available);
      }

      const ok = data.python_supported && data.whisper && data.ffmpeg;
      updateEnvBadge(ok);

      const hints = [];
      if (!data.python_supported) {
        hints.push("Python 3.10-3.12 required for Whisper turbo.");
      }
      if (!data.ffmpeg) {
        hints.push("ffmpeg missing. Install it and add to PATH.");
      }
      if (!data.ffprobe) {
        hints.push("ffprobe missing (optional, improves progress).");
      }
      if (!data.whisper && data.python_supported) {
        hints.push("Whisper is missing. Install it into this environment.");
      }
      if (data.has_amd_gpu && !data.whisperps_available) {
        hints.push("WhisperPS import failed (vendored module missing or broken).");
      }
      if (data.has_amd_gpu && data.whisperps_available && !data.whisperps_model_present) {
        hints.push("AMD WhisperPS model will auto-download on first use.");
      }
      installHint.textContent = hints.length ? hints.join(" ") : "Environment ready.";

      if (!data.python_supported || data.whisper) {
        installButton.disabled = true;
      } else {
        installButton.disabled = false;
      }

      if (data.whisper_error && !data.whisper) {
        installLog.style.display = "block";
        installLog.textContent = data.whisper_error;
      }
    })
    .catch(() => {
      installHint.textContent = "Unable to check environment status.";
    });
}

installButton.addEventListener("click", () => {
  installButton.disabled = true;
  installLog.style.display = "block";
  installLog.textContent = "Installing Whisper and dependencies...";
  fetch("/install", { method: "POST" })
    .then((resp) => resp.json().then((data) => ({ ok: resp.ok, data })))
    .then(({ ok, data }) => {
      installLog.textContent = data.output || data.message || "Install finished.";
      if (!ok) {
        installHint.textContent = data.message || "Install failed.";
      }
      fetchPreflight();
    })
    .catch(() => {
      installLog.textContent = "Install failed. Check your network and Python version.";
      installButton.disabled = false;
    });
});

debugToggle.addEventListener("click", () => {
  debugContent.classList.toggle("hidden");
  if (!debugContent.classList.contains("hidden")) {
    debugToggle.textContent = "Hide debug";
    fetchDebug();
    stopDebugTimer();
    debugTimer = setInterval(fetchDebug, 2000);
  } else {
    debugToggle.textContent = "Show debug";
    stopDebugTimer();
  }
});

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

if (deviceSelect) {
  deviceSelect.addEventListener("change", () => {
    toggleWhisperpsModel(deviceSelect.value === "amd");
  });
}

if (refreshHistory) {
  refreshHistory.addEventListener("click", () => {
    fetchHistory();
  });
}

switchTab("transcribe");
resetUI();
fetchPreflight();
