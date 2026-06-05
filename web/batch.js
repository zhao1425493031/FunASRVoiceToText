(function () {
  const apiKey = window.__BATCH_API_KEY__ || "";
  const STAGE_LABELS = {
    upload: "上传完成",
    queued: "排队中",
    start: "开始处理",
    preprocess: "加载音频",
    asr: "语音转文字",
    diarization: "说话人分离",
    align: "时间轴对齐",
    export: "导出结果",
    done: "完成",
    error: "失败",
  };

  const fileInput = document.getElementById("audioFile");
  const fileName = document.getElementById("fileName");
  const pickBtn = document.getElementById("pickBtn");
  const startBtn = document.getElementById("startBtn");
  const progressSection = document.getElementById("progressSection");
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");
  const resultSection = document.getElementById("resultSection");
  const resultText = document.getElementById("resultText");
  const resultMeta = document.getElementById("resultMeta");
  const copyBtn = document.getElementById("copyBtn");
  const copyHint = document.getElementById("copyHint");
  const errorText = document.getElementById("errorText");

  let selectedFile = null;
  let pollTimer = null;

  function headers() {
    const h = {};
    if (apiKey) h["X-API-Key"] = apiKey;
    return h;
  }

  function showError(msg) {
    errorText.hidden = !msg;
    errorText.textContent = msg || "";
  }

  function setProgress(pct, stage) {
    progressSection.hidden = false;
    progressFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    const label = STAGE_LABELS[stage] || stage || "处理中";
    progressText.textContent = `${label}… ${pct}%`;
  }

  pickBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    selectedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
    fileName.textContent = selectedFile ? selectedFile.name : "未选择文件";
    startBtn.disabled = !selectedFile;
    resultSection.hidden = true;
    showError("");
  });

  async function pollJob(jobId) {
    const res = await fetch(`/api/v1/jobs/${jobId}/status`, { headers: headers() });
    if (!res.ok) throw new Error(`状态查询失败 (${res.status})`);
    return res.json();
  }

  function wait(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  async function runTranscribe() {
    if (!selectedFile) return;
    showError("");
    resultSection.hidden = true;
    copyHint.hidden = true;
    startBtn.disabled = true;
    setProgress(2, "upload");

    const form = new FormData();
    form.append("file", selectedFile);

    let jobId;
    try {
      const res = await fetch("/api/v1/transcribe/async", {
        method: "POST",
        headers: headers(),
        body: form,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `上传失败 (${res.status})`);
      }
      const data = await res.json();
      jobId = data.job_id;
      setProgress(8, "queued");
    } catch (e) {
      showError(e.message || String(e));
      startBtn.disabled = false;
      return;
    }

    while (true) {
      try {
        const st = await pollJob(jobId);
        setProgress(st.progress || 0, st.stage || st.status);

        if (st.status === "completed") {
          resultSection.hidden = false;
          resultText.value = st.text || "";
          const segs = (st.transcript && st.transcript.segments) || [];
          resultMeta.textContent = `共 ${segs.length} 段 · job ${jobId.slice(0, 8)}…`;
          break;
        }
        if (st.status === "failed") {
          throw new Error(st.error || "转写失败");
        }
      } catch (e) {
        showError(e.message || String(e));
        break;
      }
      await wait(600);
    }

    startBtn.disabled = !selectedFile;
  }

  startBtn.addEventListener("click", runTranscribe);

  copyBtn.addEventListener("click", async () => {
    const text = resultText.value;
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      copyHint.hidden = false;
      setTimeout(() => {
        copyHint.hidden = true;
      }, 2000);
    } catch {
      resultText.select();
      document.execCommand("copy");
      copyHint.hidden = false;
    }
  });
})();
