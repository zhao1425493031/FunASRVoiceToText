/**
 * Meeting v2 Web client: multi-speaker subtitles + post-meeting LLM summary
 */

(function () {
  const TARGET_SR = 16000;
  const CHUNK_SAMPLES = 9600;
  const PING_INTERVAL_MS = 30000;
  const SESSION_READY_TIMEOUT_MS = 12000;
  const SUMMARY_WAIT_TIMEOUT_MS = 180000;

  const statusText = document.getElementById("statusText");
  const hintBox = document.getElementById("hintBox");
  const subtitleList = document.getElementById("subtitleList");
  const summaryPanel = document.getElementById("summaryPanel");
  const summaryStatus = document.getElementById("summaryStatus");
  const summaryContent = document.getElementById("summaryContent");
  const btnCopySummary = document.getElementById("btnCopySummary");
  const btnDownloadSummary = document.getElementById("btnDownloadSummary");
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnClear = document.getElementById("btnClear");

  let state = "idle";
  let ws = null;
  let audioContext = null;
  let mediaStream = null;
  let processor = null;
  let source = null;
  let pcmBuffer = [];
  let pingTimer = null;
  let sessionReadyTimeoutId = null;
  let summaryWaitTimeoutId = null;
  let waitingForSummary = false;
  let currentSummaryMarkdown = "";
  let currentSessionId = "";
  const linesBySegId = new Map();
  const PARTICIPANT_STORAGE_KEY = "voicetotext_meeting_participant_id";

  function ensureParticipantId() {
    try {
      let id = sessionStorage.getItem(PARTICIPANT_STORAGE_KEY);
      if (!id) {
        id =
          typeof crypto !== "undefined" && crypto.randomUUID
            ? crypto.randomUUID()
            : `participant-${Date.now()}`;
        sessionStorage.setItem(PARTICIPANT_STORAGE_KEY, id);
      }
      return id;
    } catch (_) {
      return `participant-${Date.now()}`;
    }
  }

  function apiKeyFromQuery() {
    return new URLSearchParams(location.search).get("key") || "";
  }

  function resolvedApiKey() {
    return apiKeyFromQuery() || window.__MEETING_API_KEY__ || "";
  }

  function isSingleSpeakerMode() {
    return (window.__MEETING_SPK_MODE__ || "single") === "single";
  }

  function speakerPrefix(msg) {
    if (isSingleSpeakerMode()) return "";
    const spk = msg.speaker_id != null ? msg.speaker_id : 0;
    return `[話者${spk + 1}] `;
  }

  function ensureUrlHasKey() {
    const key = resolvedApiKey();
    if (!key || apiKeyFromQuery()) return;
    const url = new URL(location.href);
    url.searchParams.set("key", key);
    location.replace(url.toString());
  }

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws/meeting/asr`;
  }

  function setStatusText(text) {
    statusText.textContent = text;
  }

  function setHint(text, warn) {
    if (!hintBox) return;
    hintBox.textContent = text;
    hintBox.classList.toggle("hint-warn", !!warn);
  }

  function setState(next, detail) {
    state = next;
    document.body.className = `state-${next}`;
    const labels = {
      idle: "待機",
      connecting: "接続中…",
      listening: "字幕認識中…",
      finalizing: "纪要生成中…",
      error: "エラー",
    };
    setStatusText(detail || labels[next] || next);
    btnStart.disabled = next === "listening" || next === "connecting" || next === "finalizing";
    btnStop.disabled = next !== "listening" && next !== "error";
  }

  function clearSessionReadyTimeout() {
    if (sessionReadyTimeoutId !== null) {
      clearTimeout(sessionReadyTimeoutId);
      sessionReadyTimeoutId = null;
    }
  }

  function clearSummaryWaitTimeout() {
    if (summaryWaitTimeoutId !== null) {
      clearTimeout(summaryWaitTimeoutId);
      summaryWaitTimeoutId = null;
    }
  }

  function showSummaryPanel() {
    if (summaryPanel) summaryPanel.classList.remove("hidden");
  }

  function resetSummaryPanel() {
    currentSummaryMarkdown = "";
    if (summaryPanel) summaryPanel.classList.add("hidden");
    if (summaryStatus) summaryStatus.textContent = "";
    if (summaryContent) summaryContent.innerHTML = "";
    if (btnCopySummary) btnCopySummary.disabled = true;
    if (btnDownloadSummary) btnDownloadSummary.disabled = true;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderMarkdownBasic(md) {
    const lines = String(md || "").split("\n");
    const html = [];
    let inList = false;
    for (const line of lines) {
      if (/^##\s+/.test(line)) {
        if (inList) {
          html.push("</ul>");
          inList = false;
        }
        html.push(`<h3>${escapeHtml(line.replace(/^##\s+/, ""))}</h3>`);
      } else if (/^[-*]\s+/.test(line)) {
        if (!inList) {
          html.push("<ul>");
          inList = true;
        }
        html.push(`<li>${escapeHtml(line.replace(/^[-*]\s+/, ""))}</li>`);
      } else if (line.trim()) {
        if (inList) {
          html.push("</ul>");
          inList = false;
        }
        html.push(`<p>${escapeHtml(line)}</p>`);
      }
    }
    if (inList) html.push("</ul>");
    return html.join("");
  }

  function renderSummary(msg) {
    showSummaryPanel();
    const summary = msg.summary || {};
    const md = summary.markdown || "";
    currentSummaryMarkdown = md;
    if (summaryStatus) {
      if (msg.status === "ok") {
        summaryStatus.textContent = summary.title
          ? `${summary.title}${summary.overview ? " — " + summary.overview : ""}`
          : "纪要已生成";
        summaryStatus.classList.remove("summary-error");
      } else {
        summaryStatus.textContent = msg.error || "纪要生成失败";
        summaryStatus.classList.add("summary-error");
      }
    }
    if (summaryContent) {
      if (md) {
        summaryContent.innerHTML = renderMarkdownBasic(md);
      } else if (summary.overview) {
        summaryContent.innerHTML = `<p>${escapeHtml(summary.overview)}</p>`;
      } else {
        summaryContent.innerHTML = "";
      }
    }
    const hasMd = !!md;
    if (btnCopySummary) btnCopySummary.disabled = !hasMd;
    if (btnDownloadSummary) btnDownloadSummary.disabled = !hasMd;
  }

  function finishSummaryWait() {
    waitingForSummary = false;
    clearSummaryWaitTimeout();
    stopPing();
    stopAudio();
    if (ws) {
      try {
        if (ws.readyState === WebSocket.OPEN) ws.close();
      } catch (_) {
        /* ignore */
      }
      ws = null;
    }
    setState("idle");
    updateKeyHint();
  }

  function releaseResources(skipEnd) {
    clearSessionReadyTimeout();
    clearSummaryWaitTimeout();
    stopPing();
    stopAudio();
    if (ws) {
      try {
        if (!skipEnd && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "end", is_speaking: false, skip_summary: false }));
        }
      } catch (_) {
        /* ignore */
      }
      if (!waitingForSummary) {
        try {
          if (ws.readyState === WebSocket.OPEN) ws.close();
        } catch (_) {
          /* ignore */
        }
        ws = null;
      }
    }
  }

  function showError(message) {
    waitingForSummary = false;
    releaseResources(true);
    setState("error", message.split("\n")[0]);
    setHint(message, true);
  }

  function stopSession() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      finishSummaryWait();
      return;
    }
    stopAudio();
    stopPing();
    waitingForSummary = true;
    setState("finalizing", "纪要生成中…");
    showSummaryPanel();
    if (summaryStatus) {
      summaryStatus.textContent = "纪要生成中…";
      summaryStatus.classList.remove("summary-error");
    }
    if (summaryContent) summaryContent.innerHTML = "";
    if (btnCopySummary) btnCopySummary.disabled = true;
    if (btnDownloadSummary) btnDownloadSummary.disabled = true;
    try {
      ws.send(JSON.stringify({ type: "end", is_speaking: false, skip_summary: false }));
    } catch (err) {
      showError(err.message || "終了メッセージ送信失敗");
      return;
    }
    clearSummaryWaitTimeout();
    summaryWaitTimeoutId = setTimeout(() => {
      if (!waitingForSummary) return;
      if (summaryStatus) {
        summaryStatus.textContent = "纪要生成超时，请稍后通过 API 重试";
        summaryStatus.classList.add("summary-error");
      }
      finishSummaryWait();
    }, SUMMARY_WAIT_TIMEOUT_MS);
  }

  function resampleTo16k(float32, inputRate) {
    if (inputRate === TARGET_SR) return float32;
    const ratio = inputRate / TARGET_SR;
    const outLen = Math.floor(float32.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const idx0 = Math.floor(srcIdx);
      const idx1 = Math.min(idx0 + 1, float32.length - 1);
      const frac = srcIdx - idx0;
      out[i] = float32[idx0] * (1 - frac) + float32[idx1] * frac;
    }
    return out;
  }

  function floatToInt16(float32) {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16;
  }

  function sendPcmChunk(samples) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(new Int16Array(samples).buffer);
  }

  function appendSamples(float32) {
    const int16 = floatToInt16(float32);
    for (let i = 0; i < int16.length; i++) pcmBuffer.push(int16[i]);
    while (pcmBuffer.length >= CHUNK_SAMPLES && ws && ws.readyState === WebSocket.OPEN) {
      sendPcmChunk(pcmBuffer.splice(0, CHUNK_SAMPLES));
    }
  }

  function upsertSubtitle(msg) {
    const segId = msg.seg_id || `tmp-${Date.now()}`;
    let row = linesBySegId.get(segId);
    if (!row) {
      row = document.createElement("div");
      row.className = "final-item";
      row.dataset.segId = segId;
      subtitleList.appendChild(row);
      linesBySegId.set(segId, row);
    }
    const draft = msg.type === "partial" ? " …" : "";
    row.textContent = speakerPrefix(msg) + (msg.text || "") + draft;
    if (msg.type === "final") {
      row.classList.add("subtitle-final");
      row.classList.remove("subtitle-partial");
    } else {
      row.classList.add("subtitle-partial");
      row.classList.remove("subtitle-final");
    }
    subtitleList.scrollTop = subtitleList.scrollHeight;
  }

  function formatServerError(msg) {
    const code = msg.code || "error";
    const text = msg.message || code;
    if (code === "unauthorized") {
      return `認証失敗: ${text}\nページを再読み込みしてください（?key= は自動付与されます）`;
    }
    if (code === "backend_unavailable") {
      return `ASR 未就绪: ${text}\nサーバー起動ログで「Meeting server ready」と /ready を確認してください`;
    }
    return `${code}: ${text}`;
  }

  function handleServerMessage(raw, onSessionReady) {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    if (msg.type === "session_ready") {
      clearSessionReadyTimeout();
      if (onSessionReady) onSessionReady();
      return;
    }
    if (msg.type === "pong") return;
    if (msg.type === "summary_progress") {
      if (state === "finalizing" || waitingForSummary) {
        setState("finalizing", "纪要生成中…");
        showSummaryPanel();
        if (summaryStatus) {
          summaryStatus.textContent = "纪要生成中…";
          summaryStatus.classList.remove("summary-error");
        }
      }
      return;
    }
    if (msg.type === "meeting_summary") {
      renderSummary(msg);
      if (waitingForSummary) finishSummaryWait();
      return;
    }
    if (msg.type === "speaker_change") {
      if (state === "listening" && !isSingleSpeakerMode()) {
        setStatusText(`字幕認識中…（話者${(msg.speaker_id || 0) + 1}）`);
      }
      return;
    }
    if ((msg.type === "partial" || msg.type === "final") && msg.text) {
      if (state === "listening") {
        setStatusText("字幕認識中…");
      }
      upsertSubtitle(msg);
    } else if (msg.type === "error") {
      if (waitingForSummary) {
        if (summaryStatus) {
          summaryStatus.textContent = formatServerError(msg);
          summaryStatus.classList.add("summary-error");
        }
        finishSummaryWait();
        return;
      }
      showError(formatServerError(msg));
    }
  }

  function startPing() {
    stopPing();
    pingTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, PING_INTERVAL_MS);
  }

  function stopPing() {
    if (pingTimer) {
      clearInterval(pingTimer);
      pingTimer = null;
    }
  }

  function stopAudio() {
    if (processor) {
      try {
        processor.disconnect();
      } catch (_) {
        /* ignore */
      }
      processor = null;
    }
    if (source) {
      try {
        source.disconnect();
      } catch (_) {
        /* ignore */
      }
      source = null;
    }
    if (audioContext) {
      audioContext.close().catch(() => {});
      audioContext = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    pcmBuffer = [];
  }

  function ensureMicrophoneApi() {
    const secure =
      window.isSecureContext ||
      location.protocol === "https:" ||
      location.hostname === "localhost" ||
      location.hostname === "127.0.0.1";
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (!secure) {
        throw new Error("会議字幕には HTTPS が必要です（https://<IP>:8766/meeting）");
      }
      throw new Error("マイク API が利用できません");
    }
  }

  async function startAudio() {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
      },
      video: false,
    });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const inputRate = audioContext.sampleRate;
    source = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = function (e) {
      if (state !== "listening") return;
      const input = e.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      appendSamples(resampleTo16k(copy, inputRate));
    };
    source.connect(processor);
    processor.connect(audioContext.destination);
  }

  function connectWebSocket() {
    return new Promise((resolve, reject) => {
      let settled = false;

      const fail = (err) => {
        if (settled) return;
        settled = true;
        clearSessionReadyTimeout();
        reject(err);
      };

      sessionReadyTimeoutId = setTimeout(() => {
        fail(
          new Error(
            "サーバー応答タイムアウト。ASR モデルの読み込み完了後に再試行してください"
          )
        );
      }, SESSION_READY_TIMEOUT_MS);

      ws = new WebSocket(wsUrl());
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        const key = resolvedApiKey();
        currentSessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
        const startMsg = {
          type: "start",
          protocol_version: 2,
          mode: "meeting",
          language: window.__MEETING_LANGUAGE__ || "ja",
          wav_name: "web_meeting",
          audio_fs: TARGET_SR,
          wav_format: "pcm",
          max_speakers: 8,
          itn: true,
          session_id: currentSessionId,
        };
        if ((window.__MEETING_SPK_SOURCE__ || "pyannote") !== "pyannote") {
          startMsg.participant_id = ensureParticipantId();
        }
        if (key) startMsg.api_key = key;
        ws.send(JSON.stringify(startMsg));
      };

      ws.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        handleServerMessage(ev.data, () => {
          if (settled) return;
          settled = true;
          clearSessionReadyTimeout();
          resolve();
        });
      };

      ws.onerror = () => fail(new Error("WebSocket 接続失敗"));
      ws.onclose = () => {
        if (waitingForSummary) return;
        if (settled) {
          if (state === "listening") {
            showError("接続が切断されました");
          }
          return;
        }
        fail(new Error("接続が閉じられました（サーバーまたは認証を確認）"));
      };
    });
  }

  async function startSession() {
    if (!resolvedApiKey()) {
      showError("API Key がありません。サーバーを再起動するか管理者に連絡してください");
      return;
    }
    resetSummaryPanel();
    ensureMicrophoneApi();
    setState("connecting");
    setHint("接続中…マイク許可ダイアログが出たら「許可」を選んでください", false);
    try {
      await connectWebSocket();
      setState("listening");
      setHint(
        isSingleSpeakerMode()
          ? "話したあと約2秒止めると1行確定します（単一話者）。終了後に会議纪要が自動生成されます。"
          : "1台のマイクで複数話者を認識します。終了後に会議纪要が自動生成されます。",
        false
      );
      startPing();
      await startAudio();
    } catch (err) {
      showError(err.message || String(err));
    }
  }

  function updateKeyHint() {
    if (resolvedApiKey()) {
      const spkNote = isSingleSpeakerMode() ? "（単一話者）" : "";
      setHint(
        `「字幕開始」でマイクが有効になります。終了後は纪要を自動生成します。${spkNote}`,
        false
      );
    } else {
      setHint("API Key 未設定。サーバー設定を確認してください。", true);
    }
  }

  async function copySummary() {
    if (!currentSummaryMarkdown) return;
    try {
      await navigator.clipboard.writeText(currentSummaryMarkdown);
      if (summaryStatus) summaryStatus.textContent = "已复制到剪贴板";
    } catch (_) {
      if (summaryStatus) summaryStatus.textContent = "复制失败，请手动选择文本";
    }
  }

  function downloadSummary() {
    if (!currentSummaryMarkdown) return;
    const blob = new Blob([currentSummaryMarkdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${currentSessionId || "meeting"}.summary.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  btnStart.addEventListener("click", () => {
    startSession();
  });
  btnStop.addEventListener("click", stopSession);
  btnClear.addEventListener("click", () => {
    subtitleList.innerHTML = "";
    linesBySegId.clear();
  });
  if (btnCopySummary) btnCopySummary.addEventListener("click", copySummary);
  if (btnDownloadSummary) btnDownloadSummary.addEventListener("click", downloadSummary);

  ensureUrlHasKey();
  updateKeyHint();
})();
