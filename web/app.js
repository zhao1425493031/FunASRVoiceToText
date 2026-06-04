/**
 * Mobile-first Web client: 16kHz PCM chunks over WebSocket.
 * Uses ScriptProcessorNode for broad browser compatibility.
 */

(function () {
  const TARGET_SR = 16000;
  const CHUNK_SAMPLES = 9600; // 600ms @ 16kHz, chunk_size [0,10,5]
  const CHUNK_BYTES = CHUNK_SAMPLES * 2;

  const statusText = document.getElementById("statusText");
  const partialText = document.getElementById("partialText");
  const finalList = document.getElementById("finalList");
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
  let awaitingFinal = false;
  let finalTimeoutId = null;
  let lastDraft = "";
  const FINAL_WAIT_MS = 60000;

  function apiKeyFromQuery() {
    const params = new URLSearchParams(location.search);
    return params.get("key") || "";
  }

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws/asr`;
  }

  function setState(next) {
    state = next;
    document.body.className = `state-${next}`;
    const labels = {
      idle: "待機",
      listening: "認識中…",
      finalizing: "確定稿を作成中…",
      error: "エラー",
    };
    statusText.textContent = labels[next] || next;
    btnStart.disabled = next === "listening" || next === "finalizing";
    btnStop.disabled = next !== "listening";
  }

  function resampleTo16k(float32, inputRate) {
    if (inputRate === TARGET_SR) {
      return float32;
    }
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
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    const buf = new Int16Array(samples);
    ws.send(buf.buffer);
  }

  function flushPendingPcm() {
    while (pcmBuffer.length >= CHUNK_SAMPLES && ws && ws.readyState === WebSocket.OPEN) {
      const slice = pcmBuffer.splice(0, CHUNK_SAMPLES);
      sendPcmChunk(slice);
    }
    if (pcmBuffer.length > 0 && ws && ws.readyState === WebSocket.OPEN) {
      const padded = new Int16Array(CHUNK_SAMPLES);
      for (let i = 0; i < pcmBuffer.length; i++) {
        padded[i] = pcmBuffer[i];
      }
      sendPcmChunk(padded);
      pcmBuffer = [];
    }
  }

  function appendSamples(float32) {
    const int16 = floatToInt16(float32);
    for (let i = 0; i < int16.length; i++) {
      pcmBuffer.push(int16[i]);
    }
    while (pcmBuffer.length >= CHUNK_SAMPLES && ws && ws.readyState === WebSocket.OPEN) {
      const slice = pcmBuffer.splice(0, CHUNK_SAMPLES);
      sendPcmChunk(slice);
    }
  }

  function clearFinalWait() {
    if (finalTimeoutId !== null) {
      clearTimeout(finalTimeoutId);
      finalTimeoutId = null;
    }
    awaitingFinal = false;
  }

  function finishStopSession() {
    clearFinalWait();
    if (ws) {
      ws.close();
      ws = null;
    }
    setState("idle");
  }

  function applyDraftFallbackIfNeeded() {
    const draft = partialText.textContent;
    if (draft && draft !== "—" && finalList.childElementCount === 0) {
      setSessionFinal(draft, draft);
    }
  }

  function freezeDraftDisplay(draft) {
    const text = (draft || lastDraft || "").trim();
    if (text) {
      partialText.textContent = text;
    }
    partialText.classList.add("partial-frozen");
  }

  function resetPartialDisplay() {
    lastDraft = "";
    partialText.textContent = "—";
    partialText.classList.remove("partial-frozen");
  }

  function scrollToConfirmed() {
    const panel = document.querySelector(".final-panel");
    if (panel) {
      panel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function addFinalLine(text) {
    if (!text || !text.trim()) return;
    const div = document.createElement("div");
    div.className = "final-item";
    div.textContent = text.trim();
    finalList.appendChild(div);
  }

  /** One block per recording session (stop/end), not per silence-finalize. */
  function setSessionFinal(text, draft) {
    if (!text || !text.trim()) return;
    finalList.innerHTML = "";
    addFinalLine(text);
    freezeDraftDisplay(draft);
    scrollToConfirmed();
  }

  function handleServerMessage(raw) {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    if (msg.type === "partial" && msg.text) {
      partialText.classList.remove("partial-frozen");
      lastDraft = msg.text;
      partialText.textContent = msg.text;
    } else if (msg.type === "final" && msg.text) {
      setSessionFinal(msg.text, msg.draft);
      if (awaitingFinal) {
        finishStopSession();
      }
    } else if (msg.type === "error") {
      const wasListening = state === "listening";
      setState("error");
      if (msg.code === "session_too_long") {
        statusText.textContent =
          msg.message ||
          "録音が上限時間を超えました。分割して録音してください";
      } else {
        statusText.textContent = msg.message || "サーバーエラー";
      }
      if (wasListening) {
        stopAudio();
        if (ws) {
          ws.close();
          ws = null;
        }
      }
    }
  }

  function ensureMicrophoneApi() {
    const secure =
      window.isSecureContext ||
      location.protocol === "https:" ||
      location.hostname === "localhost" ||
      location.hostname === "127.0.0.1";
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (!secure && !/^(localhost|127\.0\.0\.1)$/.test(location.hostname)) {
        throw new Error(
          "スマホ／ブラウザでは HTTPS が必要です。https://<PCのIP>:8765 でアクセスし、先に python scripts/generate_cert.py --ip <PCのIP> を実行してください"
        );
      }
      throw new Error(
        "このブラウザはマイク API に対応していません。Chrome／Safari または HTTPS をご利用ください"
      );
    }
  }

  async function startAudio() {
    ensureMicrophoneApi();
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
      video: false,
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const inputRate = audioContext.sampleRate;
    source = audioContext.createMediaStreamSource(mediaStream);
    const bufferSize = 4096;
    processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
    processor.onaudioprocess = function (e) {
      if (state !== "listening") return;
      const input = e.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      const resampled = resampleTo16k(copy, inputRate);
      appendSamples(resampled);
    };
    source.connect(processor);
    processor.connect(audioContext.destination);
  }

  function stopAudio() {
    if (processor) {
      processor.disconnect();
      processor = null;
    }
    if (source) {
      source.disconnect();
      source = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    pcmBuffer = [];
  }

  function connectWebSocket() {
    return new Promise((resolve, reject) => {
      ws = new WebSocket(wsUrl());
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        const startMsg = {
          type: "start",
          wav_name: "web_mic",
          audio_fs: TARGET_SR,
          wav_format: "pcm",
          chunk_size: [0, 10, 5],
          itn: false,
        };
        const key = apiKeyFromQuery();
        if (key) {
          startMsg.api_key = key;
        }
        ws.send(JSON.stringify(startMsg));
        resolve();
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          handleServerMessage(ev.data);
        }
      };
      ws.onerror = () => reject(new Error("WebSocket 接続に失敗しました"));
      ws.onclose = () => {
        if (awaitingFinal) {
          applyDraftFallbackIfNeeded();
          clearFinalWait();
          setState("idle");
          return;
        }
        if (state === "listening" || state === "finalizing") {
          setState("idle");
        }
      };
    });
  }

  async function startRecognition() {
    try {
      ensureMicrophoneApi();
      resetPartialDisplay();
      setState("listening");
      await connectWebSocket();
      await startAudio();
    } catch (err) {
      setState("error");
      statusText.textContent = err.message || String(err);
      stopAudio();
      if (ws) {
        ws.close();
        ws = null;
      }
    }
  }

  function stopRecognition() {
    stopAudio();
    if (ws && ws.readyState === WebSocket.OPEN) {
      flushPendingPcm();
      awaitingFinal = true;
      setState("finalizing");
      ws.send(JSON.stringify({ type: "end", is_speaking: false }));
      finalTimeoutId = setTimeout(() => {
        if (!awaitingFinal) {
          return;
        }
        applyDraftFallbackIfNeeded();
        finishStopSession();
      }, FINAL_WAIT_MS);
      return;
    }
    applyDraftFallbackIfNeeded();
    setState("idle");
  }

  btnStart.addEventListener("click", startRecognition);
  btnStop.addEventListener("click", stopRecognition);
  btnClear.addEventListener("click", () => {
    finalList.innerHTML = "";
    resetPartialDisplay();
  });
})();
