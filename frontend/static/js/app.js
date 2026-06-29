/**
 * ITC Grand Chola – Voice Agent  (v4 – True Full-Duplex Continuous Mode)
 *
 * One tap starts the session. Microphone stays open.
 * VAD detects speech → records → sends → streams response → speaks → listens again.
 * No holding. No releasing. One tap ends the session.
 */

(() => {
  // ── DOM refs ──────────────────────────────────────────────────────────────
  const chatArea        = document.getElementById('chatArea');
  const welcomeCard     = document.getElementById('welcomeCard');
  const textInput       = document.getElementById('textInput');
  const sendBtn         = document.getElementById('sendBtn');
  const micBtn          = document.getElementById('micBtn');
  const micHint         = document.getElementById('micHint');
  const endSessionHint  = document.getElementById('endSessionHint');
  const loadingOverlay  = document.getElementById('loadingOverlay');
  const loadingText     = document.getElementById('loadingText');
  const statusLabel     = document.getElementById('statusLabel');
  const statusDot       = document.querySelector('.status-dot');
  const liveIndicator   = document.getElementById('liveIndicator');

  // ── Conversation history ──────────────────────────────────────────────────
  const history = [];
  function pushHistory(role, content) {
    history.push({ role, content });
    if (history.length > 24) history.splice(0, 2);
  }

  // ── State ─────────────────────────────────────────────────────────────────
  let sessionActive = false;
  let isBusy        = false;    // HTTP non-session mode busy flag
  let turnInFlight  = false;    // a voice turn is being processed end-to-end
  let streamBubble  = null;
  let liveWS        = null;
  let pingTimer     = null;
  let toastTimer    = null;

  // TTS
  let ttsQueue    = [];
  let ttsSpeaking = false;
  let prefVoice   = null;       // cached preferred voice

  // VAD
  let vadActive    = false;
  let vadArmed     = false;     // ready to detect next utterance
  let vadRecording = false;     // currently capturing an utterance
  let vadMuted     = false;     // suppress VAD while TTS is playing
  let vadAudioCtx  = null;
  let vadAnalyser  = null;
  let vadStream    = null;
  let vadRecorder  = null;
  let vadChunks    = [];
  let vadSilTimer  = null;
  let vadRafId     = null;
  let vadSpeechMs  = 0;
  let keepAliveOsc = null;

  // VAD tuning
  const VAD_THRESHOLD    = 0.018;  // RMS speech threshold
  const VAD_BARGE_FACTOR = 1.4;   // barge-in multiplier
  const VAD_SILENCE_MS   = 1500;   // silence before end-of-utterance
  const VAD_MIN_MS       = 350;   // minimum utterance length
  const VAD_POST_GAP_MS  = 550;   // delay after TTS before re-arming mic

  // ── Voice preload ─────────────────────────────────────────────────────────
  function _pickVoice() {
    if (!window.speechSynthesis) return;
    const vv = window.speechSynthesis.getVoices();
    prefVoice =
      vv.find(v => v.lang.startsWith('en-GB') && /female|amy/i.test(v.name)) ||
      vv.find(v => v.lang.startsWith('en-GB')) ||
      vv.find(v => v.lang.startsWith('en')) || null;
  }
  if (window.speechSynthesis) {
    _pickVoice();
    window.speechSynthesis.onvoiceschanged = _pickVoice;
  }

  // ── Pill button visual states ─────────────────────────────────────────────
  // States: idle | listening | hearing | processing | speaking
  function setPillState(state) {
    micBtn.classList.remove('active', 'hearing', 'processing', 'speaking');
    if (endSessionHint) endSessionHint.classList.remove('visible');

    switch (state) {
      case 'idle':
        micHint.textContent = 'Tap to start';
        micBtn.setAttribute('aria-label', 'Start voice session');
        break;
      case 'listening':
        micBtn.classList.add('active');
        micHint.textContent = 'Listening…';
        micBtn.setAttribute('aria-label', 'End voice session');
        if (endSessionHint) endSessionHint.classList.add('visible');
        break;
      case 'hearing':
        micBtn.classList.add('active', 'hearing');
        micHint.textContent = 'Hearing you…';
        if (endSessionHint) endSessionHint.classList.add('visible');
        break;
      case 'processing':
        micBtn.classList.add('active', 'processing');
        micHint.textContent = 'Thinking…';
        if (endSessionHint) endSessionHint.classList.add('visible');
        break;
      case 'speaking':
        micBtn.classList.add('active', 'speaking');
        micHint.textContent = 'Speaking…';
        if (endSessionHint) endSessionHint.classList.add('visible');
        break;
    }
  }

  // ── Generic helpers ───────────────────────────────────────────────────────
  function setStatus(label, cls) {
    statusLabel.textContent = label;
    statusDot.className = 'status-dot' + (cls ? ` ${cls}` : '');
  }

  function showLoading(msg) {
    if (!sessionActive) { loadingText.textContent = msg || 'Thinking…'; loadingOverlay.hidden = false; }
    isBusy = true;
  }
  function hideLoading() { loadingOverlay.hidden = true; isBusy = false; }

  function showToast(msg) {
    let t = document.querySelector('.toast');
    if (!t) { t = document.createElement('div'); t.className = 'toast'; document.body.appendChild(t); }
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 4500);
  }

  function hideWelcome() {
    if (welcomeCard && welcomeCard.parentNode) welcomeCard.remove();
  }

  function esc(s) {
    const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
  }

  /**
   * Fix UTF-8 mojibake in source titles.
   * Occurs when the JSON file is UTF-8 but was read as Latin-1 by Python
   * (missing encoding='utf-8' on open()). Detects the pattern and re-decodes.
   * Example: "ITC Grand Chola â€" Hotel Introduction" → "ITC Grand Chola – Hotel Introduction"
   */
  function fixTitle(str) {
    if (!str) return str;
    // Quick check: mojibake always has sequences like 0xC2-0xEF followed by 0x80-0xBF
    if (!/[\u00c2-\u00ef][\u0080-\u00bf]/.test(str)) return str;
    try {
      const bytes = new Uint8Array(str.split('').map(c => c.charCodeAt(0) & 0xff));
      return new TextDecoder('utf-8').decode(bytes);
    } catch { return str; }
  }

  function scrollBottom() { chatArea.scrollTop = chatArea.scrollHeight; }

  // ── Chat rendering ────────────────────────────────────────────────────────
  function appendGuest(text) {
    hideWelcome();
    const g = document.createElement('div');
    g.className = 'message-group';
    g.innerHTML = `<div class="bubble-guest"><div class="sender">You</div>${esc(text)}</div>`;
    chatArea.appendChild(g);
    scrollBottom();
  }

  function appendChola(answer, sources) {
    const g = document.createElement('div');
    g.className = 'message-group';
    const src = sources && sources.length
      ? `<div class="sources">${sources.map(s =>
          `<span class="source-badge">${esc(fixTitle(s.title))}</span>`).join('')}</div>`
      : '';
    g.innerHTML = `<div class="bubble-chola"><div class="sender">Chola</div>
      <div class="answer-text">${esc(answer)}</div>${src}</div>`;
    chatArea.appendChild(g);
    scrollBottom();
  }

  function createStreamBubble() {
    hideWelcome();
    const g = document.createElement('div');
    g.className = 'message-group';
    g.innerHTML = `<div class="bubble-chola"><div class="sender">Chola</div>
      <div class="answer-text streaming-text"></div></div>`;
    chatArea.appendChild(g);
    scrollBottom();
    return g.querySelector('.answer-text');
  }

  function appendToken(token) {
    if (!streamBubble) streamBubble = createStreamBubble();
    streamBubble.textContent += token;
    scrollBottom();
  }

  function finaliseStream(sources) {
    if (!streamBubble) return;
    streamBubble.classList.remove('streaming-text');
    if (sources && sources.length) {
      const d = document.createElement('div');
      d.className = 'sources';
      d.innerHTML = sources.map(s =>
        `<span class="source-badge">${esc(fixTitle(s.title))}</span>`).join('');
      streamBubble.parentElement.appendChild(d);
    }
    streamBubble = null;
    scrollBottom();
  }

  // ── TTS ───────────────────────────────────────────────────────────────────
  function enqueueTTS(sentence) {
    ttsQueue.push(sentence);
    if (!ttsSpeaking) _drainTTS();
  }

  function _drainTTS() {
    if (!ttsQueue.length) {
      ttsSpeaking = false;
      // Re-arm mic after short gap (avoids echo pickup)
      if (sessionActive && vadActive) {
        setTimeout(() => {
          if (sessionActive && vadActive && !turnInFlight) {
            vadMuted = false;
            vadArmed = true;
            setPillState('listening');
            setStatus('Listening…', 'live');
          }
        }, VAD_POST_GAP_MS);
      }
      return;
    }
    ttsSpeaking = true;
    vadMuted    = true;
    setPillState('speaking');
    setStatus('Speaking…', 'speaking');

    const u = new SpeechSynthesisUtterance(ttsQueue.shift());
    u.lang = 'en-GB'; u.rate = 1.05; u.pitch = 1.0;
    if (prefVoice) u.voice = prefVoice;
    u.onend   = _drainTTS;
    u.onerror = _drainTTS;
    window.speechSynthesis.speak(u);
  }

  function stopTTS() {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    ttsQueue = []; ttsSpeaking = false; vadMuted = false;
  }

  // Non-session single-shot speak
  function speak(text) {
    if (!window.speechSynthesis || !text) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-GB'; u.rate = 1.0; u.pitch = 1.0;
    if (prefVoice) u.voice = prefVoice;
    u.onstart = () => setStatus('Speaking', 'speaking');
    u.onend   = () => setStatus('Ready', '');
    u.onerror = () => setStatus('Ready', '');
    window.speechSynthesis.speak(u);
  }

  // ── HTTP (non-session) ────────────────────────────────────────────────────
  async function sendQuery(query) {
    if (!query || isBusy) return;
    appendGuest(query);
    showLoading('Thinking…');
    setStatus('Processing', 'processing');
    try {
      const r = await fetch('/faq/api/chat/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      pushHistory('user', query);
      pushHistory('assistant', data.answer);
      appendChola(data.answer, data.sources);
      speak(data.tts_text || data.answer);
      setStatus('Ready', '');
    } catch (err) {
      showToast('Error: ' + err.message);
      setStatus('Error', 'error');
      setTimeout(() => setStatus('Ready', ''), 2500);
    } finally {
      hideLoading();
    }
  }

  // ── WebSocket ─────────────────────────────────────────────────────────────
  function openWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    liveWS = new WebSocket(`${proto}://${location.host}/faq/api/voice/ws/live`);

    liveWS.onopen = () => {
      setStatus('Listening…', 'live');
      pingTimer = setInterval(() => {
        if (liveWS && liveWS.readyState === WebSocket.OPEN)
          liveWS.send(JSON.stringify({ type: 'ping' }));
      }, 25_000);
    };

    liveWS.onmessage = e => {
      try { _handleMsg(JSON.parse(e.data)); } catch {}
    };

    liveWS.onerror = () => showToast('Connection error – reconnecting…');

    liveWS.onclose = () => {
      clearInterval(pingTimer); pingTimer = null;
      if (sessionActive) {
        setTimeout(() => {
          if (sessionActive) {
            openWS();
            // Re-arm VAD after reconnect
            if (vadActive && !turnInFlight) {
              vadMuted = false; vadArmed = true;
              setPillState('listening'); setStatus('Listening…', 'live');
            }
          }
        }, 2000);
      }
    };
  }

  function closeWS() {
    clearInterval(pingTimer); pingTimer = null;
    if (liveWS) { liveWS.onclose = null; liveWS.close(); liveWS = null; }
  }

  function wsSend(obj) {
    if (liveWS && liveWS.readyState === WebSocket.OPEN)
      liveWS.send(JSON.stringify(obj));
  }

  // ── WS message handler ────────────────────────────────────────────────────
  function _handleMsg(msg) {
    switch (msg.type) {

      case 'pong': break;

      case 'listening':
        // Server ready for next input
        turnInFlight = false;
        isBusy       = false;
        vadArmed     = true;
        if (!ttsSpeaking && !ttsQueue.length) {
          vadMuted = false;
          setPillState('listening');
          setStatus('Listening…', 'live');
        }
        break;

      case 'transcript':
        appendGuest(msg.text);
        pushHistory('user', msg.text);
        setPillState('processing');
        setStatus('Thinking…', 'processing');
        break;

      case 'stream_start':
        streamBubble = createStreamBubble();
        isBusy   = true;
        vadArmed = false;
        setPillState('processing');
        setStatus('Thinking…', 'processing');
        break;

      case 'token':
        if (msg.text) { appendToken(msg.text); enqueueTTS(msg.text); }
        break;

      case 'stream_end':
        finaliseStream(msg.sources || []);
        pushHistory('assistant', msg.full || '');
        isBusy = false;
        // If TTS already empty (very short answer), re-arm immediately
        if (!ttsSpeaking && !ttsQueue.length) {
          turnInFlight = false;
          vadMuted = false;
          vadArmed = true;
          setPillState('listening');
          setStatus('Listening…', 'live');
        }
        // Otherwise _drainTTS() re-arms when queue empties
        break;

      case 'barge_in_ack':
        stopTTS();
        turnInFlight = false; isBusy = false;
        vadMuted = false; vadArmed = true;
        setPillState('listening');
        setStatus('Listening…', 'live');
        break;

      case 'error':
        showToast('Chola: ' + (msg.message || 'Unknown error'));
        finaliseStream([]);
        isBusy = false; turnInFlight = false;
        vadMuted = false; vadArmed = true;
        setPillState('listening');
        setStatus('Listening…', 'live');
        break;
    }
  }

  // ── VAD ───────────────────────────────────────────────────────────────────
  async function startVAD() {
    if (vadActive) return true;
    try {
      vadStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
      });
    } catch {
      showToast('Microphone access denied. Please allow it in browser settings.');
      return false;
    }

    vadAudioCtx = new (window.AudioContext || window.webkitAudioContext)();

    // Silent oscillator keeps AudioContext alive (Chrome suspends idle contexts)
    const osc  = vadAudioCtx.createOscillator();
    const gain = vadAudioCtx.createGain();
    gain.gain.value = 0;
    osc.connect(gain);
    gain.connect(vadAudioCtx.destination);
    osc.start();
    keepAliveOsc = osc;

    vadAnalyser = vadAudioCtx.createAnalyser();
    vadAnalyser.fftSize = 512;
    vadAnalyser.smoothingTimeConstant = 0.3;
    vadAudioCtx.createMediaStreamSource(vadStream).connect(vadAnalyser);

    vadActive = true; vadArmed = true; vadRecording = false;
    vadMuted = false; turnInFlight = false;

    _vadLoop();
    return true;
  }

  function stopVAD() {
    vadActive = vadArmed = vadRecording = vadMuted = false;
    turnInFlight = false;
    if (vadRafId)    { cancelAnimationFrame(vadRafId); vadRafId = null; }
    if (vadSilTimer) { clearTimeout(vadSilTimer); vadSilTimer = null; }
    if (vadRecorder && vadRecorder.state !== 'inactive') vadRecorder.stop();
    vadRecorder = null; vadChunks = [];
    if (keepAliveOsc) { try { keepAliveOsc.stop(); } catch {} keepAliveOsc = null; }
    if (vadStream)   { vadStream.getTracks().forEach(t => t.stop()); vadStream = null; }
    if (vadAudioCtx) { vadAudioCtx.close().catch(() => {}); vadAudioCtx = null; }
  }

  function _vadLoop() {
    const buf = new Float32Array(vadAnalyser.frequencyBinCount);
    function tick() {
      if (!vadActive) return;
      vadRafId = requestAnimationFrame(tick);
      vadAnalyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);

      if (vadMuted) {
        // Barge-in: user speaks loudly over Chola
        if (rms > VAD_THRESHOLD * VAD_BARGE_FACTOR && !isBusy) {
          wsSend({ type: 'barge_in' });
          stopTTS();
        }
        return;
      }

      if (rms > VAD_THRESHOLD) {
        if (vadSilTimer) { clearTimeout(vadSilTimer); vadSilTimer = null; }
        if (!vadRecording && vadArmed && !turnInFlight) _vadStartRec();
      } else {
        if (vadRecording && !vadSilTimer) {
          vadSilTimer = setTimeout(() => { vadSilTimer = null; _vadStopRec(); }, VAD_SILENCE_MS);
        }
      }
    }
    tick();
  }

  function _getMime() {
    return ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4']
      .find(t => MediaRecorder.isTypeSupported(t)) || '';
  }

  function _vadStartRec() {
    if (vadRecording || !vadStream) return;
    vadRecording = true; vadArmed = false; turnInFlight = true;
    vadSpeechMs = Date.now(); vadChunks = [];

    const mime = _getMime();
    try { vadRecorder = new MediaRecorder(vadStream, mime ? { mimeType: mime } : {}); }
    catch { vadRecorder = new MediaRecorder(vadStream); }

    vadRecorder.ondataavailable = e => { if (e.data?.size > 0) vadChunks.push(e.data); };
    vadRecorder.onstop = () => {
      if (!vadActive) { turnInFlight = false; return; }
      vadRecording = false;
      const dur = Date.now() - vadSpeechMs;
      // Discard too-short clips (noise bursts)
      if (dur < VAD_MIN_MS || !vadChunks.length) {
        turnInFlight = false;
        if (sessionActive && !isBusy) vadArmed = true;
        return;
      }
      const mimeUsed = mime || 'audio/webm';
      const blob = new Blob(vadChunks, { type: mimeUsed });
      vadChunks = [];
      const reader = new FileReader();
      reader.onloadend = () => {
        if (!sessionActive) { turnInFlight = false; return; }
        wsSend({ type: 'vad_speech_end', data: reader.result.split(',')[1], mime: mimeUsed, history });
      };
      reader.readAsDataURL(blob);
    };

    vadRecorder.start(100);
    setPillState('hearing');
    setStatus('Hearing you…', 'listening');
    wsSend({ type: 'vad_speech_start' });
  }

  function _vadStopRec() {
    if (!vadRecording || !vadRecorder) return;
    if (vadRecorder.state !== 'inactive') vadRecorder.stop();
    setPillState('processing');
    setStatus('Processing…', 'processing');
  }

  // ── Session lifecycle ─────────────────────────────────────────────────────
  async function startSession() {
    sessionActive = true;
    if (liveIndicator) liveIndicator.hidden = false;
    openWS();
    const ok = await startVAD();
    if (!ok) {
      sessionActive = false;
      if (liveIndicator) liveIndicator.hidden = true;
      closeWS(); setPillState('idle'); return;
    }
    setPillState('listening');
    setStatus('Listening…', 'live');
  }

  function endSession() {
    sessionActive = false;
    stopVAD(); closeWS(); stopTTS();
    isBusy = false; turnInFlight = false;
    if (liveIndicator) liveIndicator.hidden = true;
    setPillState('idle');
    setStatus('Ready', '');
    showToast('Voice session ended');
  }

  // ── Mic button — instant response on mobile via touchstart ───────────────
  let _touchHandled = false;

  micBtn.addEventListener('touchstart', async (e) => {
    e.preventDefault();   // kill ghost click + 300ms delay
    _touchHandled = true;
    if (sessionActive) endSession(); else await startSession();
  }, { passive: false });

  micBtn.addEventListener('click', async () => {
    if (_touchHandled) { _touchHandled = false; return; }
    if (sessionActive) endSession(); else await startSession();
  });

  // ── Text input ────────────────────────────────────────────────────────────
  async function handleSend() {
    const q = textInput.value.trim();
    if (!q || isBusy) return;
    textInput.value = '';

    if (sessionActive && liveWS && liveWS.readyState === WebSocket.OPEN) {
      stopTTS();
      turnInFlight = true; vadArmed = false;
      appendGuest(q); pushHistory('user', q);
      wsSend({ type: 'text', query: q, history });
      setPillState('processing'); setStatus('Thinking…', 'processing');
      isBusy = true;
    } else {
      await sendQuery(q);
    }
  }

  sendBtn.addEventListener('click', handleSend);
  textInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  });
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.dataset.query;
      if (q && !isBusy) { textInput.value = q; handleSend(); }
    });
  });

  // ── Init ──────────────────────────────────────────────────────────────────
  setPillState('idle');
  (async () => {
    try {
      const r = await fetch('/faq/api/health');
      const d = await r.json();
      if (!d.index_loaded) {
        showToast('Index not ready – please wait and refresh.');
        setStatus('Loading…', 'processing');
      }
    } catch {}
  })();

})();