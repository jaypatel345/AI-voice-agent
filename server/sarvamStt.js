import WebSocket from 'ws';

/**
 * Thin wrapper around Sarvam's realtime STT WebSocket.
 *
 * We hand-roll the WS client (rather than the JS SDK) for two reasons:
 *   1. We need raw PCM16 frames straight off the browser mic (no WAV
 *      re-encoding round trip), and the JS SDK's per-message helper only
 *      accepts "audio/wav".
 *   2. We need `vad_signals` events (START_SPEECH/END_SPEECH) for
 *      barge-in, and partial transcripts as they land, not just the
 *      final result.
 *
 * Emits:
 *   'partial'  (text, isFinal)  -> fired on every transcript update. The
 *              orchestrator triggers RAG+LLM on the FIRST partial that
 *              looks like a complete-enough thought, not on final.
 *   'vad'      (signal_type)    -> START_SPEECH / END_SPEECH, used for barge-in
 *   'error'    (err)
 *   'close'    ()
 */
import EventEmitter from 'events';

class SarvamSTTStream extends EventEmitter {
  constructor({ apiKey, model, languageCode, sampleRate = 16000 }) {
    super();
    this.apiKey = apiKey;
    this.model = model || 'saaras:v4';
    this.languageCode = languageCode || 'en-IN';
    this.sampleRate = sampleRate;
    this.ws = null;
    this.ready = false;
    this._pending = [];
    this._explicitlyClosed = false;
    this._retryMs = 1000;       // exponential backoff, reset on a clean open
    this._retryTimer = null;
    this._giveUp = false;       // set on a permanent error (credits / auth)
  }

  connect() {
    const params = new URLSearchParams({
      model: this.model,
      language_code: this.languageCode,
      sample_rate: String(this.sampleRate),
      input_audio_codec: 'pcm_s16le',
      high_vad_sensitivity: 'true',
      vad_signals: 'true',
      flush_signal: 'true',
    });
    const url = `wss://api.sarvam.ai/speech-to-text/ws?${params.toString()}`;

    this.ws = new WebSocket(url, {
      headers: { 'API-SUBSCRIPTION-KEY': this.apiKey },
    });

    this.ws.on('open', () => {
      this.ready = true;
      this._retryMs = 1000; // clean connection -> reset backoff
      // flush anything queued while the socket was connecting
      this._pending.forEach((chunk) => this.ws.send(chunk));
      this._pending = [];
    });

    this.ws.on('message', (raw) => {
      let msg;
      try { msg = JSON.parse(raw.toString()); } catch (e) { 
        console.error('[STT] Failed to parse message:', raw.toString().substring(0, 100));
        return; 
      }


      if (msg.type === 'events' && msg.data?.signal_type) {
        this.emit('vad', msg.data.signal_type);
        return;
      }
      if (msg.type === 'data' && msg.data?.transcript != null) {
        const isFinal = !!msg.data.is_final || msg.data.status === 'final';
        this.emit('partial', msg.data.transcript, isFinal);
      }
      if (msg.type === 'error') {
        console.error('[STT] API error:', msg);
        const errorMsg = msg.message || msg.data?.message || JSON.stringify(msg);
        this.emit('error', new Error(errorMsg));
      }
    });

    this.ws.on('error', (err) => {
      this.emit('error', err);
    });
    this.ws.on('close', (code, reason) => {
      const text = reason?.toString() || '';
      this.ready = false;
      this.emit('close', code, text);

      // A 1003 (policy) close for exhausted credits or a bad key is permanent --
      // reconnecting just tight-loops against the API. Stop and surface it once.
      const permanent = code === 1003 && /credit|exhaust|not authori[sz]ed|invalid.*key/i.test(text);
      if (permanent && !this._giveUp) {
        this._giveUp = true;
        console.error('[STT] permanent failure, not reconnecting:', text.slice(0, 120));
        this.emit('error', new Error(text || 'STT permanently unavailable'));
        return;
      }
      if (this._explicitlyClosed || this._giveUp) return;

      // Otherwise reconnect with capped exponential backoff (1s -> 30s).
      console.log('[STT] closed', code, `- retrying in ${this._retryMs}ms`);
      clearTimeout(this._retryTimer);
      this._retryTimer = setTimeout(() => this.connect(), this._retryMs);
      this._retryMs = Math.min(this._retryMs * 2, 30_000);
    });

    return this;
  }

  /** Send a raw PCM16LE mic chunk (base64-encoded) straight through. */
 sendAudioChunk(base64Pcm) {
    const payload = JSON.stringify({
      audio: { data: base64Pcm, sample_rate: this.sampleRate, encoding: 'audio/wav' },
    });

    if (this.ready) {
      this.ws.send(payload);
    } else {
      this._pending.push(payload);
    }
  }


  /** Force immediate processing instead of waiting on VAD silence. */
  flush() {
    if (this.ready) this.ws.send(JSON.stringify({ type: 'flush' }));
  }

  close() {
    this._explicitlyClosed = true;
    clearTimeout(this._retryTimer);
    try { this.ws?.close(1000, 'client done'); } catch { /* noop */ }
  }
}

export { SarvamSTTStream };
