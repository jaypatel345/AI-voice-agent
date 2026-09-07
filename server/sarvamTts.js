import WebSocket from 'ws';

/**
 * Thin wrapper around Sarvam's realtime TTS WebSocket.
 *
 * Hand-rolled (like sarvamStt.js) so we can request linear16 PCM output
 * and stream chunks to the browser as they arrive.
 *
 * Emits:
 *   'audio'  (base64Pcm, contentType) -> each synthesized audio chunk
 *   'done'   ()                        -> synthesis finished (final event)
 *   'error'  (err)
 *   'close'  (code, reason)
 */
import EventEmitter from 'events';

class SarvamTTSStream extends EventEmitter {
  constructor({ apiKey, model, speaker, targetLanguageCode, sampleRate }) {
    super();
    this.apiKey = apiKey;
    this.model = model || 'bulbul:v3';
    this.speaker = speaker || 'shubh';
    this.targetLanguageCode = targetLanguageCode || 'en-IN';
    this.sampleRate = sampleRate || 24000;
    this.ws = null;
    this.ready = false;
    this._configured = false;
    this._pending = [];
    this._doneEmitted = false;
  }

  connect() {
    const params = new URLSearchParams({ model: this.model });
    const url = `wss://api.sarvam.ai/text-to-speech/ws?${params.toString()}`;

    this.ws = new WebSocket(url, {
      headers: { 'API-SUBSCRIPTION-KEY': this.apiKey },
    });

    this.ws.on('open', () => {
      this.ready = true;
      this._sendConfig();
      this._flushPending();
    });

    this.ws.on('message', (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw.toString());
      } catch {
        console.error('[TTS] Failed to parse message:', raw.toString().substring(0, 200));
        return;
      }



      if (msg.type === 'audio' && msg.data?.audio) {
        const contentType = msg.data.content_type || 'audio/linear16';
        this.emit('audio', msg.data.audio, contentType);
        return;
      }

      if (msg.type === 'event' && msg.data?.event_type === 'final') {
        if (!this._doneEmitted) {
          this._doneEmitted = true;
          this.emit('done');
          // Close socket immediately after final to prevent timeout
          this.ws?.close(1000, 'synthesis complete');
        }
        return;
      }

      if (msg.type === 'error') {
        const message = msg.data?.message || msg.message || 'TTS API error';
        this.emit('error', new Error(message));
      }
    });

    this.ws.on('error', (err) => {
      console.error('[TTS] WebSocket error:', err.message);
      this.emit('error', err);
    });

    this.ws.on('close', (code, reason) => {
      console.log('[TTS] WebSocket closed:', code, reason?.toString());
      this.ready = false;
      this._configured = false;
      if (!this._doneEmitted) {
        this._doneEmitted = true;
        this.emit('done');
      }
      this.emit('close', code, reason?.toString());
    });

    return this;
  }

  _sendConfig() {
    const payload = JSON.stringify({
      type: 'config',
      data: {
        target_language_code: this.targetLanguageCode,
        speaker: this.speaker,
        speech_sample_rate: this.sampleRate,
        output_audio_codec: 'linear16',
        min_buffer_size: 30,
        max_chunk_length: 200,
      },
    });
    this.ws.send(payload);
    this._configured = true;
  }

  /** Feed text to synthesize. Queued until the socket is ready. */
  pushText(text) {
    if (!text) return;
    const payload = JSON.stringify({ type: 'text', data: { text } });
    if (this.ready && this._configured) this.ws.send(payload);
    else this._pending.push(payload);
  }

  /** Force processing of any buffered text. */
  flush() {
    const payload = JSON.stringify({ type: 'flush' });
    if (this.ready && this._configured) this.ws.send(payload);
    else this._pending.push(payload);
  }

  _flushPending() {
    if (!this.ready || !this._configured) return;
    this._pending.forEach((payload) => this.ws.send(payload));
    this._pending = [];
  }

  close() {
    try {
      if (!this.ws) return;
      
      // If still connecting, just let it timeout - don't force close
      if (this.ws.readyState === WebSocket.CONNECTING) {
        return;
      }
      
      // Remove all listeners to prevent error events
      this.ws.removeAllListeners();
      
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.close(1000, 'client done');
      }
    } catch {
      /* noop */
    }
  }
}

export { SarvamTTSStream };
