import { createClient } from 'redis';
import crypto from 'crypto';

/**
 * Caches full (question -> {reply text, rag context}) pairs so a repeated
 * question skips RAG + LLM entirely and goes straight to TTS -- this is
 * where the biggest latency wins come from for FAQ-style voice bots.
 */
class RedisCache {
  constructor({ url, ttlSeconds }) {
    this.client = createClient({ url: url || 'redis://localhost:6379' });
    this.client.on('error', (err) => console.error('[redis] error', err.message));
    this.ttl = ttlSeconds || 3600;
    this.connected = false;
  }

  async connect() {
    if (!this.connected) {
      await this.client.connect();
      this.connected = true;
    }
  }

  _key(text) {
    const normalized = text.trim().toLowerCase().replace(/\s+/g, ' ');
    return `voice_qa:${crypto.createHash('sha1').update(normalized).digest('hex')}`;
  }

  async get(text) {
    if (!this.connected) return null;
    const raw = await this.client.get(this._key(text));
    return raw ? JSON.parse(raw) : null;
  }

  async set(text, value) {
    if (!this.connected) return;
    await this.client.set(this._key(text), JSON.stringify(value), { EX: this.ttl });
  }
}

export { RedisCache };
