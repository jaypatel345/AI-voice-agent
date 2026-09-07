/**
 * Tracks per-turn latency across every stage of the pipeline so we can
 * prove the sub-2s (target <1.5s) perceived-latency requirement and
 * generate the benchmark report deliverable.
 *
 * A "turn" starts the moment the first partial transcript triggers the
 * pipeline and ends the moment the first audio byte is flushed to the
 * browser (that's the number the user actually feels).
 */
class TurnTimer {
  constructor(turnId) {
    this.turnId = turnId;
    this.marks = {};
    this.t0 = process.hrtime.bigint();
    this.mark('turn_start');
  }

  mark(label) {
    const now = process.hrtime.bigint();
    this.marks[label] = Number(now - this.t0) / 1e6; // ms since turn start
    return this.marks[label];
  }

  elapsedSince(label) {
    if (!(label in this.marks)) return null;
    return this.mark('_now_tmp') - this.marks[label];
  }

  summary() {
    const m = this.marks;
    return {
      turnId: this.turnId,
      partial_transcript_ms: m.partial_transcript ?? null,
      rag_retrieved_ms: m.rag_retrieved ?? null,
      rag_latency_ms: m.rag_retrieved != null && m.partial_transcript != null
        ? +(m.rag_retrieved - m.partial_transcript).toFixed(1) : null,
      llm_first_token_ms: m.llm_first_token ?? null,
      llm_first_token_latency_ms: m.llm_first_token != null && m.partial_transcript != null
        ? +(m.llm_first_token - m.partial_transcript).toFixed(1) : null,
      tts_first_chunk_ms: m.tts_first_chunk ?? null,
      first_audio_to_client_ms: m.first_audio_to_client ?? null, // <-- the headline number
      cache_hit: !!m.cache_hit,
      turn_end_ms: m.turn_end ?? null,
    };
  }

  log() {
    const s = this.summary();
    const headline = s.first_audio_to_client_ms;
    const flag = headline == null ? '?' : headline <= 1500 ? '✅' : headline <= 2500 ? '⚠️' : '❌';
    // eslint-disable-next-line no-console
    console.log(
      `[latency] turn=${s.turnId} first_audio=${headline?.toFixed?.(1)}ms ${flag} ` +
      `rag=${s.rag_latency_ms}ms llm_ttft=${s.llm_first_token_latency_ms}ms cache=${s.cache_hit}`
    );
    return s;
  }
}

export { TurnTimer };
