import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import WebSocket from 'ws';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Drives the full pipeline end-to-end over the real WebSocket, N times,
 * and writes benchmark/report.json + report.md with p50/p90/max for the
 * "time to first audio byte" metric -- the deliverable latency report.
 *
 * Usage:
 *   node benchmark/latency_test.js --file ./sample.wav --runs 10
 *
 * The WAV must be 16-bit PCM mono at 16000 Hz (matches STT config).
 * If no --file is given, a short synthetic tone is used instead (useful
 * for smoke-testing the pipeline plumbing, not for real STT accuracy).
 */

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i !== -1 ? args[i + 1] : fallback;
};

const WS_URL = getArg('url', `ws://localhost:${process.env.PORT || 8080}/ws/voice`);
const RUNS = Number(getArg('runs', 5));
const FILE = getArg('file', null);
const USE_TEXT = getArg('text', 'false') === 'true'; // Use text input instead of audio
const CHUNK_MS = 100; // send mic audio in 100ms frames, like a real browser would

function loadPcmFromWav(filePath) {
  const buf = fs.readFileSync(filePath);
  // Minimal WAV parser: assumes canonical PCM header (44-byte offset)
  const dataStart = buf.indexOf(Buffer.from('data')) + 8;
  return buf.slice(dataStart);
}

function syntheticPcm(seconds = 2, sampleRate = 16000) {
  const samples = seconds * sampleRate;
  const buf = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i++) {
    const v = Math.round(Math.sin((2 * Math.PI * 220 * i) / sampleRate) * 8000);
    buf.writeInt16LE(v, i * 2);
  }
  return buf;
}

function runOnce(pcmBuffer) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL);
    let sendTimer = null;
    let firstAudioAt = null;
    const t0 = Date.now();
    let latencySummary = null;

    ws.on('open', () => {
      if (USE_TEXT) {
        // Use text input instead of audio
        const timestamp = Date.now();
        const testText = `नमस्ते, आप कैसे हैं? ${timestamp}`;
        ws.send(JSON.stringify({ type: 'text_input', text: testText }));
      } else {
        // Use audio streaming
        const bytesPerChunk = (16000 * (CHUNK_MS / 1000)) * 2; // 16-bit mono
        let offset = 0;
        sendTimer = setInterval(() => {
          if (offset >= pcmBuffer.length) {
            clearInterval(sendTimer);
            return;
          }
          const chunk = pcmBuffer.slice(offset, offset + bytesPerChunk);
          offset += bytesPerChunk;
          ws.send(JSON.stringify({ type: 'audio_chunk', audio: chunk.toString('base64') }));
        }, CHUNK_MS);
      }
    });

    ws.on('message', (raw) => {
      const msg = JSON.parse(raw.toString());
      if (msg.type === 'audio_chunk' && firstAudioAt == null) {
        firstAudioAt = Date.now() - t0;
      }
      if (msg.type === 'latency') latencySummary = msg.latency;
      if (msg.type === 'turn_done') {
        clearInterval(sendTimer);
        ws.close();
      }
    });

    ws.on('close', () => resolve({ firstAudioMs: firstAudioAt, latencySummary }));
    ws.on('error', (err) => reject(err));

    // Safety timeout per run
    setTimeout(() => { try { ws.close(); } catch { /* noop */ } }, 15000);
  });
}

function percentile(sorted, p) {
  const idx = Math.floor((p / 100) * (sorted.length - 1));
  return sorted[idx];
}

async function main() {
  const pcm = FILE ? loadPcmFromWav(path.resolve(FILE)) : syntheticPcm();
  console.log(`Running ${RUNS} turns against ${WS_URL}...`);

  const results = [];
  for (let i = 0; i < RUNS; i++) {
    console.log(`Run ${i + 1}/${RUNS}...`);
    try {
      const r = await runOnce(pcm);
      results.push(r);
      console.log(`  -> first audio in ${r.firstAudioMs}ms`);
    } catch (err) {
      console.error(`  -> failed: ${err.message}`);
    }
  }

  const times = results.map((r) => r.firstAudioMs).filter((t) => t != null).sort((a, b) => a - b);
  const report = {
    generatedAt: new Date().toISOString(),
    runs: RUNS,
    successful: times.length,
    first_audio_ms: {
      p50: percentile(times, 50),
      p90: percentile(times, 90),
      max: times[times.length - 1],
      min: times[0],
    },
    target_ms: 1500,
    max_allowed_ms: 2000,
    raw: results,
  };

  fs.writeFileSync(path.join(__dirname, 'report.json'), JSON.stringify(report, null, 2));
  const md = `# Latency Benchmark Report

Generated: ${report.generatedAt}
Runs: ${report.successful}/${report.runs} successful

| Metric | Value |
|---|---|
| p50 first-audio latency | ${report.first_audio_ms.p50} ms |
| p90 first-audio latency | ${report.first_audio_ms.p90} ms |
| min | ${report.first_audio_ms.min} ms |
| max | ${report.first_audio_ms.max} ms |
| Target | <= 1500 ms |
| Max allowed | <= 2000 ms |

${report.first_audio_ms.p90 <= 1500 ? '✅ p90 meets the 1.5s target.' : report.first_audio_ms.p90 <= 2000 ? '⚠️ p90 within max budget but above target.' : '❌ p90 exceeds the 2s max budget.'}
`;
  fs.writeFileSync(path.join(__dirname, 'report.md'), md);
  console.log('\nWrote benchmark/report.json and benchmark/report.md');
  console.log(md);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
