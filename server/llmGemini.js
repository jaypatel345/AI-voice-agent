import { GoogleGenAI } from '@google/genai';

/**
 * Streaming Gemini calls via Vertex AI or Gemini API based on configuration.
 * Uses the current @google/genai SDK with switchable authentication.
 */
class GeminiStreamingLLM {
  constructor({ model, apiKey, useVertexAI, project, location }) {
    this.model = model || 'gemini-3.1-flash-lite';
    this.useVertexAI = useVertexAI === 'true' || useVertexAI === true;
    
    if (this.useVertexAI) {
      // Use Vertex AI (requires billing)
      this.client = new GoogleGenAI({
        vertexai: true,
        project: project || process.env.GOOGLE_CLOUD_PROJECT,
        location: location || process.env.GOOGLE_CLOUD_LOCATION || 'us-central1',
      });
    } else {
      // Use Gemini API (no billing required)
      this.client = new GoogleGenAI({
        apiKey: apiKey || process.env.GOOGLE_API_KEY,
      });
    }
  }

  /**
   * Builds the minimal prompt required (system instructions kept tiny per
   * the latency directives) and streams the response.
   *
   * @param {string} systemPrompt - short, static instruction
   * @param {Array<{role:'user'|'model', text:string}>} shortTermTurns - last 2-3 turns only
   * @param {string} ragContext - <=1000 tokens of retrieved context
   * @param {string} userText - the (partial) transcript that triggered this turn
   * @param {(chunkText:string)=>void} onToken - called for every streamed text delta
   * @returns {Promise<string>} full response text
   */
  async streamReply({ systemPrompt, shortTermTurns = [], ragContext, userText, onToken }) {
    try {
      const contents = [
        ...shortTermTurns.map((t) => ({ role: t.role, parts: [{ text: t.text }] })),
        {
          role: 'user',
          parts: [{
            text: ragContext
              ? `Context:\n${ragContext}\n\nUser: ${userText}`
              : userText,
          }],
        },
      ];

      console.log('[LLM] Calling model:', this.model, 'with useVertexAI:', this.useVertexAI);
      const stream = await this.client.models.generateContentStream({
        model: this.model,
        contents,
        config: {
          systemInstruction: systemPrompt,
          maxOutputTokens: 200, // keep replies short -> shorter TTS queue -> lower perceived latency
          temperature: 0.4,
        },
      });

      let full = '';
      for await (const chunk of stream) {
        const delta = chunk.text || '';
        if (delta) {
          full += delta;
          onToken(delta);
        }
      }
      console.log('[LLM] Response complete, length:', full.length);
      return full;
    } catch (error) {
      console.error('[LLM] Error:', error.message);
      throw error;
    }
  }
}

export { GeminiStreamingLLM };
