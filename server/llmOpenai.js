import OpenAI from 'openai';

/**
 * Streaming OpenAI calls for low-latency voice assistant.
 * Uses OpenAI API with streaming support.
 */
class OpenAIStreamingLLM {
  constructor({ model, apiKey }) {
    this.model = model || 'gpt-4o-mini';
    this.client = new OpenAI({
      apiKey: apiKey || process.env.OPENAI_API_KEY,
    });
  }

  /**
   * Builds the minimal prompt required (system instructions kept tiny per
   * the latency directives) and streams the response.
   *
   * @param {string} systemPrompt - short, static instruction
   * @param {Array<{role:'user'|'assistant', content:string}>} shortTermTurns - last 2-3 turns only
   * @param {string} ragContext - <=1000 tokens of retrieved context
   * @param {string} userText - the (partial) transcript that triggered this turn
   * @param {(chunkText:string)=>void} onToken - called for every streamed text delta
   * @returns {Promise<string>} full response text
   */
  async streamReply({ systemPrompt, shortTermTurns = [], ragContext, userText, onToken }) {
    try {
      const messages = [
        ...shortTermTurns.map((t) => ({ role: t.role === 'model' ? 'assistant' : t.role, content: t.text })),
        {
          role: 'user',
          content: ragContext
            ? `Context:\n${ragContext}\n\nUser: ${userText}`
            : userText,
        },
      ];

      console.log('[OpenAI] Calling model:', this.model);
      const stream = await this.client.chat.completions.create({
        model: this.model,
        messages: [
          { role: 'system', content: systemPrompt },
          ...messages,
        ],
        max_tokens: 200, // keep replies short -> shorter TTS queue -> lower perceived latency
        temperature: 0.4,
        stream: true,
      });

      let full = '';
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content || '';
        if (delta) {
          full += delta;
          onToken(delta);
        }
      }
      console.log('[OpenAI] Response complete, length:', full.length);
      return full;
    } catch (error) {
      console.error('[OpenAI] Error:', error.message);
      throw error;
    }
  }
}

export { OpenAIStreamingLLM };
