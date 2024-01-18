import { ok } from "node:assert";
import { CallbackManager, Event } from "../callbacks/CallbackManager";
import { BaseEmbedding } from "../embeddings";

const messageAccessor = (data: any): ChatResponseChunk => {
  return {
    delta: data.message.content,
  };
};
const completionAccessor = (data: any): CompletionResponse => {
  return { text: data.response };
};

// https://github.com/jmorganca/ollama
export class Ollama extends BaseEmbedding {
  readonly hasStreaming = true;

  // https://ollama.ai/library
  model: string;
  baseURL: string = "http://127.0.0.1:11434";
  temperature: number = 0.7;
  topP: number = 0.9;
  contextWindow: number = 4096;
  requestTimeout: number = 60 * 1000; // Default is 60 seconds
  additionalChatOptions?: Record<string, unknown>;
  callbackManager?: CallbackManager;

  constructor(
    init: Partial<Ollama> & {
      // model is required
      model: string;
    },
  ) {
    super();
    this.model = init.model;
    Object.assign(this, init);
  }

  get metadata(): LLMMetadata {
    return {
      model: this.model,
      temperature: this.temperature,
      topP: this.topP,
      maxTokens: undefined,
      contextWindow: this.contextWindow,
      tokenizer: undefined,
    };
  }

  private async getOllamaEmbeddings(prompt: string): Promise<number[]> {
    const payload = {
      model: this.model,
      prompt,
      options: {
        temperature: this.temperature,
        num_ctx: this.contextWindow,
        top_p: this.topP,
        ...this.additionalChatOptions,
      },
    };
    const response = await fetch(`${this.baseURL}/api/embeddings`, {
      body: JSON.stringify(payload),
      method: "POST",
      signal: AbortSignal.timeout(this.requestTimeout),
      headers: {
        "Content-Type": "application/json",
      },
    });
    const { embeddings } = await response.json();
    return embedding;
  }

  async getTextEmbedding(text: string): Promise<number[]> {
    return this.getEmbedding(text);
  }

  async getQueryEmbedding(query: string): Promise<number[]> {
    return this.getEmbedding(query);
  }
}
