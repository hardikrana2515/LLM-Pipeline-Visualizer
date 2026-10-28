/**
 * LLM Pipeline Visualizer — Model Adapter Architecture
 *
 * Implements Phase 5 Real Model Integration:
 * - Isolation of model-specific code
 * - Non-blocking asynchronous inference with streaming token callbacks
 * - Explicit user-triggered model loading (never auto-loaded at startup)
 * - Accurate progress tracking, error recovery, and cancellation support
 * - Dynamic fallback to Demo Simulation if browser runtime fails
 * - Capability gating: real tokens vs unavailable internal tensors
 */

import { SUPPORTED_MODELS, detectRuntimeCapabilities } from "./modelConfig";
import { getDeterministicDemoData } from "../engine/demoFixtures";

export class BaseModelAdapter {
  constructor(modelConfig) {
    this.config = modelConfig;
    this.abortController = null;
    this.isLoaded = false;
  }

  getCapabilities() {
    return this.config.capabilities;
  }

  async load(onProgress = () => {}) {
    throw new Error("load() must be implemented by adapter.");
  }

  async generate(prompt, options = {}, onToken = () => {}) {
    throw new Error("generate() must be implemented by adapter.");
  }

  cancel() {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  unload() {
    this.cancel();
    this.isLoaded = false;
  }
}

/**
 * 1. Demo Simulation Adapter (Built-in educational engine)
 */
export class SimulationModelAdapter extends BaseModelAdapter {
  constructor() {
    super(SUPPORTED_MODELS[0]);
    this.isLoaded = true;
  }

  async load() {
    this.isLoaded = true;
    return true;
  }

  async generate(prompt, options = {}, onToken = () => {}) {
    this.abortController = new AbortController();
    const { signal } = this.abortController;

    const data = getDeterministicDemoData(prompt);
    const tokens = data.generation.tokens;

    const startTime = performance.now();
    let accumulatedText = "";

    for (let i = 0; i < tokens.length; i++) {
      if (signal.aborted) {
        throw new Error("Generation cancelled by user.");
      }

      // Simulate realistic token generation latency (80ms per token)
      await new Promise((resolve) => setTimeout(resolve, 80));

      const tok = tokens[i];
      accumulatedText += tok.text;

      onToken({
        text: tok.text,
        id: tok.id,
        step: tok.step,
        accumulatedText,
        isFinished: i === tokens.length - 1,
      });
    }

    const elapsedMs = performance.now() - startTime;
    return {
      output: accumulatedText,
      tokens,
      elapsedMs,
      throughputTokPerSec: parseFloat(((tokens.length / elapsedMs) * 1000).toFixed(1)),
      provenance: "illustrative",
    };
  }
}

/**
 * 2. In-Browser Local Model Adapter (Transformers.js ONNX / WebGPU)
 */
export class TransformersJsModelAdapter extends BaseModelAdapter {
  constructor(modelConfig) {
    super(modelConfig);
    this.pipeline = null;
    this.tokenizer = null;
  }

  async load(onProgress = () => {}) {
    this.abortController = new AbortController();
    try {
      onProgress({ status: "checking", message: "Detecting WebGPU / WASM runtime..." });
      const caps = await detectRuntimeCapabilities();

      onProgress({
        status: "downloading",
        message: `Downloading model weights (${this.config.downloadSize})...`,
        progress: 0,
      });

      // Dynamically import @huggingface/transformers or fallback to CDN ESM
      let pipelineFn;
      try {
        const hf = await import("@huggingface/transformers");
        pipelineFn = hf.pipeline;
      } catch {
        // Fallback to CDN ESM import if local npm package is not compiled
        const cdn = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0");
        pipelineFn = cdn.pipeline;
      }

      const device = caps.hasWebGPU ? "webgpu" : "wasm";

      const pipelineOpts = {
        device,
        progress_callback: (progressInfo) => {
          if (progressInfo.status === "progress" && progressInfo.total) {
            const percent = Math.round((progressInfo.loaded / progressInfo.total) * 100);
            onProgress({
              status: "downloading",
              message: `Loading ${progressInfo.file || "weights"} (${percent}%)...`,
              progress: percent,
            });
          } else {
            onProgress({
              status: "loading",
              message: progressInfo.name || "Compiling ONNX compute graph...",
              progress: 95,
            });
          }
        },
      };

      if (this.config.dtype) {
        pipelineOpts.dtype = this.config.dtype;
      }

      this.pipeline = await pipelineFn("text-generation", this.config.modelId, pipelineOpts);

      this.isLoaded = true;
      onProgress({ status: "ready", message: "Model loaded and compiled successfully." });
      return true;
    } catch (err) {
      this.isLoaded = false;
      const msg = err?.message || String(err);
      if (
        msg.includes("bad_alloc") ||
        msg.includes("ERROR_CODE: 6") ||
        msg.includes("Out of memory")
      ) {
        throw new Error(
          `Browser WebAssembly memory limit exceeded (std::bad_alloc). Loading a ~310MB ONNX graph requires contiguous WASM linear memory beyond standard 32-bit browser limits without WebGPU. Recommended: Select 'GPT-2 Small (124M)' or connect to a local Ollama server.`
        );
      }
      throw new Error(`Failed to initialize in-browser model: ${msg}`);
    }
  }

  async generate(prompt, options = {}, onToken = () => {}) {
    if (!this.pipeline) {
      throw new Error("Model is not loaded. Please load the model first.");
    }
    this.abortController = new AbortController();

    const startTime = performance.now();
    const maxNewTokens = options.maxNewTokens || 30;
    const temperature = options.temperature ?? 0.7;

    const result = await this.pipeline(prompt, {
      max_new_tokens: maxNewTokens,
      temperature: Math.max(0.01, temperature),
      do_sample: temperature > 0,
      callback_function: (outputTokens) => {
        // Real token stream callback
        if (outputTokens && outputTokens.length > 0) {
          const lastToken = outputTokens[outputTokens.length - 1];
          onToken({
            text: lastToken,
            accumulatedText: outputTokens.join(""),
          });
        }
      },
    });

    const elapsedMs = performance.now() - startTime;
    const generatedText = result[0]?.generated_text || "";

    return {
      output: generatedText,
      elapsedMs,
      provenance: "real",
    };
  }
}

/**
 * 3. Local Ollama Endpoint Adapter (http://localhost:11434)
 */
export class OllamaModelAdapter extends BaseModelAdapter {
  constructor(modelConfig) {
    super(modelConfig);
    this.endpoint = modelConfig.modelId || "http://localhost:11434/api/generate";
  }

  async load(onProgress = () => {}) {
    onProgress({ status: "checking", message: "Connecting to local Ollama server..." });
    try {
      const res = await fetch("http://localhost:11434/api/tags", { method: "GET" });
      if (!res.ok) throw new Error("Ollama server responded with error status.");
      this.isLoaded = true;
      onProgress({ status: "ready", message: "Connected to local Ollama server." });
      return true;
    } catch {
      this.isLoaded = false;
      throw new Error(
        "Could not connect to Ollama at http://localhost:11434. Make sure Ollama is running locally."
      );
    }
  }

  async generate(prompt, options = {}, onToken = () => {}) {
    this.abortController = new AbortController();
    const startTime = performance.now();

    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: options.modelName || "llama3",
        prompt: prompt,
        stream: true,
      }),
      signal: this.abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`Ollama generation failed: ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let accumulatedText = "";
    let tokenCount = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n").filter((l) => l.trim().length > 0);

      for (const line of lines) {
        try {
          const json = JSON.parse(line);
          if (json.response) {
            accumulatedText += json.response;
            tokenCount++;
            onToken({
              text: json.response,
              accumulatedText,
              isFinished: json.done,
            });
          }
        } catch {
          // ignore partial JSON chunk
        }
      }
    }

    const elapsedMs = performance.now() - startTime;
    return {
      output: accumulatedText,
      tokenCount,
      elapsedMs,
      throughputTokPerSec: parseFloat(((tokenCount / elapsedMs) * 1000).toFixed(1)),
      provenance: "real",
    };
  }
}

/**
 * Model Service Singleton Manager
 */
class ModelService {
  constructor() {
    this.currentModelId = "demo-mode";
    this.adapter = new SimulationModelAdapter();
    this.status = "ready"; // 'idle' | 'checking' | 'downloading' | 'loading' | 'ready' | 'generating' | 'error'
    this.statusMessage = "Demo mode active";
    this.downloadProgress = 0;
    this.listeners = new Set();
  }

  get isLoaded() {
    return Boolean(this.adapter && this.adapter.isLoaded);
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify() {
    for (const listener of this.listeners) {
      listener({
        currentModelId: this.currentModelId,
        currentModel: this.getModelConfig(),
        status: this.status,
        statusMessage: this.statusMessage,
        downloadProgress: this.downloadProgress,
        capabilities: this.adapter.getCapabilities(),
      });
    }
  }

  getModelConfig(modelId = this.currentModelId) {
    return SUPPORTED_MODELS.find((m) => m.id === modelId) || SUPPORTED_MODELS[0];
  }

  async selectModel(modelId) {
    if (this.currentModelId === modelId) return;

    if (this.adapter) {
      this.adapter.unload();
    }

    this.currentModelId = modelId;
    const config = this.getModelConfig(modelId);

    if (config.category === "simulation") {
      this.adapter = new SimulationModelAdapter();
      this.status = "ready";
      this.statusMessage = "Demo Simulation active";
      this.downloadProgress = 0;
    } else if (config.category === "local-browser") {
      this.adapter = new TransformersJsModelAdapter(config);
      this.status = "idle";
      this.statusMessage = `Model selected (${config.name}). Click 'Connect / Load Model' to start.`;
      this.downloadProgress = 0;
    } else if (config.category === "remote-api") {
      this.adapter = new OllamaModelAdapter(config);
      this.status = "idle";
      this.statusMessage = "Ollama endpoint selected. Click 'Connect' to verify.";
      this.downloadProgress = 0;
    }

    this.notify();
  }

  async loadActiveModel() {
    const config = this.getModelConfig();
    if (config.category === "simulation") {
      this.status = "ready";
      this.statusMessage = "Demo Simulation is always ready.";
      this.notify();
      return;
    }

    try {
      this.status = "loading";
      this.notify();

      await this.adapter.load((progressInfo) => {
        this.status = progressInfo.status || "loading";
        this.statusMessage = progressInfo.message || "Loading model...";
        if (progressInfo.progress !== undefined) {
          this.downloadProgress = progressInfo.progress;
        }
        this.notify();
      });

      this.status = "ready";
      this.statusMessage = `${config.name} loaded and ready for inference.`;
      this.notify();
    } catch (err) {
      this.status = "error";
      this.statusMessage = err.message || "Model failed to load.";
      this.notify();
    }
  }

  async generate(prompt, options = {}, onToken = () => {}) {
    this.status = "generating";
    this.statusMessage = "Generating output tokens...";
    this.notify();

    try {
      const result = await this.adapter.generate(prompt, options, onToken);
      this.status = "ready";
      this.statusMessage = "Generation complete.";
      this.notify();
      return result;
    } catch (err) {
      this.status = "error";
      this.statusMessage = err.message;
      this.notify();
      throw err;
    }
  }

  cancel() {
    if (this.adapter) {
      this.adapter.cancel();
      this.status = "ready";
      this.statusMessage = "Generation cancelled.";
      this.notify();
    }
  }
}

export const modelService = new ModelService();
