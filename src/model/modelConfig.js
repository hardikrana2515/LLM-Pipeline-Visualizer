/**
 * LLM Pipeline Visualizer — Model Registry & Metadata Configuration
 *
 * Adheres strictly to DEVELOPMENT_PHASES.md Phase 5:
 * Documents model ID, revision, source, license, format, approximate download size,
 * browser requirements, and explicit verified capabilities vs unavailable internals.
 */

export const SUPPORTED_MODELS = [
  {
    id: "demo-mode",
    name: "Educational Simulation (Demo Mode)",
    category: "simulation",
    modelId: "builtin/educational-simulation",
    source: "Built-in Deterministic Pipeline",
    license: "MIT",
    format: "In-memory JavaScript",
    downloadSize: "0 MB",
    hardwareReq: "Any modern browser (No WebGPU/WASM required)",
    description:
      "Deterministic, instant walkthrough providing complete educational inspection across all 7 stages including illustrative attention heatmaps and intermediate tensors.",
    capabilities: {
      generation: true,
      tokenIds: true,
      logits: true,
      attention: true,
      hiddenStates: true,
      embeddings: true,
      provenance: "illustrative",
    },
    default: true,
  },
  {
    id: "qwen-0.5b-chat",
    name: "Qwen 1.5 (0.5B Chat)",
    category: "local-browser",
    modelId: "Xenova/Qwen1.5-0.5B-Chat",
    dtype: "q4",
    source: "Hugging Face / Alibaba Cloud",
    license: "Apache 2.0",
    format: "ONNX Runtime (int4 quantized)",
    downloadSize: "~310 MB",
    hardwareReq: "WebGPU strongly recommended (32-bit WASM may encounter std::bad_alloc)",
    description:
      "Smallest modern instruction-tuned LLM capable of coherent chat and multilingual completions entirely inside the browser.",
    capabilities: {
      generation: true,
      tokenIds: true,
      logits: false, // In-browser ONNX pipeline returns decoded text and token IDs; intermediate logits tensor is gated
      attention: false, // Internal attention weights not exposed by standard ONNX pipeline
      hiddenStates: false,
      embeddings: false,
      provenance: "real",
    },
    default: false,
  },
  {
    id: "gpt2",
    name: "GPT-2 Small (124M)",
    category: "local-browser",
    modelId: "Xenova/gpt2",
    dtype: "q4",
    source: "Hugging Face / OpenAI",
    license: "MIT",
    format: "ONNX Runtime (quantized)",
    downloadSize: "~135 MB",
    hardwareReq: "WASM or WebGPU (Compatible with all modern browsers)",
    description:
      "Classic foundation autoregressive language model. Fast download and reliable browser execution.",
    capabilities: {
      generation: true,
      tokenIds: true,
      logits: false,
      attention: false,
      hiddenStates: false,
      embeddings: false,
      provenance: "real",
    },
    default: false,
  },
  {
    id: "custom-ollama",
    name: "Local Ollama Endpoint",
    category: "remote-api",
    modelId: "http://localhost:11434/api/generate",
    source: "Local Ollama Server",
    license: "Dependent on loaded model",
    format: "HTTP REST Streaming JSON",
    downloadSize: "0 MB (uses locally installed model on user machine)",
    hardwareReq: "Running Ollama server",
    description:
      "Connects to a local Ollama server running Llama 3, Mistral, or Phi on the user's host machine via localhost.",
    capabilities: {
      generation: true,
      tokenIds: true,
      logits: false,
      attention: false,
      hiddenStates: false,
      embeddings: false,
      provenance: "real",
    },
    default: false,
  },
];

/**
 * Detects browser client hardware capabilities.
 * @returns {Promise<{ hasWebGPU: boolean, hasWasm: boolean, cpuCores: number, memoryGB?: number }>}
 */
export async function detectRuntimeCapabilities() {
  const hasWasm = typeof WebAssembly !== "undefined";
  let hasWebGPU = false;

  if (typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      hasWebGPU = Boolean(adapter);
    } catch {
      hasWebGPU = false;
    }
  }

  const cpuCores = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;
  const memoryGB = typeof navigator !== "undefined" && navigator.deviceMemory ? navigator.deviceMemory : undefined;

  return {
    hasWebGPU,
    hasWasm,
    cpuCores,
    memoryGB,
  };
}
