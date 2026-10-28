/**
 * Pipeline stage definitions for the LLM Pipeline Visualizer.
 * Each stage represents a step in the LLM inference or training pipeline.
 */

export const PIPELINE_MODES = ["inference", "training"];

export const STAGE_IDS = [
  "prompt",
  "tokenization",
  "embeddings",
  "attention",
  "logits",
  "sampling",
  "generation",
];

/**
 * Full stage metadata for the inference pipeline.
 * Each stage has an id, title, icon label, short description, and detailed description.
 */
export const INFERENCE_STAGES = [
  {
    id: "prompt",
    title: "Input Prompt",
    icon: "📝",
    shortDescription: "Text input & prompt construction",
    description:
      "The raw text you type is combined with system instructions and conversation context before being sent to the model.",
  },
  {
    id: "tokenization",
    title: "Tokenization",
    icon: "🔤",
    shortDescription: "Text → token pieces & IDs",
    description:
      "A tokenizer splits the assembled prompt into token pieces and maps each to a numerical token ID from the model's vocabulary.",
  },
  {
    id: "embeddings",
    title: "Embeddings",
    icon: "📐",
    shortDescription: "Token IDs → vector representations",
    description:
      "Each token ID is looked up in an embedding matrix to produce a dense vector. Position information is also added so the model knows token order.",
  },
  {
    id: "attention",
    title: "Transformer",
    icon: "🧠",
    shortDescription: "Self-attention & feed-forward layers",
    description:
      "The embedded vectors pass through multiple Transformer blocks. Each block uses self-attention to mix information across positions, then a feed-forward network transforms each position independently.",
  },
  {
    id: "logits",
    title: "Logits",
    icon: "📊",
    shortDescription: "Hidden states → vocabulary scores",
    description:
      "The final hidden state is projected to a vector of size V (vocabulary size). These raw scores are called logits — they are not yet probabilities.",
  },
  {
    id: "sampling",
    title: "Sampling",
    icon: "🎲",
    shortDescription: "Temperature, top-k, top-p → token selection",
    description:
      "Logits are scaled by temperature, filtered by top-k/top-p, converted to probabilities via softmax, and then a token is selected through greedy or stochastic sampling.",
  },
  {
    id: "generation",
    title: "Generation",
    icon: "✨",
    shortDescription: "Autoregressive loop & output",
    description:
      "The selected token is appended to the context. The model repeats the pipeline with the updated sequence until a stop condition is met (EOS token, max length, or stop sequence).",
  },
];

/**
 * Training pipeline stages.
 */
export const TRAINING_STAGES = [
  {
    id: "training-data",
    title: "Training Data",
    icon: "📚",
    shortDescription: "Text examples for learning",
    description:
      "Training text is prepared as sequences of tokens. The model will learn to predict each next token given the preceding context.",
  },
  {
    id: "tokenization",
    title: "Tokenization",
    icon: "🔤",
    shortDescription: "Text → input & target sequences",
    description:
      "Training text is tokenized, and input/target pairs are created by shifting the sequence: the target at each position is the next token.",
  },
  {
    id: "forward-pass",
    title: "Forward Pass",
    icon: "➡️",
    shortDescription: "Compute predictions",
    description:
      "The model processes input tokens through all layers to produce logits — predicted scores for the next token at each position.",
  },
  {
    id: "loss",
    title: "Loss",
    icon: "📉",
    shortDescription: "Cross-entropy against targets",
    description:
      "Cross-entropy loss measures how far the model's predicted distribution is from the correct next token. Lower loss means better predictions.",
  },
  {
    id: "backprop",
    title: "Backpropagation",
    icon: "🔙",
    shortDescription: "Compute gradients",
    description:
      "The loss is propagated backward through the network to compute gradients — how much each parameter contributed to the error.",
  },
  {
    id: "optimizer",
    title: "Optimizer",
    icon: "⚙️",
    shortDescription: "Update model parameters",
    description:
      "An optimizer (like Adam) uses the gradients to adjust model parameters, reducing the loss for similar examples in the future.",
  },
];

/** Status values for pipeline stages */
export const STAGE_STATUSES = [
  "idle",
  "active",
  "complete",
  "unavailable",
  "error",
];

/** Data provenance categories */
export const PROVENANCE_KINDS = [
  "real",
  "derived",
  "illustrative",
  "simulated",
  "unavailable",
];

/** Run states for the pipeline execution */
export const RUN_STATES = [
  "idle",
  "loading",
  "running",
  "paused",
  "completed",
  "error",
  "cancelled",
];

/** Navigation items for the sidebar */
export const NAV_ITEMS = [
  { id: "pipeline", label: "Pipeline", icon: "⚡" },
  { id: "tokens", label: "Token Inspector", icon: "🔤" },
  { id: "embeddings", label: "Embeddings", icon: "📐" },
  { id: "attention", label: "Attention", icon: "🔍" },
  { id: "layers", label: "Transformer Layers", icon: "🧱" },
  { id: "sampling", label: "Logits & Sampling", icon: "🎲" },
  { id: "training", label: "Training Lab", icon: "🧪" },
  { id: "glossary", label: "Glossary", icon: "📖" },
];

/** Sample prompts for the demo */
export const SAMPLE_PROMPTS = [
  {
    id: "capital",
    label: "Capital of France",
    text: "What is the capital of France?",
  },
  {
    id: "explain",
    label: "Explain transformers",
    text: "Explain how transformer models work in simple terms.",
  },
  {
    id: "story",
    label: "Short story",
    text: "Once upon a time, in a land far away, there lived a",
  },
  {
    id: "code",
    label: "Python function",
    text: "Write a Python function that calculates the fibonacci sequence.",
  },
];
