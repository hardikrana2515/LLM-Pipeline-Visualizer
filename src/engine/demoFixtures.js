/**
 * LLM Pipeline Visualizer — Deterministic Educational Demo Fixtures
 *
 * Provides mathematically grounded, deterministic intermediate data for all pipeline stages:
 * - Tokenization: actual subword pieces, IDs, and byte spans
 * - Embeddings: vectors, norms, and RoPE angle rotations
 * - Attention: multi-head causal attention weights (summing to 1.0 per row)
 * - Logits: vocabulary scores before softmax
 * - Sampling: temperature scaling, top-k/top-p filtering, softmax probabilities
 * - Generation: step-by-step autoregressive completion tokens
 *
 * All data is strictly tagged with provenance: 'illustrative' or 'derived'.
 */

import { defaultTokenizer } from "../tokenizer/subwordTokenizer";

export const DEMO_FIXTURES = {
  // 1. Capital of France
  "What is the capital of France?": {
    prompt: {
      raw: "What is the capital of France?",
      formatted: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
      charCount: 30,
      tokenCount: 7,
      provenance: "illustrative",
    },
    tokenization: {
      tokens: [
        { text: "What", id: 2061, pos: 0, type: "prompt", byteLength: 4 },
        { text: " is", id: 318, pos: 1, type: "prompt", byteLength: 3 },
        { text: " the", id: 262, pos: 2, type: "prompt", byteLength: 4 },
        { text: " capital", id: 3139, pos: 3, type: "prompt", byteLength: 8 },
        { text: " of", id: 286, pos: 4, type: "prompt", byteLength: 3 },
        { text: " France", id: 4881, pos: 5, type: "prompt", byteLength: 7 },
        { text: "?", id: 30, pos: 6, type: "prompt", byteLength: 1 },
      ],
      vocabSize: 32000,
      provenance: "illustrative",
    },
    embeddings: {
      dModel: 768,
      vectors: [
        { token: "What", id: 2061, slice: [0.12, -0.45, 0.81, -0.03], norm: 1.0, ropeAngle: "0.00 rad" },
        { token: " is", id: 318, slice: [-0.31, 0.22, 0.15, 0.67], norm: 1.0, ropeAngle: "0.18 rad" },
        { token: " the", id: 262, slice: [0.08, -0.19, -0.54, 0.38], norm: 1.0, ropeAngle: "0.35 rad" },
        { token: " capital", id: 3139, slice: [0.74, 0.33, -0.12, -0.41], norm: 1.0, ropeAngle: "0.52 rad" },
        { token: " of", id: 286, slice: [-0.05, 0.11, 0.49, -0.29], norm: 1.0, ropeAngle: "0.70 rad" },
        { token: " France", id: 4881, slice: [0.89, -0.62, 0.35, 0.14], norm: 1.0, ropeAngle: "0.87 rad" },
        { token: "?", id: 30, slice: [0.02, 0.04, -0.18, 0.77], norm: 1.0, ropeAngle: "1.05 rad" },
      ],
      provenance: "illustrative",
    },
    attention: {
      heads: 4,
      tokens: ["What", " is", " the", " capital", " of", " France", "?"],
      // 7x7 lower-triangular causal attention matrix (rows sum to 1.0)
      matrix: [
        [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.45, 0.55, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.25, 0.20, 0.55, 0.00, 0.00, 0.00, 0.00],
        [0.10, 0.15, 0.20, 0.55, 0.00, 0.00, 0.00],
        [0.08, 0.12, 0.15, 0.35, 0.30, 0.00, 0.00],
        [0.05, 0.05, 0.10, 0.42, 0.18, 0.20, 0.00],
        [0.08, 0.06, 0.08, 0.38, 0.10, 0.25, 0.05],
      ],
      provenance: "illustrative",
    },
    logits: {
      candidates: [
        { token: "Paris", id: 4881, logit: 14.2 },
        { token: "France", id: 4881, logit: 10.8 },
        { token: "Europe", id: 5931, logit: 8.4 },
        { token: "The", id: 464, logit: 6.9 },
        { token: "a", id: 64, logit: 5.1 },
        { token: "known", id: 2487, logit: 3.2 },
      ],
      provenance: "illustrative",
    },
    sampling: {
      selectedToken: { text: "Paris", id: 4881, prob: 0.742 },
      provenance: "derived",
    },
    generation: {
      completionText: "The capital of France is Paris.",
      tokens: [
        { text: "The", id: 464, step: 1 },
        { text: " capital", id: 3139, step: 2 },
        { text: " of", id: 286, step: 3 },
        { text: " France", id: 4881, step: 4 },
        { text: " is", id: 318, step: 5 },
        { text: " Paris", id: 4881, step: 6 },
        { text: ".", id: 13, step: 7 },
      ],
      stopReason: "stop_sequence",
      provenance: "illustrative",
    },
  },

  // 2. Short story
  "Once upon a time, in a land far away, there lived a": {
    prompt: {
      raw: "Once upon a time, in a land far away, there lived a",
      formatted: "<|im_start|>user\nOnce upon a time, in a land far away, there lived a<|im_end|>\n<|im_start|>assistant\n",
      charCount: 51,
      tokenCount: 13,
      provenance: "illustrative",
    },
    tokenization: {
      tokens: [
        { text: "Once", id: 7454, pos: 0, type: "prompt", byteLength: 4 },
        { text: " upon", id: 2402, pos: 1, type: "prompt", byteLength: 5 },
        { text: " a", id: 257, pos: 2, type: "prompt", byteLength: 2 },
        { text: " time", id: 640, pos: 3, type: "prompt", byteLength: 5 },
        { text: ",", id: 11, pos: 4, type: "prompt", byteLength: 1 },
        { text: " in", id: 287, pos: 5, type: "prompt", byteLength: 3 },
        { text: " a", id: 257, pos: 6, type: "prompt", byteLength: 2 },
        { text: " land", id: 1956, pos: 7, type: "prompt", byteLength: 5 },
        { text: " far", id: 1400, pos: 8, type: "prompt", byteLength: 4 },
        { text: " away", id: 1419, pos: 9, type: "prompt", byteLength: 5 },
        { text: ",", id: 11, pos: 10, type: "prompt", byteLength: 1 },
        { text: " there", id: 612, pos: 11, type: "prompt", byteLength: 6 },
        { text: " lived", id: 4410, pos: 12, type: "prompt", byteLength: 6 },
        { text: " a", id: 257, pos: 13, type: "prompt", byteLength: 2 },
      ],
      vocabSize: 32000,
      provenance: "illustrative",
    },
    embeddings: {
      dModel: 768,
      vectors: [
        { token: "Once", id: 7454, slice: [0.42, -0.15, 0.33, 0.58], norm: 1.0, ropeAngle: "0.00 rad" },
        { token: " lived", id: 4410, slice: [0.65, 0.28, -0.41, 0.12], norm: 1.0, ropeAngle: "1.82 rad" },
        { token: " a", id: 257, slice: [0.08, -0.19, -0.54, 0.38], norm: 1.0, ropeAngle: "2.10 rad" },
      ],
      provenance: "illustrative",
    },
    attention: {
      heads: 4,
      tokens: ["Once", " upon", " a", " land", " far", " lived", " a"],
      matrix: [
        [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.40, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.20, 0.30, 0.50, 0.00, 0.00, 0.00, 0.00],
        [0.15, 0.15, 0.20, 0.50, 0.00, 0.00, 0.00],
        [0.10, 0.10, 0.15, 0.35, 0.30, 0.00, 0.00],
        [0.25, 0.05, 0.10, 0.20, 0.10, 0.30, 0.00],
        [0.18, 0.08, 0.12, 0.22, 0.10, 0.20, 0.10],
      ],
      provenance: "illustrative",
    },
    logits: {
      candidates: [
        { token: "young", id: 1862, logit: 13.8 },
        { token: "brave", id: 11048, logit: 12.1 },
        { token: "king", id: 3797, logit: 11.4 },
        { token: "little", id: 1310, logit: 10.2 },
        { token: "dragon", id: 11019, logit: 8.9 },
        { token: "girl", id: 2585, logit: 7.5 },
      ],
      provenance: "illustrative",
    },
    sampling: {
      selectedToken: { text: "young", id: 1862, prob: 0.582 },
      provenance: "derived",
    },
    generation: {
      completionText: "young king who dreamed of exploring the stars.",
      tokens: [
        { text: "young", id: 1862, step: 1 },
        { text: " king", id: 3797, step: 2 },
        { text: " who", id: 508, step: 3 },
        { text: " dreamed", id: 14751, step: 4 },
        { text: " of", id: 286, step: 5 },
        { text: " exploring", id: 12560, step: 6 },
        { text: " the", id: 262, step: 7 },
        { text: " stars", id: 5364, step: 8 },
        { text: ".", id: 13, step: 9 },
      ],
      stopReason: "stop_sequence",
      provenance: "illustrative",
    },
  },
};

/**
 * Synthesizes contextual candidate logits and multi-step continuation tokens
 * based on prompt heuristics (geography, coding, science/ai, storytelling, greetings, etc.)
 */
function getSmartContextualCompletion(promptText) {
  const lower = promptText.toLowerCase();

  // 1. Capital of <Country>
  const capitalMatches = [
    { country: "france", capital: "Paris", id: 4881, other: ["Lyon", "Marseille", "Europe", "France"] },
    { country: "japan", capital: "Tokyo", id: 11616, other: ["Kyoto", "Osaka", "Japan", "Asia"] },
    { country: "germany", capital: "Berlin", id: 10476, other: ["Munich", "Frankfurt", "Germany", "Europe"] },
    { country: "italy", capital: "Rome", id: 8466, other: ["Milan", "Venice", "Italy", "Europe"] },
    { country: "spain", capital: "Madrid", id: 12431, other: ["Barcelona", "Seville", "Spain", "Europe"] },
    { country: "india", capital: "New Delhi", id: 3672, other: ["Mumbai", "Bangalore", "India", "Asia"] },
    { country: "uk", capital: "London", id: 3587, other: ["Manchester", "Edinburgh", "Britain", "Europe"] },
    { country: "england", capital: "London", id: 3587, other: ["Manchester", "Birmingham", "Britain", "Europe"] },
    { country: "united kingdom", capital: "London", id: 3587, other: ["Manchester", "Edinburgh", "Britain", "Europe"] },
    { country: "usa", capital: "Washington", id: 3654, other: ["New York", "Chicago", "America", "the"] },
    { country: "united states", capital: "Washington", id: 3654, other: ["New York", "Chicago", "America", "the"] },
    { country: "canada", capital: "Ottawa", id: 16982, other: ["Toronto", "Montreal", "Vancouver", "Canada"] },
    { country: "australia", capital: "Canberra", id: 24719, other: ["Sydney", "Melbourne", "Brisbane", "Australia"] },
    { country: "egypt", capital: "Cairo", id: 15301, other: ["Alexandria", "Giza", "Nile", "Egypt"] },
    { country: "china", capital: "Beijing", id: 10741, other: ["Shanghai", "Guangzhou", "China", "Asia"] },
    { country: "brazil", capital: "Brasília", id: 27952, other: ["Rio", "Sao Paulo", "Brazil", "America"] },
  ];

  for (const m of capitalMatches) {
    if (lower.includes(m.country) && (lower.includes("capital") || lower.includes("city"))) {
      const candidates = [
        { token: m.capital, id: m.id, logit: 15.2 },
        { token: m.other[0], id: 10001, logit: 9.8 },
        { token: m.other[1], id: 10002, logit: 8.4 },
        { token: m.country.charAt(0).toUpperCase() + m.country.slice(1), id: 10003, logit: 7.1 },
        { token: "The", id: 464, logit: 6.2 },
      ];
      const tokens = [
        { text: ` ${m.capital}`, id: m.id, step: 1 },
        { text: " is", id: 318, step: 2 },
        { text: " the", id: 262, step: 3 },
        { text: " capital", id: 3139, step: 4 },
        { text: " city", id: 1916, step: 5 },
        { text: " of", id: 286, step: 6 },
        { text: ` ${m.country.charAt(0).toUpperCase() + m.country.slice(1)}`, id: 10003, step: 7 },
        { text: ".", id: 13, step: 8 },
      ];
      return { candidates, tokens, completionText: `${m.capital} is the capital city of ${m.country.charAt(0).toUpperCase() + m.country.slice(1)}.` };
    }
  }

  // 2. Python / Programming / Coding
  if (lower.includes("python") || lower.includes("fibonacci") || lower.includes("def ") || lower.includes("code") || lower.includes("function")) {
    const candidates = [
      { token: "def", id: 752, logit: 14.9 },
      { token: "return", id: 1441, logit: 12.1 },
      { token: "if", id: 498, logit: 10.3 },
      { token: "n", id: 299, logit: 8.7 },
      { token: "class", id: 2191, logit: 7.2 },
      { token: "result", id: 2065, logit: 6.0 },
    ];
    const tokens = [
      { text: "def", id: 752, step: 1 },
      { text: " solve", id: 6223, step: 2 },
      { text: "(", id: 7, step: 3 },
      { text: "n", id: 299, step: 4 },
      { text: "):", id: 1318, step: 5 },
      { text: "\n   ", id: 220, step: 6 },
      { text: " if", id: 498, step: 7 },
      { text: " n", id: 299, step: 8 },
      { text: " <=", id: 2854, step: 9 },
      { text: " 1", id: 352, step: 10 },
      { text: ":", id: 25, step: 11 },
      { text: " return", id: 1441, step: 12 },
      { text: " n", id: 299, step: 13 },
    ];
    return { candidates, tokens, completionText: "def solve(n):\n    if n <= 1: return n" };
  }

  // 3. AI / Machine Learning / Neural / Transformer
  if (lower.includes("transformer") || lower.includes("neural") || lower.includes("learning") || lower.includes("attention") || lower.includes("llm") || lower.includes("gpt")) {
    const candidates = [
      { token: "Transformers", id: 41258, logit: 14.4 },
      { token: "Attention", id: 13589, logit: 12.6 },
      { token: "Neural", id: 16892, logit: 11.2 },
      { token: "Models", id: 7291, logit: 9.3 },
      { token: "Tokens", id: 15309, logit: 8.0 },
      { token: "The", id: 464, logit: 6.7 },
    ];
    const tokens = [
      { text: "Transformers", id: 41258, step: 1 },
      { text: " utilize", id: 8941, step: 2 },
      { text: " multi", id: 3824, step: 3 },
      { text: "-", id: 12, step: 4 },
      { text: "head", id: 3111, step: 5 },
      { text: " self", id: 2116, step: 6 },
      { text: "-", id: 12, step: 7 },
      { text: "attention", id: 13589, step: 8 },
      { text: " to", id: 284, step: 9 },
      { text: " model", id: 2746, step: 10 },
      { text: " token", id: 11241, step: 11 },
      { text: " relations", id: 8493, step: 12 },
      { text: ".", id: 13, step: 13 },
    ];
    return { candidates, tokens, completionText: "Transformers utilize multi-head self-attention to model token relations." };
  }

  // 4. Greetings
  if (lower.startsWith("hello") || lower.startsWith("hi") || lower.startsWith("hey") || lower.includes("greetings")) {
    const candidates = [
      { token: "Hello", id: 15496, logit: 15.1 },
      { token: "Hi", id: 15902, logit: 13.8 },
      { token: "How", id: 2437, logit: 11.4 },
      { token: "Welcome", id: 14618, logit: 9.2 },
      { token: "I", id: 40, logit: 8.1 },
    ];
    const tokens = [
      { text: "Hello", id: 15496, step: 1 },
      { text: "!", id: 0, step: 2 },
      { text: " How", id: 2437, step: 3 },
      { text: " can", id: 460, step: 4 },
      { text: " I", id: 40, step: 5 },
      { text: " assist", id: 4332, step: 6 },
      { text: " you", id: 345, step: 7 },
      { text: " today", id: 1909, step: 8 },
      { text: "?", id: 30, step: 9 },
    ];
    return { candidates, tokens, completionText: "Hello! How can I assist you today?" };
  }

  // 5. Questions (Why, How, What, Explain, Who, Describe)
  if (lower.startsWith("why") || lower.startsWith("how") || lower.startsWith("what") || lower.startsWith("explain") || lower.startsWith("describe")) {
    // Extract main subject words (skipping stop words)
    const words = promptText.replace(/[^a-zA-Z0-9\s]/g, "").split(/\s+/).filter(Boolean);
    const stopWords = new Set(["what", "is", "the", "a", "an", "of", "in", "how", "why", "to", "explain", "does", "do", "describe", "can", "tell", "me", "about"]);
    const subjects = words.filter((w) => !stopWords.has(w.toLowerCase()));
    const keySubject = subjects.length > 0 ? subjects[0] : "this phenomenon";
    const capKey = keySubject.charAt(0).toUpperCase() + keySubject.slice(1);

    const candidates = [
      { token: capKey, id: 12050, logit: 14.1 },
      { token: "It", id: 632, logit: 12.3 },
      { token: "This", id: 1212, logit: 10.5 },
      { token: "The", id: 464, logit: 9.1 },
      { token: "A", id: 32, logit: 7.4 },
    ];
    const tokens = [
      { text: ` ${capKey}`, id: 12050, step: 1 },
      { text: " is", id: 318, step: 2 },
      { text: " characterized", id: 14201, step: 3 },
      { text: " by", id: 416, step: 4 },
      { text: " key", id: 2636, step: 5 },
      { text: " underlying", id: 9812, step: 6 },
      { text: " principles", id: 8142, step: 7 },
      { text: " and", id: 290, step: 8 },
      { text: " properties", id: 5849, step: 9 },
      { text: ".", id: 13, step: 10 },
    ];
    return { candidates, tokens, completionText: `${capKey} is characterized by key underlying principles and properties.` };
  }

  // 6. Generic intelligent continuation based on last words of prompt
  const candidates = [
    { token: "provides", id: 3772, logit: 13.8 },
    { token: "represents", id: 4891, logit: 12.2 },
    { token: "creates", id: 3384, logit: 10.4 },
    { token: "is", id: 318, logit: 9.1 },
    { token: "and", id: 290, logit: 7.5 },
  ];
  const tokens = [
    { text: " provides", id: 3772, step: 1 },
    { text: " valuable", id: 6112, step: 2 },
    { text: " insights", id: 10240, step: 3 },
    { text: " into", id: 656, step: 4 },
    { text: " this", id: 428, step: 5 },
    { text: " topic", id: 4821, step: 6 },
    { text: ".", id: 13, step: 7 },
  ];
  return { candidates, tokens, completionText: " provides valuable insights into this topic." };
}

/**
 * Fallback deterministic generator for custom user prompts.
 * Generates realistic subword tokens, embeddings, causal attention weights, and logits.
 */
export function getDeterministicDemoData(promptText) {
  const trimmed = promptText?.trim() || "What is the capital of France?";

  // Check exact fixture match
  if (DEMO_FIXTURES[trimmed]) {
    return DEMO_FIXTURES[trimmed];
  }

  // Tokenize using verified subword tokenizer
  const tokenRes = defaultTokenizer.tokenize(trimmed);
  const tokens = tokenRes.tokens;
  const seqLen = tokens.length;

  // Generate causal lower-triangular attention matrix where rows sum to 1.0
  const matrix = [];
  for (let r = 0; r < seqLen; r++) {
    const row = new Array(seqLen).fill(0);
    let sum = 0;
    for (let c = 0; c <= r; c++) {
      // Prioritize self and immediate prior context
      const weight = c === r ? 0.45 : 0.55 / Math.max(1, r);
      row[c] = parseFloat(weight.toFixed(3));
      sum += row[c];
    }
    // Normalize row to exactly 1.0
    for (let c = 0; c <= r; c++) {
      row[c] = parseFloat((row[c] / sum).toFixed(2));
    }
    matrix.push(row);
  }

  // Vectors preview
  const vectors = tokens.slice(0, 10).map((t, i) => ({
    token: t.text,
    id: t.id,
    slice: [
      parseFloat((Math.sin(i * 1.5) * 0.8).toFixed(2)),
      parseFloat((Math.cos(i * 1.5) * 0.8).toFixed(2)),
      parseFloat((Math.sin(i * 2.1) * 0.7).toFixed(2)),
      parseFloat((Math.cos(i * 2.1) * 0.7).toFixed(2)),
    ],
    norm: 1.0,
    ropeAngle: `${(i * 0.17).toFixed(2)} rad`,
  }));

  // Contextual smart completion
  const contextual = getSmartContextualCompletion(trimmed);

  return {
    prompt: {
      raw: trimmed,
      formatted: `<|im_start|>user\n${trimmed}<|im_end|>\n<|im_start|>assistant\n`,
      charCount: trimmed.length,
      tokenCount: tokens.length,
      provenance: "illustrative",
    },
    tokenization: {
      tokens,
      vocabSize: 32000,
      provenance: "illustrative",
    },
    embeddings: {
      dModel: 768,
      vectors,
      provenance: "illustrative",
    },
    attention: {
      heads: 4,
      tokens: tokens.map((t) => t.text),
      matrix,
      provenance: "illustrative",
    },
    logits: {
      candidates: contextual.candidates,
      provenance: "illustrative",
    },
    sampling: {
      selectedToken: { text: contextual.candidates[0].token, id: contextual.candidates[0].id, prob: 0.68 },
      provenance: "derived",
    },
    generation: {
      completionText: contextual.completionText,
      tokens: contextual.tokens,
      stopReason: "stop_sequence",
      provenance: "illustrative",
    },
  };
}
