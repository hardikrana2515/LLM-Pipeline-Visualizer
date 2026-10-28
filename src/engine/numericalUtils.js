/**
 * LLM Pipeline Visualizer — Comprehensive Numerical Engine & Sampling Utilities
 *
 * Implements pure, testable mathematical operations for LLM decoding:
 * - Stable softmax with max-logit subtraction (prevents overflow & underflow)
 * - Temperature scaling (T=0 greedy, T>0 smooth)
 * - Greedy decoding (deterministic argmax)
 * - Top-K filtering (with explicit eligible vs removed separation)
 * - Top-P / Nucleus cumulative probability filtering
 * - Re-normalization guaranteeing probabilities sum to 1.0
 * - Categorical multinomial sampling with injected float or seeded PRNG
 * - Repetition penalty application
 * - Full pipeline trace generator for educational inspection
 */

/**
 * Numerically stable softmax with maximum subtraction.
 * @param {number[]} logits - Array of float scores
 * @returns {number[]} Normalized probabilities summing to 1.0
 */
export function stableSoftmax(logits) {
  if (!logits || !Array.isArray(logits) || logits.length === 0) return [];
  // Filter non-finite values safely
  const cleanLogits = logits.map((val) =>
    Number.isFinite(val) ? val : -1e9
  );
  const maxLogit = Math.max(...cleanLogits);
  const expScores = cleanLogits.map((z) => Math.exp(z - maxLogit));
  const sumExp = expScores.reduce((acc, val) => acc + val, 0);

  if (sumExp === 0 || !Number.isFinite(sumExp)) {
    return cleanLogits.map(() => 1 / cleanLogits.length);
  }
  return expScores.map((val) => val / sumExp);
}

/**
 * Scale logits by temperature T.
 * Clamps positive temperature to a minimum safe threshold to avoid division by zero.
 * @param {number[]} logits
 * @param {number} temperature
 * @returns {number[]}
 */
export function scaleLogits(logits, temperature) {
  if (!logits || !Array.isArray(logits)) return [];
  if (temperature <= 0) {
    // Greedy mode representation: large multiplier for relative separation
    return [...logits];
  }
  const safeTemp = Math.max(0.0001, temperature);
  return logits.map((z) => (Number.isFinite(z) ? z / safeTemp : z));
}

/**
 * Greedy decoding: chooses the candidate with the highest logit score.
 * Ties broken deterministically by first occurrence index.
 * @param {Array<{token: string, logit: number}>} candidates
 * @returns {{token: string, logit: number, index: number} | null}
 */
export function greedyDecode(candidates) {
  if (!candidates || candidates.length === 0) return null;
  let maxIdx = 0;
  let maxVal = candidates[0].logit;

  for (let i = 1; i < candidates.length; i++) {
    if (candidates[i].logit > maxVal) {
      maxVal = candidates[i].logit;
      maxIdx = i;
    }
  }
  return { ...candidates[maxIdx], index: maxIdx };
}

/**
 * Applies repetition penalty to candidate logits based on prior context token IDs.
 * Formula (Keskar et al.):
 * if logit > 0: logit / penalty
 * if logit <= 0: logit * penalty
 * @param {Array<{id: number, logit: number}>} candidates
 * @param {number[]} contextTokenIds
 * @param {number} penalty - Penalty factor (>= 1.0)
 * @returns {Array<{id: number, logit: number}>}
 */
export function applyRepetitionPenalty(candidates, contextTokenIds = [], penalty = 1.0) {
  if (!candidates || penalty <= 1.0 || !contextTokenIds || contextTokenIds.length === 0) {
    return candidates ? [...candidates] : [];
  }
  const contextSet = new Set(contextTokenIds);

  return candidates.map((cand) => {
    if (!contextSet.has(cand.id)) return { ...cand };
    const z = cand.logit;
    const penalized = z > 0 ? z / penalty : z * penalty;
    return { ...cand, logit: penalized, penalized: true };
  });
}

/**
 * Filters candidates to retain only the top K highest scores.
 * Separates eligible candidates from removed candidates.
 * @param {Array<{token: string, logit: number}>} candidates
 * @param {number} k
 * @returns {{ eligible: Array, removed: Array }}
 */
export function filterTopK(candidates, k) {
  if (!candidates || candidates.length === 0) {
    return { eligible: [], removed: [] };
  }
  const sorted = [...candidates].sort((a, b) => b.logit - a.logit);
  const safeK = Math.max(1, Math.min(k || 1, sorted.length));

  return {
    eligible: sorted.slice(0, safeK),
    removed: sorted.slice(safeK).map((c) => ({ ...c, filterReason: "top-k" })),
  };
}

/**
 * Filters candidates using nucleus (Top-P) cumulative probability threshold.
 * Keeps the smallest leading set whose cumulative probability reaches or exceeds p.
 * @param {Array<{token: string, logit: number, prob: number}>} candidates - Sorted descending by prob
 * @param {number} p - Cumulative threshold (0 < p <= 1.0)
 * @returns {{ eligible: Array, removed: Array }}
 */
export function filterTopP(candidates, p) {
  if (!candidates || candidates.length === 0) {
    return { eligible: [], removed: [] };
  }
  const sorted = [...candidates].sort((a, b) => b.prob - a.prob);
  const safeP = Math.max(0.001, Math.min(p || 1.0, 1.0));

  let cumulative = 0;
  const eligible = [];
  const removed = [];

  for (let i = 0; i < sorted.length; i++) {
    const item = sorted[i];
    if (cumulative < safeP || eligible.length === 0) {
      eligible.push(item);
      cumulative += item.prob;
    } else {
      removed.push({ ...item, filterReason: "top-p" });
    }
  }

  return { eligible, removed };
}

/**
 * Re-normalizes candidate probabilities so they sum to exactly 1.0.
 * @param {Array<{prob: number}>} candidates
 * @returns {Array<{prob: number, percentage: string}>}
 */
export function reNormalize(candidates) {
  if (!candidates || candidates.length === 0) return [];
  const sum = candidates.reduce((acc, c) => acc + (c.prob || 0), 0);

  if (sum === 0) {
    const equalProb = 1 / candidates.length;
    return candidates.map((c) => ({
      ...c,
      prob: equalProb,
      percentage: `${(equalProb * 100).toFixed(1)}%`,
    }));
  }

  return candidates.map((c) => {
    const norm = c.prob / sum;
    return {
      ...c,
      prob: norm,
      percentage: `${(norm * 100).toFixed(1)}%`,
    };
  });
}

/**
 * Categorical / multinomial sampling from a normalized probability distribution.
 * Supports injected random value (0.0 <= r < 1.0) for deterministic tests.
 * @param {Array<{token: string, prob: number}>} candidates - Must have normalized probabilities
 * @param {number} [injectedRandom] - Injected float in [0, 1)
 * @returns {Object} Selected candidate
 */
export function sampleCategorical(candidates, injectedRandom = null) {
  if (!candidates || candidates.length === 0) return null;
  if (candidates.length === 1) return candidates[0];

  const roll =
    injectedRandom !== null && Number.isFinite(injectedRandom)
      ? Math.max(0, Math.min(0.999999, injectedRandom))
      : Math.random();

  let cumulative = 0;
  for (let i = 0; i < candidates.length; i++) {
    cumulative += candidates[i].prob;
    if (roll < cumulative || i === candidates.length - 1) {
      return { ...candidates[i], sampleRoll: roll };
    }
  }

  return { ...candidates[candidates.length - 1], sampleRoll: roll };
}

/**
 * Simple Linear Congruential Generator (LCG) PRNG for repeatable seeded sampling.
 * @param {number} seed
 * @returns {() => number} Returns float in [0, 1)
 */
export function createSeededRandom(seed = 42) {
  let state = Math.abs(Math.floor(seed)) % 2147483647;
  if (state === 0) state = 1;

  return function next() {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

/**
 * Full end-to-end Sampling Pipeline Trace.
 * Executes the full sequence:
 * Raw Logits -> Repetition Penalty -> Temperature Scaling -> Softmax -> Top-K -> Top-P -> Re-normalization -> Selection
 * Returns complete trace for educational visualization.
 */
export function computeFullSamplingPipeline(
  candidates,
  {
    temperature = 0.7,
    topK = 50,
    topP = 0.9,
    repetitionPenalty = 1.0,
    contextTokenIds = [],
    randomRoll = null,
    strategy = "combined", // 'greedy' | 'temperature' | 'topk' | 'topp' | 'combined'
  } = {}
) {
  if (!candidates || candidates.length === 0) {
    return {
      raw: [],
      scaled: [],
      softmaxProbs: [],
      eligible: [],
      removed: [],
      cumulativeProbs: [],
      selected: null,
    };
  }

  // 1. Repetition penalty
  const penalized = applyRepetitionPenalty(
    candidates,
    contextTokenIds,
    repetitionPenalty
  );

  // 2. Pure Greedy branch
  if (strategy === "greedy" || temperature === 0) {
    const topCandidate = greedyDecode(penalized);
    const sorted = [...penalized].sort((a, b) => b.logit - a.logit);
    const eligible = sorted.map((c, idx) => ({
      ...c,
      prob: idx === 0 ? 1.0 : 0.0,
      percentage: idx === 0 ? "100.0%" : "0.0%",
      cumulativeProb: 1.0,
    }));

    return {
      raw: candidates,
      scaled: penalized,
      softmaxProbs: eligible,
      eligible,
      removed: [],
      cumulativeProbs: [{ token: topCandidate?.token, cumulative: 1.0 }],
      selected: topCandidate ? { ...topCandidate, prob: 1.0, percentage: "100.0%" } : null,
      temperature: 0,
      strategy: "greedy",
    };
  }

  // 3. Top-K filtering (if enabled by strategy)
  const applyK = strategy === "topk" || strategy === "combined";
  const { eligible: afterTopK, removed: topKRemoved } = applyK
    ? filterTopK(penalized, topK)
    : { eligible: [...penalized].sort((a, b) => b.logit - a.logit), removed: [] };

  // 4. Temperature scaling
  const logits = afterTopK.map((c) => c.logit);
  const scaledLogits = scaleLogits(logits, temperature);

  // 5. Softmax on remaining candidates
  const probs = stableSoftmax(scaledLogits);
  const withSoftmax = afterTopK.map((c, i) => ({
    ...c,
    scaledLogit: scaledLogits[i],
    prob: probs[i],
    percentage: `${(probs[i] * 100).toFixed(1)}%`,
  }));

  // 6. Top-P (nucleus) filtering (if enabled by strategy)
  const applyP = strategy === "topp" || strategy === "combined";
  const { eligible: afterTopP, removed: topPRemoved } = applyP
    ? filterTopP(withSoftmax, topP)
    : { eligible: withSoftmax, removed: [] };

  // 7. Re-normalize eligible candidates
  const normalizedEligible = reNormalize(afterTopP);

  // 8. Compute cumulative step function values
  let cum = 0;
  const cumulativeProbs = normalizedEligible.map((c) => {
    cum += c.prob;
    return {
      token: c.token,
      prob: c.prob,
      cumulative: Math.min(1.0, cum),
    };
  });

  // 9. Categorical sampling selection
  const selected = sampleCategorical(normalizedEligible, randomRoll);

  // Combined removed list
  const allRemoved = [...topKRemoved, ...topPRemoved];

  return {
    raw: candidates,
    scaled: afterTopK,
    softmaxProbs: withSoftmax,
    eligible: normalizedEligible,
    removed: allRemoved,
    cumulativeProbs,
    selected,
    temperature,
    topK,
    topP,
    strategy,
  };
}

/**
 * Computes cosine similarity between two numeric vectors.
 * @param {number[]} vecA
 * @param {number[]} vecB
 * @returns {number} Cosine similarity in range [-1.0, 1.0]
 */
export function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) return 0;
  const len = Math.min(vecA.length, vecB.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0 || !Number.isFinite(denom)) return 0;
  return Math.max(-1.0, Math.min(1.0, dot / denom));
}

/**
 * Computes Rotary Position Embedding (RoPE) angles for a given sequence position.
 * theta_i = position * base^(-2i / dim)
 * @param {number} dim - Total dimension (must be even, typically d_head e.g. 64 or 128)
 * @param {number} position - Sequence position index (0, 1, 2, ...)
 * @param {number} [base=10000] - Frequency base (standard 10,000)
 * @returns {number[]} Array of angles in radians for each 2D rotation plane
 */
export function computeRopeAngles(dim, position, base = 10000) {
  if (!dim || dim <= 0) return [];
  const numPlanes = Math.floor(dim / 2);
  const angles = [];
  for (let i = 0; i < numPlanes; i++) {
    const freq = Math.pow(base, -(2 * i) / dim);
    angles.push(position * freq);
  }
  return angles;
}

/**
 * Applies 2D rotation to a coordinate pair [x1, x2] by angle theta (RoPE core operation).
 * [x1', x2'] = [x1 * cos(theta) - x2 * sin(theta), x1 * sin(theta) + x2 * cos(theta)]
 * @param {number} x1
 * @param {number} x2
 * @param {number} theta - Angle in radians
 * @returns {[number, number]}
 */
export function applyRope2D(x1, x2, theta) {
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  return [x1 * cosT - x2 * sinT, x1 * sinT + x2 * cosT];
}

/**
 * Root Mean Square Normalization (RMSNorm) as used in LLaMA / Qwen / Mistral.
 * x_norm = x / sqrt(mean(x^2) + eps)
 * @param {number[]} vector
 * @param {number} [eps=1e-6]
 * @returns {number[]}
 */
export function computeRMSNorm(vector, eps = 1e-6) {
  if (!vector || !Array.isArray(vector) || vector.length === 0) return [];
  const meanSquare =
    vector.reduce((acc, val) => acc + val * val, 0) / vector.length;
  const rms = Math.sqrt(meanSquare + eps);
  return vector.map((val) => val / rms);
}

/**
 * Scaled dot-product multi-head attention scores.
 * Attention(Q, K) = softmax(Q K^T / sqrt(d_k) + M)
 * @param {number[][]} Q - Query vectors [seq_len, d_k]
 * @param {number[][]} K - Key vectors [seq_len, d_k]
 * @param {number} [dk=64] - Dimension of each head
 * @param {boolean} [isCausal=true] - Whether to apply lower-triangular causal mask
 * @returns {{rawScores: number[][], attentionWeights: number[][]}}
 */
export function computeAttentionScores(Q, K, dk = 64, isCausal = true) {
  if (!Q || !K || Q.length === 0 || K.length === 0) {
    return { rawScores: [], attentionWeights: [] };
  }
  const scale = 1 / Math.sqrt(Math.max(1, dk));
  const numQ = Q.length;
  const numK = K.length;
  const rawScores = [];
  const attentionWeights = [];

  for (let i = 0; i < numQ; i++) {
    const rowScores = [];
    for (let j = 0; j < numK; j++) {
      if (isCausal && j > i) {
        rowScores.push(-1e9); // Causal mask
      } else {
        const dot = Q[i].reduce((sum, val, idx) => sum + val * (K[j][idx] || 0), 0);
        rowScores.push(dot * scale);
      }
    }
    rawScores.push(rowScores);
    attentionWeights.push(stableSoftmax(rowScores));
  }

  return { rawScores, attentionWeights };
}

/**
 * Computes Cross-Entropy Loss for a target token given its predicted probability.
 * Loss = -ln(P(target))
 * @param {number} targetProb - Predicted probability in range (0, 1]
 * @returns {number} Non-negative scalar cross-entropy loss
 */
export function crossEntropyLoss(targetProb) {
  const safeProb = Math.max(1e-12, Math.min(1.0, targetProb || 0));
  return -Math.log(safeProb);
}

/**
 * Computes Perplexity (PPL) from cross-entropy loss.
 * PPL = exp(Loss) = 1 / P(target)
 * @param {number} loss - Cross-entropy loss
 * @returns {number}
 */
export function computePerplexity(loss) {
  if (!Number.isFinite(loss) || loss < 0) return 1.0;
  return Math.exp(Math.min(100, loss));
}

/**
 * Computes loss gradient at the output logits layer for next-token prediction.
 * dL / dz_i = P(y_i) - 1_{y_i == target}
 * @param {number} predictedProb - Model's predicted softmax probability
 * @param {boolean} isTarget - True if this candidate is the ground truth target token
 * @returns {number} Gradient delta (negative for target, positive for non-targets)
 */
export function computeOutputGradient(predictedProb, isTarget) {
  const p = Math.max(0, Math.min(1, predictedProb || 0));
  return p - (isTarget ? 1.0 : 0.0);
}

/**
 * Performs a single AdamW optimizer update step on a parameter weight.
 * Decoupled weight decay: w_{t+1} = w_t - lr * (m_hat / (sqrt(v_hat) + eps) + lambda * w_t)
 * @param {object} params
 * @param {number} params.weight - Current parameter weight
 * @param {number} params.grad - Gradient dL / dw
 * @param {number} [params.m=0] - First momentum state
 * @param {number} [params.v=0] - Second uncentered variance state
 * @param {number} [params.lr=0.0001] - Learning rate (eta)
 * @param {number} [params.beta1=0.9] - First moment decay rate
 * @param {number} [params.beta2=0.999] - Second moment decay rate
 * @param {number} [params.weightDecay=0.01] - Decoupled weight decay factor (lambda)
 * @param {number} [params.step=1] - Current optimization step index
 * @param {number} [params.eps=1e-8] - Numerical stability epsilon
 * @returns {{weight: number, m: number, v: number, update: number}}
 */
export function adamwStep({
  weight,
  grad,
  m = 0,
  v = 0,
  lr = 0.0001,
  beta1 = 0.9,
  beta2 = 0.999,
  weightDecay = 0.01,
  step = 1,
  eps = 1e-8,
}) {
  const safeStep = Math.max(1, step);
  // Update biased first moment
  const nextM = beta1 * m + (1 - beta1) * grad;
  // Update biased second raw moment
  const nextV = beta2 * v + (1 - beta2) * (grad * grad);

  // Compute bias-corrected moments
  const mHat = nextM / (1 - Math.pow(beta1, safeStep));
  const vHat = nextV / (1 - Math.pow(beta2, safeStep));

  // Compute adaptive update with decoupled weight decay
  const adaptiveStep = mHat / (Math.sqrt(Math.max(0, vHat)) + eps);
  const decayTerm = weightDecay * weight;
  const update = lr * (adaptiveStep + decayTerm);

  const nextWeight = weight - update;

  return {
    weight: nextWeight,
    m: nextM,
    v: nextV,
    update,
  };
}


