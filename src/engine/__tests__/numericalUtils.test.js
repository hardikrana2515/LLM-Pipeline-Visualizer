import { describe, it, expect } from "vitest";
import {
  stableSoftmax,
  scaleLogits,
  greedyDecode,
  applyRepetitionPenalty,
  filterTopK,
  filterTopP,
  reNormalize,
  sampleCategorical,
  createSeededRandom,
  computeFullSamplingPipeline,
  cosineSimilarity,
  computeRopeAngles,
  applyRope2D,
  computeRMSNorm,
  computeAttentionScores,
  crossEntropyLoss,
  computePerplexity,
  computeOutputGradient,
  adamwStep,
} from "../numericalUtils";

describe("Numerical Engine & Sampling Utilities", () => {
  const dummyCandidates = [
    { token: "Paris", id: 4881, logit: 14.2 },
    { token: "France", id: 4881, logit: 10.8 },
    { token: "Europe", id: 5931, logit: 8.4 },
    { token: "the", id: 262, logit: 6.9 },
    { token: "a", id: 64, logit: 5.1 },
    { token: "known", id: 2487, logit: 3.2 },
  ];

  it("stableSoftmax computes non-negative finite probabilities summing to 1.0", () => {
    const logits = [2.0, 1.0, 0.1];
    const probs = stableSoftmax(logits);
    const sum = probs.reduce((a, b) => a + b, 0);

    expect(sum).toBeCloseTo(1.0, 5);
    probs.forEach((p) => {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(p)).toBe(true);
    });
  });

  it("stableSoftmax handles extreme logits without NaN or Infinity", () => {
    const extremeLogits = [10000, 10005, 9995];
    const probs = stableSoftmax(extremeLogits);
    const sum = probs.reduce((a, b) => a + b, 0);

    expect(sum).toBeCloseTo(1.0, 5);
    expect(Number.isNaN(probs[0])).toBe(false);
  });

  it("scaleLogits scales by temperature properly", () => {
    const logits = [10, 5];
    const scaledLow = scaleLogits(logits, 0.5);
    expect(scaledLow).toEqual([20, 10]);

    const scaledHigh = scaleLogits(logits, 2.0);
    expect(scaledHigh).toEqual([5, 2.5]);
  });

  it("greedyDecode deterministically selects candidate with maximum score", () => {
    const chosen = greedyDecode(dummyCandidates);
    expect(chosen.token).toBe("Paris");
    expect(chosen.logit).toBe(14.2);
    expect(chosen.index).toBe(0);
  });

  it("applyRepetitionPenalty reduces scores for repeated context tokens", () => {
    const context = [4881]; // "Paris" ID
    const penalized = applyRepetitionPenalty(dummyCandidates, context, 1.5);
    const paris = penalized.find((c) => c.token === "Paris");
    const france = penalized.find((c) => c.token === "the");

    expect(paris.logit).toBeCloseTo(14.2 / 1.5, 4);
    expect(paris.penalized).toBe(true);
    expect(france.penalized).toBeUndefined();
  });

  it("filterTopK separates eligible from removed candidates", () => {
    const { eligible, removed } = filterTopK(dummyCandidates, 3);
    expect(eligible.length).toBe(3);
    expect(removed.length).toBe(3);
    expect(eligible.map((c) => c.token)).toEqual(["Paris", "France", "Europe"]);
    expect(removed[0].filterReason).toBe("top-k");
  });

  it("filterTopP separates eligible from removed based on cumulative threshold", () => {
    const candidatesWithProbs = [
      { token: "A", logit: 10, prob: 0.6 },
      { token: "B", logit: 8, prob: 0.25 },
      { token: "C", logit: 6, prob: 0.1 },
      { token: "D", logit: 4, prob: 0.05 },
    ];
    // Threshold 0.8: A (0.6) + B (0.25) = 0.85 >= 0.8 -> A and B eligible, C and D removed
    const { eligible, removed } = filterTopP(candidatesWithProbs, 0.8);
    expect(eligible.length).toBe(2);
    expect(removed.length).toBe(2);
    expect(eligible.map((c) => c.token)).toEqual(["A", "B"]);
    expect(removed[0].filterReason).toBe("top-p");
  });

  it("reNormalize ensures probabilities sum to exactly 1.0", () => {
    const partial = [
      { token: "A", prob: 0.4 },
      { token: "B", prob: 0.2 },
    ];
    const normalized = reNormalize(partial);
    const sum = normalized.reduce((acc, c) => acc + c.prob, 0);
    expect(sum).toBeCloseTo(1.0, 5);
    expect(normalized[0].prob).toBeCloseTo(0.4 / 0.6, 5);
    expect(normalized[1].prob).toBeCloseTo(0.2 / 0.6, 5);
  });

  it("sampleCategorical selects candidate deterministically with injected float", () => {
    const dist = [
      { token: "A", prob: 0.7 },
      { token: "B", prob: 0.2 },
      { token: "C", prob: 0.1 },
    ];
    // Roll 0.1 (< 0.7) -> selects "A"
    const rollLow = sampleCategorical(dist, 0.1);
    expect(rollLow.token).toBe("A");

    // Roll 0.75 (>= 0.7 and < 0.9) -> selects "B"
    const rollMid = sampleCategorical(dist, 0.75);
    expect(rollMid.token).toBe("B");

    // Roll 0.95 (>= 0.9) -> selects "C"
    const rollHigh = sampleCategorical(dist, 0.95);
    expect(rollHigh.token).toBe("C");
  });

  it("createSeededRandom generates deterministic repeatable sequence", () => {
    const rng1 = createSeededRandom(12345);
    const rng2 = createSeededRandom(12345);

    const val1 = [rng1(), rng1(), rng1()];
    const val2 = [rng2(), rng2(), rng2()];

    expect(val1).toEqual(val2);
    val1.forEach((v) => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    });
  });

  it("computeFullSamplingPipeline returns full trace with eligible sum to 1.0", () => {
    const trace = computeFullSamplingPipeline(dummyCandidates, {
      temperature: 0.7,
      topK: 4,
      topP: 0.95,
      randomRoll: 0.2,
    });

    expect(trace.raw.length).toBe(6);
    expect(trace.eligible.length).toBeLessThanOrEqual(4);
    expect(trace.selected).not.toBeNull();

    const sumEligible = trace.eligible.reduce((acc, c) => acc + c.prob, 0);
    expect(sumEligible).toBeCloseTo(1.0, 4);

    // Verify removed candidate is never selected
    const removedTokens = new Set(trace.removed.map((r) => r.token));
    expect(removedTokens.has(trace.selected.token)).toBe(false);
  });

  it("computeFullSamplingPipeline handles greedy mode with T=0", () => {
    const trace = computeFullSamplingPipeline(dummyCandidates, {
      temperature: 0,
      strategy: "greedy",
    });

    expect(trace.selected.token).toBe("Paris");
    expect(trace.selected.prob).toBe(1.0);
    expect(trace.selected.percentage).toBe("100.0%");
  });

  describe("Transformer Internal Mathematics (Phase 6)", () => {
    it("cosineSimilarity computes identical vectors as 1.0 and orthogonal vectors as 0", () => {
      const v1 = [1, 0, 0];
      const v2 = [1, 0, 0];
      const v3 = [0, 1, 0];
      const v4 = [-1, 0, 0];

      expect(cosineSimilarity(v1, v2)).toBeCloseTo(1.0, 5);
      expect(cosineSimilarity(v1, v3)).toBeCloseTo(0.0, 5);
      expect(cosineSimilarity(v1, v4)).toBeCloseTo(-1.0, 5);
    });

    it("computeRopeAngles generates correct frequency decays per 2D rotation plane", () => {
      const anglesPos0 = computeRopeAngles(64, 0);
      expect(anglesPos0.length).toBe(32);
      expect(anglesPos0.every((a) => a === 0)).toBe(true);

      const anglesPos2 = computeRopeAngles(64, 2);
      expect(anglesPos2[0]).toBeCloseTo(2.0, 5); // first plane freq = 10000^0 = 1, theta = 2 * 1 = 2
      expect(anglesPos2[31]).toBeLessThan(anglesPos2[0]); // frequencies decay with dimension
    });

    it("applyRope2D preserves vector magnitude during rotation", () => {
      const [x1, x2] = [3, 4]; // norm = 5
      const angle = Math.PI / 4;
      const [r1, r2] = applyRope2D(x1, x2, angle);
      const rotNorm = Math.sqrt(r1 * r1 + r2 * r2);
      expect(rotNorm).toBeCloseTo(5.0, 5);
    });

    it("computeRMSNorm normalizes vector to unit root-mean-square", () => {
      const vec = [2, 2, 2, 2]; // rms = 2
      const normed = computeRMSNorm(vec, 0);
      expect(normed).toEqual([1, 1, 1, 1]);
    });

    it("computeAttentionScores enforces causal mask on upper triangular entries", () => {
      // 3 tokens, d_k = 2
      const Q = [
        [1, 0],
        [0, 1],
        [1, 1],
      ];
      const K = [
        [1, 0],
        [0, 1],
        [1, 1],
      ];
      const { attentionWeights } = computeAttentionScores(Q, K, 2, true);

      // Token 0 can only attend to Token 0
      expect(attentionWeights[0][0]).toBeCloseTo(1.0, 5);
      expect(attentionWeights[0][1]).toBeCloseTo(0.0, 5);
      expect(attentionWeights[0][2]).toBeCloseTo(0.0, 5);

      // Token 1 can attend to Token 0 and Token 1, but not Token 2
      expect(attentionWeights[1][2]).toBeCloseTo(0.0, 5);
      expect(attentionWeights[1][0] + attentionWeights[1][1]).toBeCloseTo(1.0, 5);

      // Rows must sum to 1.0
      attentionWeights.forEach((row) => {
        const sum = row.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1.0, 5);
      });
    });
  });

  describe("Training Mathematics (Phase 7)", () => {
    it("crossEntropyLoss computes -ln(P) accurately", () => {
      expect(crossEntropyLoss(1.0)).toBeCloseTo(0.0, 5);
      expect(crossEntropyLoss(Math.E ** -2)).toBeCloseTo(2.0, 5);
      expect(crossEntropyLoss(0.5)).toBeCloseTo(Math.log(2), 5);
      // Handles 0 without crashing to NaN or undefined
      expect(crossEntropyLoss(0)).toBeGreaterThan(25);
    });

    it("computePerplexity calculates exp(Loss) = 1/P", () => {
      const loss = Math.log(4); // P = 0.25
      expect(computePerplexity(loss)).toBeCloseTo(4.0, 5);
      expect(computePerplexity(0)).toBeCloseTo(1.0, 5);
    });

    it("computeOutputGradient calculates P - 1 for target and P for non-target", () => {
      // If target token was predicted with 80% probability, gradient = 0.8 - 1 = -0.2 (push score up)
      expect(computeOutputGradient(0.8, true)).toBeCloseTo(-0.2, 5);
      // If non-target token was predicted with 15% probability, gradient = 0.15 - 0 = +0.15 (push score down)
      expect(computeOutputGradient(0.15, false)).toBeCloseTo(0.15, 5);
    });

    it("adamwStep applies momentum, variance, and decoupled weight decay", () => {
      const step1 = adamwStep({
        weight: 1.0,
        grad: 0.5,
        m: 0,
        v: 0,
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        weightDecay: 0.1,
        step: 1,
      });

      expect(step1.weight).toBeLessThan(1.0); // weight updated in descent direction
      expect(step1.m).toBeGreaterThan(0);
      expect(step1.v).toBeGreaterThan(0);
      expect(step1.update).toBeGreaterThan(0);
    });
  });
});

