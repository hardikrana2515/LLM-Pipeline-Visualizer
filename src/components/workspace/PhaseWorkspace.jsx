import { useState, useMemo, useEffect } from "react";
import Badge from "../common/Badge";
import { modelService } from "../../model/modelAdapter";
import { tokenizePrompt } from "../../tokenizer/subwordTokenizer";
import {
  stableSoftmax,
  scaleLogits,
  filterTopK,
  filterTopP,
  reNormalize,
  sampleCategorical,
  cosineSimilarity,
  computeRopeAngles,
  applyRope2D,
  computeRMSNorm,
  computeAttentionScores,
  crossEntropyLoss,
  computePerplexity,
  computeOutputGradient,
  adamwStep,
} from "../../engine/numericalUtils";
import "./PhaseWorkspace.css";

// Pastel chip color classes for tokens
const TOKEN_COLOR_CLASSES = [
  "token-chip--blue",
  "token-chip--green",
  "token-chip--lavender",
  "token-chip--cyan",
  "token-chip--slate",
  "token-chip--peach",
  "token-chip--blue",
];

// Synthetic embedding generator for educational display
function generateSyntheticEmbedding(tokenId, dim = 768) {
  const vec = [];
  let seed = (tokenId * 2654435761) ^ 0x9e3779b9;
  for (let i = 0; i < dim; i++) {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    const val = (seed / 4294967296) * 2 - 1;
    vec.push(parseFloat(val.toFixed(4)));
  }
  return vec;
}

// Synthetic QK generator
function generateSyntheticQK(tokens, headIndex, dk = 64) {
  const Q = [];
  const K = [];
  tokens.forEach((t, i) => {
    const qRow = [];
    const kRow = [];
    let seedQ = (t.id * 1664525 + headIndex * 1013904223 + i * 37) % 4294967296;
    let seedK = (t.id * 22695477 + headIndex * 999999 + i * 53) % 4294967296;
    for (let d = 0; d < dk; d++) {
      seedQ = (seedQ * 1103515245 + 12345) % 4294967296;
      seedK = (seedK * 1103515245 + 12345) % 4294967296;
      qRow.push((seedQ / 4294967296) * 2 - 1);
      kRow.push((seedK / 4294967296) * 2 - 1);
    }
    Q.push(qRow);
    K.push(kRow);
  });
  return { Q, K };
}

/**
 * PhaseWorkspace — Central Focused Learning Canvas.
 * Implements the minimal, editorial, phase-by-phase learning sequence.
 */
export default function PhaseWorkspace({
  stage,
  stageIndex,
  totalStages,
  prompt,
  demoData,
  mode,
  selectedTokenIndex,
  onSelectToken,
  onEditPrompt,
}) {
  // Phase 3 Embeddings states
  const [ropePosition, setRopePosition] = useState(2);
  const [activeEmbeddingTab, setActiveEmbeddingTab] = useState("vector");

  // Phase 4 Attention states
  const [activeHead, setActiveHead] = useState(0);
  const [isCausal, setIsCausal] = useState(true);

  // Phase 5 Logits states
  const [logitsViewMode, setLogitsViewMode] = useState("probabilities"); // 'probabilities' | 'raw'

  // Phase 6 Sampling states
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(4);
  const [topP, setTopP] = useState(0.9);
  const [manualRoll, setManualRoll] = useState(null);
  const [showAdvancedSampling, setShowAdvancedSampling] = useState(false);

  // Model state subscription
  const [modelState, setModelState] = useState(() => ({
    isLoaded: modelService.isLoaded,
    currentModelId: modelService.currentModelId,
  }));

  useEffect(() => {
    const unsub = modelService.subscribe((state) => {
      setModelState({
        isLoaded: state.status === "ready",
        currentModelId: state.currentModelId,
      });
    });
    return unsub;
  }, []);

  const isRealModel = modelState.isLoaded && modelState.currentModelId !== "demo-mode";

  // Phase 7 Generation states (dynamically bound to active prompt & demoData)
  const [generatedTokensList, setGeneratedTokensList] = useState(() => {
    if (demoData?.generation?.tokens && demoData.generation.tokens.length > 0) {
      return [demoData.generation.tokens[0]];
    }
    return [{ text: " Paris", id: 4881 }];
  });
  const [isAutoGenerating, setIsAutoGenerating] = useState(false);

  // Synchronize generated tokens with the active prompt and demoData
  useEffect(() => {
    if (demoData?.generation?.tokens && demoData.generation.tokens.length > 0) {
      setGeneratedTokensList([demoData.generation.tokens[0]]);
    } else {
      setGeneratedTokensList([]);
    }
  }, [prompt, demoData]);

  // Training Mode states
  const [trainingTargetProb, setTrainingTargetProb] = useState(0.42);
  const [trainingStep, setTrainingStep] = useState(1);
  const [simWeight, setSimWeight] = useState(0.75);
  const [simM, setSimM] = useState(0);
  const [simV, setSimV] = useState(0);

  // Technical Details Disclosure state
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  // Tokenize the prompt
  const tokens = useMemo(() => {
    return tokenizePrompt(prompt || "What is the capital of France?");
  }, [prompt]);

  // Candidate logits from demo data
  const candidateLogits = useMemo(() => {
    return (
      demoData?.logits?.candidates || [
        { token: "Paris", id: 4881, logit: 14.2 },
        { token: "France", id: 4881, logit: 10.8 },
        { token: "Europe", id: 5931, logit: 8.4 },
        { token: "the", id: 262, logit: 6.9 },
        { token: "a", id: 64, logit: 5.1 },
        { token: "known", id: 2487, logit: 3.2 },
      ]
    );
  }, [demoData]);

  // Live sampling trace
  const samplingTrace = useMemo(() => {
    const penalized = candidateLogits;
    const { eligible: afterTopK } = filterTopK(penalized, topK);
    const logits = afterTopK.map((c) => c.logit);
    const scaled = scaleLogits(logits, temperature);
    const probs = stableSoftmax(scaled);
    const withProbs = afterTopK.map((c, i) => ({
      ...c,
      scaledLogit: scaled[i],
      prob: probs[i],
      percentage: `${(probs[i] * 100).toFixed(1)}%`,
    }));
    const { eligible: afterTopP } = filterTopP(withProbs, topP);
    const normalized = reNormalize(afterTopP);
    const selected = sampleCategorical(normalized, manualRoll);

    return {
      eligible: normalized,
      selected: selected || normalized[0],
    };
  }, [candidateLogits, temperature, topK, topP, manualRoll]);

  const handleRerollSample = () => {
    setManualRoll(Math.random());
  };

  const handleGenerateNextToken = async () => {
    // 1. If a real neural model is loaded, run real inference!
    if (isRealModel) {
      try {
        setIsAutoGenerating(true);
        const currentGeneratedText = generatedTokensList.map((t) => t.text).join("");
        const fullPrompt = `${prompt}${currentGeneratedText}`;
        const res = await modelService.generate(fullPrompt, { maxNewTokens: 1, temperature });
        const newText = res.output.slice(fullPrompt.length) || res.output;
        if (newText) {
          setGeneratedTokensList((prev) => [
            ...prev,
            {
              text: newText.startsWith(" ") ? newText : ` ${newText}`,
              id: Math.floor(Math.random() * 50000),
            },
          ]);
        }
      } catch (err) {
        console.error("Real model generation error:", err);
      } finally {
        setIsAutoGenerating(false);
      }
      return;
    }

    // 2. Otherwise in Educational Simulation mode, advance through the prompt's sequence
    const sequence = demoData?.generation?.tokens || [];
    const currentIdx = generatedTokensList.length;

    if (currentIdx < sequence.length) {
      const nextTok = sequence[currentIdx];
      setGeneratedTokensList((prev) => [...prev, nextTok]);
    } else {
      // If sequence completed, pick the top sampled candidate
      const chosen = samplingTrace.selected?.token || ".";
      setGeneratedTokensList((prev) => [
        ...prev,
        {
          text: chosen.startsWith(" ") ? chosen : ` ${chosen}`,
          id: samplingTrace.selected?.id || Math.floor(Math.random() * 50000),
        },
      ]);
    }
  };

  const handleResetGeneration = () => {
    if (demoData?.generation?.tokens && demoData.generation.tokens.length > 0) {
      setGeneratedTokensList([demoData.generation.tokens[0]]);
    } else {
      setGeneratedTokensList([]);
    }
  };

  // Embeddings calculation for active token
  const tokenVectors = useMemo(() => {
    return tokens.map((t) => ({
      ...t,
      vector: generateSyntheticEmbedding(t.id, 768),
    }));
  }, [tokens]);

  const activeTokenVector =
    tokenVectors[selectedTokenIndex] || tokenVectors[0] || null;

  // RoPE rotation for Phase 3
  const ropeAngles = useMemo(() => computeRopeAngles(64, ropePosition, 10000), [ropePosition]);
  const activeRopeAngle = ropeAngles[0] || 0;
  const [ropeRotatedX, ropeRotatedY] = useMemo(
    () => applyRope2D(1.0, 0.0, activeRopeAngle),
    [activeRopeAngle]
  );

  // Attention scores for Phase 4
  const { attentionWeights } = useMemo(() => {
    const { Q, K } = generateSyntheticQK(tokens, activeHead, 64);
    return computeAttentionScores(Q, K, 64, isCausal);
  }, [tokens, activeHead, isCausal]);

  const currentStep = stageIndex + 1;

  // ============================================================
  // RENDER INFERENCE WORKSPACE PHASES
  // ============================================================
  return (
    <div className="phase-workspace" role="main" aria-label={`Phase: ${stage.title}`}>
      {/* Top Progress Track */}
      <div className="workspace-progress-row">
        <span className="workspace-step-label">
          Step {currentStep} of {totalStages}
        </span>
        <div className="workspace-step-track">
          <div
            className="workspace-step-bar"
            style={{ width: `${(currentStep / totalStages) * 100}%` }}
          />
        </div>
      </div>

      {/* Main Phase Header */}
      <div className="workspace-header">
        <h1 className="workspace-title text-display">{stage.title}</h1>
        <p className="workspace-intro">
          {mode === "inference" ? (
            stageIndex === 0 ? (
              "Every language model interaction begins with text input. This prompt becomes the context conditioning all subsequent predictions."
            ) : stageIndex === 1 ? (
              <>
                The input text is split into smaller pieces called <strong>tokens</strong>. These tokens are then converted into numerical IDs that the model can process.
              </>
            ) : stageIndex === 2 ? (
              <>
                Each discrete token ID is looked up in an embedding table to become a high-dimensional continuous vector. <strong>Rotary Position Embeddings (RoPE)</strong> add sequence order information.
              </>
            ) : stageIndex === 3 ? (
              <>
                The vectors pass through decoder Transformer blocks. <strong>Causal self-attention</strong> allows tokens to compare their queries and keys to route contextual information across previous positions.
              </>
            ) : stageIndex === 4 ? (
              <>
                The final hidden state vector is projected into raw, unnormalized scores across the entire vocabulary called <strong>logits</strong>.
              </>
            ) : stageIndex === 5 ? (
              <>
                <strong>Temperature</strong> scales the logits before softmax normalization. Filtering (Top-K / Top-P) prunes low-probability tokens before stochastic or greedy selection.
              </>
            ) : (
              <>
                The selected token is appended to the prompt. In an <strong>autoregressive loop</strong>, this expanded sequence becomes the input context for predicting the subsequent token.
              </>
            )
          ) : (
            // Training mode descriptions
            stageIndex === 0 ? (
              "Training begins with large corpora of natural text, prepared into contiguous sequences for next-token prediction."
            ) : stageIndex === 1 ? (
              "Input tokens and shifted target tokens are aligned: for each token at position t, the target is the subsequent token at t+1."
            ) : stageIndex === 2 ? (
              "The model processes the entire input sequence simultaneously in parallel to produce logits across all positions."
            ) : stageIndex === 3 ? (
              "Cross-entropy loss measures the discrepancy between predicted probability distributions and true target tokens."
            ) : stageIndex === 4 ? (
              "Loss gradients flow backward through the residual stream, calculating parameter error contributions via the chain rule."
            ) : (
              "An adaptive optimizer (AdamW) updates model weight matrices with decoupled weight decay, progressively reducing loss."
            )
          )}
        </p>
      </div>

      {/* ============================================================
          MAIN PHASE-SPECIFIC DEMONSTRATION CARDS
          ============================================================ */}

      {/* ── PHASE 1: INPUT PROMPT ── */}
      {mode === "inference" && stageIndex === 0 && (
        <div className="workspace-card">
          <div className="workspace-card__header">
            <span className="workspace-card__title">Input Text → Model Context</span>
            <Badge type="provenance" variant="real" label="REAL USER INPUT" />
          </div>
          <div className="workspace-card__body">
            <div className="prompt-display-box">
              <span className="prompt-display-quote">“</span>
              <p className="prompt-display-text">{prompt}</p>
              <span className="prompt-display-quote">”</span>
            </div>

            <div className="prompt-stats-row">
              <div className="stat-pill">
                <span className="stat-pill__label">Characters:</span>
                <span className="stat-pill__val text-mono">{prompt.length}</span>
              </div>
              <div className="stat-pill">
                <span className="stat-pill__label">Words:</span>
                <span className="stat-pill__val text-mono">{prompt.trim().split(/\s+/).length}</span>
              </div>
              <div className="stat-pill">
                <span className="stat-pill__label">Estimated Tokens:</span>
                <span className="stat-pill__val text-mono">{tokens.length}</span>
              </div>
              <button
                type="button"
                className="edit-prompt-pill-btn"
                onClick={onEditPrompt}
              >
                ✎ Change prompt
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── PHASE 2: TOKENIZATION (Exact match to reference design!) ── */}
      {mode === "inference" && stageIndex === 1 && (
        <div className="workspace-card">
          <div className="workspace-card__header">
            <span className="workspace-card__title">Text → Tokens</span>
            <Badge type="provenance" variant="illustrative" label="Illustrative" />
          </div>
          <div className="workspace-card__body">
            {/* Input String Box */}
            <div className="token-input-box">
              <code className="token-input-text">{prompt}</code>
            </div>

            {/* Down Arrow */}
            <div className="flow-down-arrow" aria-hidden="true">
              ↓
            </div>

            {/* Token Chips with ID Pills */}
            <div className="token-chips-grid">
              {tokens.map((token, idx) => {
                const isSelected = idx === selectedTokenIndex;
                const colorClass = isSelected
                  ? "token-chip--peach"
                  : TOKEN_COLOR_CLASSES[idx % TOKEN_COLOR_CLASSES.length];

                return (
                  <div
                    key={idx}
                    className={`token-item-wrap ${isSelected ? "token-item-wrap--selected" : ""}`}
                    onClick={() => onSelectToken(idx)}
                    role="button"
                    tabIndex={0}
                    aria-label={`Token: ${token.text}, ID: ${token.id}`}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") onSelectToken(idx);
                    }}
                  >
                    <div className={`token-chip ${colorClass}`}>
                      <span className="token-chip__text">{token.text}</span>
                    </div>
                    <div className="token-id-pill">
                      <span className="token-id-pill__number text-mono">{token.id}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── PHASE 3: EMBEDDINGS & ROPE GEOMETRY ── */}
      {mode === "inference" && stageIndex === 2 && (
        <div className="workspace-card">
          <div className="workspace-card__header">
            <span className="workspace-card__title">Token IDs → 768-D Vectors & RoPE Rotation</span>
            <Badge type="provenance" variant="illustrative" label="ILLUSTRATIVE (768-D Synthetic)" />
          </div>
          <div className="workspace-card__body">
            {/* Sub-tab Switcher */}
            <div className="card-subtabs">
              <button
                type="button"
                className={`card-subtab-btn ${activeEmbeddingTab === "vector" ? "card-subtab-btn--active" : ""}`}
                onClick={() => setActiveEmbeddingTab("vector")}
              >
                Vector Coordinates
              </button>
              <button
                type="button"
                className={`card-subtab-btn ${activeEmbeddingTab === "rope" ? "card-subtab-btn--active" : ""}`}
                onClick={() => setActiveEmbeddingTab("rope")}
              >
                RoPE 2D Rotation Plane
              </button>
              <button
                type="button"
                className={`card-subtab-btn ${activeEmbeddingTab === "similarity" ? "card-subtab-btn--active" : ""}`}
                onClick={() => setActiveEmbeddingTab("similarity")}
              >
                Cosine Similarity Matrix
              </button>
            </div>

            {activeEmbeddingTab === "vector" && (
              <div className="embedding-vector-view">
                <div className="vector-token-summary">
                  <span className="vector-token-label">Active token:</span>
                  <span className="token-chip token-chip--peach">{tokens[selectedTokenIndex]?.text || "France"}</span>
                  <span className="vector-norm-label text-muted">
                    L2 Norm: <strong className="text-mono">{(27.7).toFixed(1)}</strong>
                  </span>
                </div>

                <div className="vector-slice-grid">
                  {activeTokenVector?.vector.slice(0, 8).map((val, dimIdx) => (
                    <div key={dimIdx} className="vector-dim-cell">
                      <span className="dim-index text-mono">d_{dimIdx}</span>
                      <span className="dim-value text-mono">{val > 0 ? `+${val}` : val}</span>
                      <div
                        className="dim-meter"
                        style={{
                          width: `${Math.abs(val) * 100}%`,
                          background: val >= 0 ? "var(--primary)" : "var(--amber)",
                        }}
                      />
                    </div>
                  ))}
                </div>
                <p className="vector-footnote text-muted">
                  Showing first 8 of 768 dimensions. Embeddings map discrete vocabulary IDs into continuous semantic coordinates.
                </p>
              </div>
            )}

            {activeEmbeddingTab === "rope" && (
              <div className="rope-interactive-view">
                <div className="rope-controls-row">
                  <label className="rope-label">Sequence Position (m):</label>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    value={ropePosition}
                    onChange={(e) => setRopePosition(parseInt(e.target.value, 10))}
                    className="rope-slider"
                  />
                  <span className="rope-val-badge text-mono">m = {ropePosition}</span>
                </div>

                <div className="rope-canvas-wrap">
                  <svg className="rope-svg" viewBox="-120 -120 240 240">
                    <circle cx="0" cy="0" r="90" fill="none" stroke="var(--border-default)" strokeWidth="1" strokeDasharray="3 3" />
                    <line x1="-100" y1="0" x2="100" y2="0" stroke="var(--border-subtle)" strokeWidth="1" />
                    <line x1="0" y1="-100" x2="0" y2="100" stroke="var(--border-subtle)" strokeWidth="1" />
                    {/* Unrotated Vector */}
                    <line x1="0" y1="0" x2="90" y2="0" stroke="var(--border-strong)" strokeWidth="2" strokeDasharray="4 2" />
                    <circle cx="90" cy="0" r="4" fill="var(--border-strong)" />
                    {/* Rotated Vector */}
                    <line
                      x1="0"
                      y1="0"
                      x2={ropeRotatedX * 90}
                      y2={-ropeRotatedY * 90}
                      stroke="var(--primary)"
                      strokeWidth="3"
                    />
                    <circle cx={ropeRotatedX * 90} cy={-ropeRotatedY * 90} r="5" fill="var(--primary)" />
                  </svg>
                  <div className="rope-math-card">
                    <span className="rope-math-title">Rotary Rotation Formula</span>
                    <code className="rope-math-code text-mono">
                      [x1', x2'] = [x1 cos(mθ) - x2 sin(mθ), x1 sin(mθ) + x2 cos(mθ)]
                    </code>
                    <p className="rope-math-desc text-muted">
                      Angle θ = {activeRopeAngle.toFixed(3)} rad. Rotary position embedding rotates pairs of coordinates without altering vector magnitude.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {activeEmbeddingTab === "similarity" && (
              <div className="similarity-matrix-view">
                <table className="similarity-table">
                  <thead>
                    <tr>
                      <th className="th-empty"></th>
                      {tokens.map((t, j) => (
                        <th key={j} className="sim-token-th text-mono">{t.text}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tokens.map((tRow, i) => (
                      <tr key={i}>
                        <td className="sim-token-td text-mono">{tRow.text}</td>
                        {tokens.map((tCol, j) => {
                          const sim = cosineSimilarity(
                            tokenVectors[i]?.vector || [],
                            tokenVectors[j]?.vector || []
                          );
                          const isSelf = i === j;
                          return (
                            <td
                              key={j}
                              className={`sim-cell text-mono ${isSelf ? "sim-cell--self" : ""}`}
                              style={{
                                background: isSelf
                                  ? "var(--primary-soft)"
                                  : `rgba(27, 77, 62, ${Math.max(0.04, Math.abs(sim) * 0.4)})`,
                              }}
                            >
                              {sim.toFixed(2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── PHASE 4: TRANSFORMER & CAUSAL ATTENTION ── */}
      {mode === "inference" && stageIndex === 3 && (
        <div className="workspace-card">
          <div className="workspace-card__header">
            <span className="workspace-card__title">Causal Multi-Head Self-Attention</span>
            <Badge type="provenance" variant="illustrative" label="ILLUSTRATIVE (Multi-Head)" />
          </div>
          <div className="workspace-card__body">
            {/* Attention Controls */}
            <div className="attention-bar-controls">
              <div className="head-selector-chips">
                <span className="head-label">Head:</span>
                {[0, 1, 2, 3, 4, 5, 6, 7].map((h) => (
                  <button
                    key={h}
                    type="button"
                    className={`head-chip ${activeHead === h ? "head-chip--active" : ""}`}
                    onClick={() => setActiveHead(h)}
                  >
                    H{h + 1}
                  </button>
                ))}
              </div>
              <label className="causal-toggle">
                <input
                  type="checkbox"
                  checked={isCausal}
                  onChange={(e) => setIsCausal(e.target.checked)}
                />
                <span>Causal Mask (No future peeking)</span>
              </label>
            </div>

            {/* Interactive Attention Matrix Heatmap */}
            <div className="attention-heatmap-wrap">
              <table className="attention-matrix-table">
                <thead>
                  <tr>
                    <th className="th-corner">Query ↓ \ Key →</th>
                    {tokens.map((t, j) => (
                      <th key={j} className="att-token-th text-mono">{t.text}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tokens.map((qTok, i) => (
                    <tr key={i}>
                      <td className="att-query-td text-mono">{qTok.text}</td>
                      {tokens.map((kTok, j) => {
                        const weight = attentionWeights[i] ? attentionWeights[i][j] || 0 : 0;
                        const isMasked = isCausal && j > i;

                        return (
                          <td
                            key={j}
                            className={`att-cell text-mono ${isMasked ? "att-cell--masked" : ""}`}
                            style={{
                              backgroundColor: isMasked
                                ? "var(--bg-surface-subtle)"
                                : `rgba(27, 77, 62, ${Math.max(0.06, weight * 0.9)})`,
                              color: isMasked
                                ? "var(--text-disabled)"
                                : weight > 0.4 ? "#ffffff" : "var(--text-primary)",
                            }}
                            title={`Query: "${qTok.text}" -> Key: "${kTok.text}": ${(weight * 100).toFixed(1)}%`}
                          >
                            {isMasked ? "0.00" : weight.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="card-footnote text-muted">
              Lower-triangular causal attention ensures that position <em>t</em> can only attend to positions <em>≤ t</em>.
            </p>
          </div>
        </div>
      )}

      {/* ── PHASE 5: LOGITS (UNNORMALIZED SCORES) ── */}
      {mode === "inference" && stageIndex === 4 && (
        <div className="workspace-card">
          <div className="workspace-card__header">
            <span className="workspace-card__title">Un-embedding LM Head → Next-Token Logits</span>
            <Badge type="provenance" variant="derived" label="DERIVED SCORES" />
          </div>
          <div className="workspace-card__body">
            {/* View Mode Toggle */}
            <div className="logits-view-toggle">
              <button
                type="button"
                className={`strat-chip ${logitsViewMode === "probabilities" ? "strat-chip--active" : ""}`}
                onClick={() => setLogitsViewMode("probabilities")}
              >
                Softmax Probabilities (p ∈ [0, 1])
              </button>
              <button
                type="button"
                className={`strat-chip ${logitsViewMode === "raw" ? "strat-chip--active" : ""}`}
                onClick={() => setLogitsViewMode("raw")}
              >
                Raw Logits (z ∈ ℝ)
              </button>
            </div>

            {/* Candidates Bar Chart */}
            <div className="candidates-barchart">
              {candidateLogits.map((cand, idx) => {
                const maxLogit = 16.0;
                const prob = stableSoftmax(candidateLogits.map((c) => c.logit))[idx];
                const percent = (prob * 100).toFixed(1);
                const isTop = idx === 0;

                return (
                  <div key={idx} className="candidate-bar-row">
                    <div className="candidate-bar-label">
                      <span className={`candidate-token-badge ${isTop ? "candidate-token-badge--top" : ""}`}>
                        {cand.token}
                      </span>
                      <span className="candidate-id text-mono">id: {cand.id}</span>
                    </div>

                    <div className="candidate-bar-track">
                      <div
                        className="candidate-bar-fill"
                        style={{
                          width: logitsViewMode === "probabilities"
                            ? `${percent}%`
                            : `${Math.min(100, (cand.logit / maxLogit) * 100)}%`,
                          background: isTop ? "var(--primary)" : "var(--primary-border)",
                        }}
                      />
                    </div>

                    <div className="candidate-bar-score text-mono">
                      {logitsViewMode === "probabilities" ? `${percent}%` : cand.logit.toFixed(1)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── PHASE 6: SAMPLING & TEMPERATURE ── */}
      {mode === "inference" && stageIndex === 5 && (
        <div className="workspace-card">
          <div className="workspace-card__header">
            <span className="workspace-card__title">Temperature Scaling & Probability Sampling</span>
            <Badge type="provenance" variant="derived" label="LIVE DERIVED" />
          </div>
          <div className="workspace-card__body">
            {/* Temperature Slider & Action Row */}
            <div className="temperature-control-box">
              <div className="temp-slider-group">
                <div className="temp-slider-header">
                  <label className="temp-label">Temperature (T):</label>
                  <span className="temp-val-badge text-mono">T = {temperature.toFixed(2)}</span>
                  <span className="temp-flavor text-muted">
                    {temperature === 0
                      ? "Deterministic / Greedy (Argmax)"
                      : temperature < 0.5
                      ? "Focused & Accurate"
                      : temperature <= 0.8
                      ? "Balanced Creative"
                      : "High Variance / Diverse"}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1.8"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="sampling-slider"
                />
              </div>

              <button
                type="button"
                className="sample-reroll-btn"
                onClick={handleRerollSample}
              >
                <span>🎲 Sample / Reroll</span>
                {samplingTrace.selected?.sampleRoll !== undefined && (
                  <span className="roll-subtext text-mono">
                    r = {samplingTrace.selected.sampleRoll.toFixed(3)}
                  </span>
                )}
              </button>
            </div>

            {/* Presets */}
            <div className="temp-presets-row">
              <span className="presets-label">Presets:</span>
              {[
                { label: "Greedy (0.0)", val: 0.0 },
                { label: "Focused (0.3)", val: 0.3 },
                { label: "Balanced (0.7)", val: 0.7 },
                { label: "Creative (1.2)", val: 1.2 },
              ].map((p) => (
                <button
                  key={p.label}
                  type="button"
                  className={`preset-btn ${temperature === p.val ? "preset-btn--active" : ""}`}
                  onClick={() => setTemperature(p.val)}
                >
                  {p.label}
                </button>
              ))}
            </div>

            {/* Live Sampled Distribution */}
            <div className="candidates-barchart" style={{ marginTop: "16px" }}>
              {samplingTrace.eligible.map((item, idx) => {
                const isSelected = item.token === samplingTrace.selected?.token;

                return (
                  <div key={idx} className="candidate-bar-row">
                    <div className="candidate-bar-label">
                      <span className={`candidate-token-badge ${isSelected ? "candidate-token-badge--selected" : ""}`}>
                        {item.token}
                        {isSelected && <span className="selected-star"> ★ Selected</span>}
                      </span>
                    </div>

                    <div className="candidate-bar-track">
                      <div
                        className="candidate-bar-fill"
                        style={{
                          width: `${item.prob * 100}%`,
                          background: isSelected ? "var(--peach)" : "var(--primary)",
                        }}
                      />
                    </div>

                    <div className="candidate-bar-score text-mono">{item.percentage}</div>
                  </div>
                );
              })}
            </div>

            {/* Advanced Sampling Disclosure */}
            <div className="advanced-sampling-disclosure">
              <button
                type="button"
                className="disclosure-toggle-btn"
                onClick={() => setShowAdvancedSampling(!showAdvancedSampling)}
              >
                <span>{showAdvancedSampling ? "▾ Hide" : "▸ Show"} Top-K and Top-P filtering controls</span>
              </button>
              {showAdvancedSampling && (
                <div className="advanced-sampling-panel">
                  <div className="slider-row">
                    <label>Top-K ({topK}):</label>
                    <input
                      type="range"
                      min="1"
                      max="6"
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                    />
                  </div>
                  <div className="slider-row">
                    <label>Top-P / Nucleus ({topP.toFixed(2)}):</label>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={topP}
                      onChange={(e) => setTopP(parseFloat(e.target.value))}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── PHASE 7: GENERATED OUTPUT & AUTOREGRESSIVE LOOP ── */}
      {mode === "inference" && stageIndex === 6 && (
        <div className="workspace-card">
          <div className="workspace-card__header">
            <span className="workspace-card__title">Autoregressive Loop & Appended Tokens</span>
            <Badge
              type="provenance"
              variant={isRealModel ? "real" : "illustrative"}
              label={isRealModel ? "REAL MODEL INFERENCE" : "ILLUSTRATIVE DEMO"}
            />
          </div>
          <div className="workspace-card__body">
            {/* Generated Text Box */}
            <div className="generation-result-box">
              <span className="gen-prompt-text">{prompt}</span>
              <span className="gen-new-tokens">
                {generatedTokensList.map((tok, i) => (
                  <span key={i} className="gen-token-chip">
                    {tok.text}
                  </span>
                ))}
              </span>
              <span className="blinking-cursor" aria-hidden="true">▍</span>
            </div>

            {/* Loop Actions */}
            <div className="gen-actions-row">
              <button
                type="button"
                className="gen-next-btn"
                onClick={handleGenerateNextToken}
                disabled={isAutoGenerating}
              >
                {isAutoGenerating ? "Generating..." : "+ Generate Next Token"}
              </button>
              <button
                type="button"
                className="gen-reset-btn"
                onClick={handleResetGeneration}
                disabled={isAutoGenerating}
              >
                Reset Generated Tokens
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ============================================================
          TRAINING MODE PHASES
          ============================================================ */}
      {mode === "training" && (
        <div className="workspace-card">
          <div className="workspace-card__header">
            <span className="workspace-card__title">Training Phase: {stage.title}</span>
            <Badge type="provenance" variant="simulated" label="TRAINING SIMULATION" />
          </div>
          <div className="workspace-card__body">
            {stageIndex === 1 ? (
              // Sequence Shift
              <div className="training-shift-view">
                <p className="training-note">Input sequence shifted by 1 position against target tokens:</p>
                <div className="shift-table-wrap">
                  <div className="shift-row">
                    <span className="shift-label">Inputs (X):</span>
                    {tokens.slice(0, tokens.length - 1).map((t, idx) => (
                      <span key={idx} className="shift-chip shift-chip--in text-mono">{t.text}</span>
                    ))}
                  </div>
                  <div className="shift-arrow-row">↓ shifted by 1 ↓</div>
                  <div className="shift-row">
                    <span className="shift-label">Targets (Y):</span>
                    {tokens.slice(1).map((t, idx) => (
                      <span key={idx} className="shift-chip shift-chip--target text-mono">{t.text}</span>
                    ))}
                  </div>
                </div>
              </div>
            ) : stageIndex === 3 ? (
              // Loss curve
              <div className="loss-interactive-view">
                <div className="slider-row">
                  <label>Predicted Target Probability P(y):</label>
                  <input
                    type="range"
                    min="0.05"
                    max="0.99"
                    step="0.01"
                    value={trainingTargetProb}
                    onChange={(e) => setTrainingTargetProb(parseFloat(e.target.value))}
                  />
                  <span className="text-mono">{(trainingTargetProb * 100).toFixed(0)}%</span>
                </div>
                <div className="loss-readout-row">
                  <div className="readout-card">
                    <span>Cross-Entropy Loss (-ln P):</span>
                    <strong className="text-mono">{crossEntropyLoss(trainingTargetProb).toFixed(3)}</strong>
                  </div>
                  <div className="readout-card">
                    <span>Perplexity (e^Loss):</span>
                    <strong className="text-mono">{computePerplexity(crossEntropyLoss(trainingTargetProb)).toFixed(2)}</strong>
                  </div>
                </div>
              </div>
            ) : stageIndex === 5 ? (
              // Optimizer
              <div className="optimizer-sim-view">
                <div className="optimizer-stats-grid">
                  <div className="readout-card">
                    <span>Step:</span>
                    <strong className="text-mono">{trainingStep}</strong>
                  </div>
                  <div className="readout-card">
                    <span>Parameter Weight:</span>
                    <strong className="text-mono">{simWeight.toFixed(4)}</strong>
                  </div>
                  <div className="readout-card">
                    <span>1st Moment (m):</span>
                    <strong className="text-mono">{simM.toFixed(4)}</strong>
                  </div>
                </div>
                <button
                  type="button"
                  className="step-adamw-btn"
                  onClick={() => {
                    const nextStep = trainingStep + 1;
                    const res = adamwStep({
                      weight: simWeight,
                      grad: 0.2 / Math.sqrt(nextStep),
                      m: simM,
                      v: simV,
                      step: nextStep,
                    });
                    setTrainingStep(nextStep);
                    setSimWeight(res.weight);
                    setSimM(res.m);
                    setSimV(res.v);
                  }}
                >
                  ⚡ Perform AdamW Step
                </button>
              </div>
            ) : (
              <p className="training-generic-text">
                Training optimizes billions of parameters through continuous backpropagation and gradient descent updates.
              </p>
            )}
          </div>
        </div>
      )}

      {/* ============================================================
          PROCESS FLOW (3 Connected Cards: Input -> Transformation -> Output)
          ============================================================ */}
      <div className="workspace-flow-section">
        <h3 className="workspace-flow-title">
          {mode === "inference"
            ? stageIndex === 1 ? "Token Flow" : "Stage Data Pipeline"
            : "Training Flow"}
        </h3>
        <div className="flow-cards-track">
          {mode === "inference" ? (
            stageIndex === 1 ? (
              <>
                <div className="flow-step-card">
                  <span className="flow-card-icon">📄</span>
                  <div>
                    <h4 className="flow-card-title">Text</h4>
                    <p className="flow-card-desc">Raw input</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card flow-step-card--active">
                  <span className="flow-card-icon">✂️</span>
                  <div>
                    <h4 className="flow-card-title">Tokenizer</h4>
                    <p className="flow-card-desc">Splits & encodes</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card">
                  <span className="flow-card-icon">⚅</span>
                  <div>
                    <h4 className="flow-card-title">Token IDs</h4>
                    <p className="flow-card-desc">Numerical input</p>
                  </div>
                </div>
              </>
            ) : stageIndex === 2 ? (
              <>
                <div className="flow-step-card">
                  <span className="flow-card-icon">⚅</span>
                  <div>
                    <h4 className="flow-card-title">Token IDs</h4>
                    <p className="flow-card-desc">Integer IDs</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card flow-step-card--active">
                  <span className="flow-card-icon">📐</span>
                  <div>
                    <h4 className="flow-card-title">Embedding Table</h4>
                    <p className="flow-card-desc">Lookup & RoPE</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card">
                  <span className="flow-card-icon">🌐</span>
                  <div>
                    <h4 className="flow-card-title">768-D Vectors</h4>
                    <p className="flow-card-desc">Continuous space</p>
                  </div>
                </div>
              </>
            ) : stageIndex === 3 ? (
              <>
                <div className="flow-step-card">
                  <span className="flow-card-icon">🌐</span>
                  <div>
                    <h4 className="flow-card-title">Vectors</h4>
                    <p className="flow-card-desc">Input states</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card flow-step-card--active">
                  <span className="flow-card-icon">🧠</span>
                  <div>
                    <h4 className="flow-card-title">Self-Attention</h4>
                    <p className="flow-card-desc">Q, K, V dot product</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card">
                  <span className="flow-card-icon">⚡</span>
                  <div>
                    <h4 className="flow-card-title">Contextual Vectors</h4>
                    <p className="flow-card-desc">Mixed representations</p>
                  </div>
                </div>
              </>
            ) : stageIndex === 4 ? (
              <>
                <div className="flow-step-card">
                  <span className="flow-card-icon">⚡</span>
                  <div>
                    <h4 className="flow-card-title">Final Hidden State</h4>
                    <p className="flow-card-desc">Dimension d=768</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card flow-step-card--active">
                  <span className="flow-card-icon">📊</span>
                  <div>
                    <h4 className="flow-card-title">LM Head Matrix</h4>
                    <p className="flow-card-desc">Project to Vocab</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card">
                  <span className="flow-card-icon">🎯</span>
                  <div>
                    <h4 className="flow-card-title">Raw Logits</h4>
                    <p className="flow-card-desc">Vector size V</p>
                  </div>
                </div>
              </>
            ) : stageIndex === 5 ? (
              <>
                <div className="flow-step-card">
                  <span className="flow-card-icon">🎯</span>
                  <div>
                    <h4 className="flow-card-title">Raw Logits</h4>
                    <p className="flow-card-desc">Unscaled scores</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card flow-step-card--active">
                  <span className="flow-card-icon">🎲</span>
                  <div>
                    <h4 className="flow-card-title">Temp & Softmax</h4>
                    <p className="flow-card-desc">Scale & sample</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card">
                  <span className="flow-card-icon">✨</span>
                  <div>
                    <h4 className="flow-card-title">Chosen Token</h4>
                    <p className="flow-card-desc">Next token ID</p>
                  </div>
                </div>
              </>
            ) : stageIndex === 6 ? (
              <>
                <div className="flow-step-card">
                  <span className="flow-card-icon">✨</span>
                  <div>
                    <h4 className="flow-card-title">New Token</h4>
                    <p className="flow-card-desc">From sampling</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card flow-step-card--active">
                  <span className="flow-card-icon">🔄</span>
                  <div>
                    <h4 className="flow-card-title">Append Context</h4>
                    <p className="flow-card-desc">Update sequence</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card">
                  <span className="flow-card-icon">🔁</span>
                  <div>
                    <h4 className="flow-card-title">Next Step</h4>
                    <p className="flow-card-desc">Autoregressive loop</p>
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="flow-step-card">
                  <span className="flow-card-icon">✍️</span>
                  <div>
                    <h4 className="flow-card-title">User Prompt</h4>
                    <p className="flow-card-desc">Typed text</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card flow-step-card--active">
                  <span className="flow-card-icon">📝</span>
                  <div>
                    <h4 className="flow-card-title">Template Wrapper</h4>
                    <p className="flow-card-desc">ChatML / Context</p>
                  </div>
                </div>
                <span className="flow-step-arrow" aria-hidden="true">→</span>
                <div className="flow-step-card">
                  <span className="flow-card-icon">📄</span>
                  <div>
                    <h4 className="flow-card-title">Formatted Prompt</h4>
                    <p className="flow-card-desc">Ready for tokens</p>
                  </div>
                </div>
              </>
            )
          ) : (
            // Training Flow
            <>
              <div className="flow-step-card">
                <span className="flow-card-icon">📚</span>
                <div>
                  <h4 className="flow-card-title">Corpus</h4>
                  <p className="flow-card-desc">Text data</p>
                </div>
              </div>
              <span className="flow-step-arrow" aria-hidden="true">→</span>
              <div className="flow-step-card flow-step-card--active">
                <span className="flow-card-icon">📉</span>
                <div>
                  <h4 className="flow-card-title">Cross-Entropy</h4>
                  <p className="flow-card-desc">Loss calculation</p>
                </div>
              </div>
              <span className="flow-step-arrow" aria-hidden="true">→</span>
              <div className="flow-step-card">
                <span className="flow-card-icon">⚙️</span>
                <div>
                  <h4 className="flow-card-title">AdamW</h4>
                  <p className="flow-card-desc">Weight update</p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* ============================================================
          WHY THIS MATTERS (Concise Plain-English Learning Card)
          ============================================================ */}
      <div className="why-matters-card">
        <div className="why-matters-header">
          <span className="why-matters-icon" aria-hidden="true">ⓘ</span>
          <h3 className="why-matters-title">Why this matters</h3>
        </div>
        <p className="why-matters-text">
          {mode === "inference" ? (
            stageIndex === 0 ? (
              <>
                <strong>Context conditioning:</strong> Unlike traditional search engines that query an external database, an LLM generates all answers based solely on the activations triggered by its input prompt.
              </>
            ) : stageIndex === 1 ? (
              <>
                <strong>Tokenization</strong> converts human-readable text into a format the model can understand. Tokens can be words, parts of words, or even single characters. This allows the model to work with a fixed vocabulary and handle a wide range of text.
              </>
            ) : stageIndex === 2 ? (
              <>
                <strong>Vector spaces:</strong> Neural networks perform continuous linear algebra. Embeddings place tokens with similar semantic usages near one another, while RoPE injects relative token distance directly into the attention mechanism.
              </>
            ) : stageIndex === 3 ? (
              <>
                <strong>Context mixing:</strong> Tokens in isolation lack meaning. Self-attention dynamically recalculates each token's representation based on the surrounding sentence context, resolving pronouns and word disambiguation.
              </>
            ) : stageIndex === 4 ? (
              <>
                <strong>Logits vs Probabilities:</strong> Logits are raw dot-product scores produced by the linear output layer. They must pass through a normalization function before they can be treated as probabilistic candidate distributions.
              </>
            ) : stageIndex === 5 ? (
              <>
                <strong>Balancing creativity and precision:</strong> Temperature directly controls the shape of the probability distribution. Lower temperatures prevent hallucinations on factual queries, while moderate temperatures enable fluent, expressive storytelling.
              </>
            ) : (
              <>
                <strong>Autoregressive chaining:</strong> An LLM does not write a whole response in one shot. It constructs full paragraphs by recursively feeding its own generated output back into the input sequence.
              </>
            )
          ) : (
            <>
              <strong>Self-Supervised Learning:</strong> LLMs do not need human-annotated answers for pre-training. By shifting text by one token, every sentence automatically provides its own ground-truth target.
            </>
          )}
        </p>
      </div>

      {/* ============================================================
          TECHNICAL DETAILS DISCLOSURE (Collapsible)
          ============================================================ */}
      <div className="technical-disclosure-wrap">
        <button
          type="button"
          className="technical-disclosure-btn"
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          aria-expanded={showTechnicalDetails}
        >
          <span>{showTechnicalDetails ? "▾ Hide technical details" : "▸ Explore technical details"}</span>
        </button>

        {showTechnicalDetails && (
          <div className="technical-details-content">
            <div className="tech-spec-grid">
              <div className="tech-spec-item">
                <span className="tech-spec-label">Tensor Dimension:</span>
                <span className="tech-spec-val text-mono">
                  {stageIndex === 1 ? `[1, ${tokens.length}]` : stageIndex === 2 ? `[1, ${tokens.length}, 768]` : `[1, ${tokens.length}, 32000]`}
                </span>
              </div>
              <div className="tech-spec-item">
                <span className="tech-spec-label">Data Type:</span>
                <span className="tech-spec-val text-mono">FP32 / BF16</span>
              </div>
              <div className="tech-spec-item">
                <span className="tech-spec-label">Core Formula:</span>
                <code className="tech-spec-code text-mono">
                  {stageIndex === 1
                    ? "T(x) = BPE_merge(bytes(x))"
                    : stageIndex === 2
                    ? "v = Embedding[id] * RoPE(m, θ)"
                    : stageIndex === 3
                    ? "softmax(QK^T / √d_k + M) V"
                    : stageIndex === 4
                    ? "z = W_head · h_final"
                    : stageIndex === 5
                    ? "p_i = exp(z_i / T) / Σ exp(z_j / T)"
                    : "x_{t+1} = append(x_t, tok)"}
                </code>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
