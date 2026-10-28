import { useState, useMemo } from "react";
import { tokenizePrompt } from "../../tokenizer/subwordTokenizer";
import "./PhaseInspector.css";

/**
 * PhaseInspector — Right-column contextual educational inspector.
 * Matches the reference design for Token Inspector and adapts cleanly to each phase.
 */
export default function PhaseInspector({
  stage,
  stageIndex,
  prompt,
  selectedTokenIndex,
  onSelectToken,
  mode,
}) {
  const [allTokensOpen, setAllTokensOpen] = useState(true);

  // Tokenize the prompt
  const tokens = useMemo(() => {
    return tokenizePrompt(prompt || "What is the capital of France?");
  }, [prompt]);

  const activeToken = tokens[selectedTokenIndex] || tokens[0] || {
    text: "France",
    id: 1672,
    cleanText: "France",
    type: "regular",
    startChar: 16,
    endChar: 22,
  };

  return (
    <aside className="phase-inspector-rail" aria-label="Phase Contextual Inspector">
      {/* ── PHASE 2: TOKEN INSPECTOR (Exact match to reference design) ── */}
      {mode === "inference" && stageIndex === 1 && (
        <>
          {/* Main Token Inspector Card */}
          <div className="inspector-card">
            <div className="inspector-card__header">
              <span className="inspector-card__icon">ⓘ</span>
              <h3 className="inspector-card__title">Token Inspector</h3>
            </div>

            <div className="inspector-card__body">
              {/* Selected Token Headline Chip */}
              <div className="token-preview-banner">
                <span className="token-preview-chip">{activeToken.text}</span>
                <span className="token-type-pill">Regular token</span>
              </div>

              {/* Token Spec Rows */}
              <div className="token-spec-table">
                <div className="spec-row">
                  <span className="spec-label">Token text</span>
                  <span className="spec-value text-mono">{activeToken.text}</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Token ID</span>
                  <span className="spec-value text-mono">{activeToken.id}</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Position</span>
                  <span className="spec-value text-mono">{selectedTokenIndex}</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Type</span>
                  <span className="spec-value">Regular token</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Offset (start-end)</span>
                  <span className="spec-value text-mono">
                    {activeToken.startChar ?? 0} - {activeToken.endChar ?? activeToken.text.length}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* All Tokens Expandable Table */}
          <div className="inspector-card">
            <button
              type="button"
              className="inspector-card__header inspector-card__header--collapsible"
              onClick={() => setAllTokensOpen(!allTokensOpen)}
              aria-expanded={allTokensOpen}
            >
              <div className="header-left">
                <span className="inspector-card__icon">⚅</span>
                <h3 className="inspector-card__title">All Tokens</h3>
              </div>
              <span className="chevron-toggle">{allTokensOpen ? "⌃" : "⌄"}</span>
            </button>

            {allTokensOpen && (
              <div className="inspector-card__body inspector-card__body--table">
                <table className="all-tokens-table">
                  <thead>
                    <tr>
                      <th>Token</th>
                      <th>ID</th>
                      <th>Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tokens.map((tok, idx) => {
                      const isSelected = idx === selectedTokenIndex;
                      return (
                        <tr
                          key={idx}
                          className={`token-row ${isSelected ? "token-row--selected" : ""}`}
                          onClick={() => onSelectToken(idx)}
                        >
                          <td className="token-name-td">
                            <span className={`mini-token-chip ${isSelected ? "mini-token-chip--selected" : ""}`}>
                              {tok.text}
                            </span>
                          </td>
                          <td className="token-id-td text-mono">{tok.id}</td>
                          <td className="token-type-td">
                            <span className="table-type-pill">Regular</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Educational Note Card */}
          <div className="inspector-note-card">
            <div className="note-card-header">
              <span className="note-icon">ⓘ</span>
              <h4 className="note-title">Note</h4>
            </div>
            <p className="note-text">
              Token boundaries don&apos;t always match words. The exact tokens and IDs depend on the model&apos;s tokenizer.
            </p>
          </div>
        </>
      )}

      {/* ── PHASE 1: PROMPT INSPECTOR ── */}
      {mode === "inference" && stageIndex === 0 && (
        <>
          <div className="inspector-card">
            <div className="inspector-card__header">
              <span className="inspector-card__icon">ⓘ</span>
              <h3 className="inspector-card__title">Prompt Metadata</h3>
            </div>
            <div className="inspector-card__body">
              <div className="token-spec-table">
                <div className="spec-row">
                  <span className="spec-label">Encoding</span>
                  <span className="spec-value text-mono">UTF-8</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Character count</span>
                  <span className="spec-value text-mono">{prompt.length}</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Token count</span>
                  <span className="spec-value text-mono">{tokens.length}</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Format</span>
                  <span className="spec-value">Plain text / User prompt</span>
                </div>
              </div>
            </div>
          </div>

          <div className="inspector-note-card">
            <div className="note-card-header">
              <span className="note-icon">ⓘ</span>
              <h4 className="note-title">Context Window</h4>
            </div>
            <p className="note-text">
              Every token in the prompt consumes context window capacity. Modern models support between 2,048 and 128,000+ context tokens.
            </p>
          </div>
        </>
      )}

      {/* ── PHASE 3: EMBEDDINGS INSPECTOR ── */}
      {mode === "inference" && stageIndex === 2 && (
        <>
          <div className="inspector-card">
            <div className="inspector-card__header">
              <span className="inspector-card__icon">📐</span>
              <h3 className="inspector-card__title">Embedding Specs</h3>
            </div>
            <div className="inspector-card__body">
              <div className="token-spec-table">
                <div className="spec-row">
                  <span className="spec-label">Selected token</span>
                  <span className="spec-value text-mono">{activeToken.text}</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Embedding dimension</span>
                  <span className="spec-value text-mono">768 (d_model)</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Vector L2 Norm</span>
                  <span className="spec-value text-mono">27.71</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Positional encoding</span>
                  <span className="spec-value">RoPE (Rotary)</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Frequency base (θ)</span>
                  <span className="spec-value text-mono">10,000</span>
                </div>
              </div>
            </div>
          </div>

          <div className="inspector-note-card">
            <div className="note-card-header">
              <span className="note-icon">ⓘ</span>
              <h4 className="note-title">Semantic Proximity</h4>
            </div>
            <p className="note-text">
              Embedding vectors position words in semantic space such that geometric angle (cosine similarity) reflects contextual relationship.
            </p>
          </div>
        </>
      )}

      {/* ── PHASE 4: ATTENTION INSPECTOR ── */}
      {mode === "inference" && stageIndex === 3 && (
        <>
          <div className="inspector-card">
            <div className="inspector-card__header">
              <span className="inspector-card__icon">🧠</span>
              <h3 className="inspector-card__title">Attention Structure</h3>
            </div>
            <div className="inspector-card__body">
              <div className="token-spec-table">
                <div className="spec-row">
                  <span className="spec-label">Attention heads</span>
                  <span className="spec-value text-mono">8 heads</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Head dimension (d_k)</span>
                  <span className="spec-value text-mono">64</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Scaling factor</span>
                  <span className="spec-value text-mono">1 / √64 = 0.125</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Masking</span>
                  <span className="spec-value">Causal (lower-triangular)</span>
                </div>
              </div>
            </div>
          </div>

          <div className="inspector-note-card">
            <div className="note-card-header">
              <span className="note-icon">ⓘ</span>
              <h4 className="note-title">Multi-Head Diversity</h4>
            </div>
            <p className="note-text">
              Different attention heads specialize in distinct linguistic features: some track syntactic dependencies, while others track coreference or positional adjacency.
            </p>
          </div>
        </>
      )}

      {/* ── PHASE 5: LOGITS INSPECTOR ── */}
      {mode === "inference" && stageIndex === 4 && (
        <>
          <div className="inspector-card">
            <div className="inspector-card__header">
              <span className="inspector-card__icon">📊</span>
              <h3 className="inspector-card__title">Logits Projection</h3>
            </div>
            <div className="inspector-card__body">
              <div className="token-spec-table">
                <div className="spec-row">
                  <span className="spec-label">Vocabulary size (V)</span>
                  <span className="spec-value text-mono">32,000</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">LM Head weight shape</span>
                  <span className="spec-value text-mono">[32000, 768]</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Max logit value</span>
                  <span className="spec-value text-mono">14.2 (&quot;Paris&quot;)</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Logit range</span>
                  <span className="spec-value text-mono">[-18.5, +14.2]</span>
                </div>
              </div>
            </div>
          </div>

          <div className="inspector-note-card">
            <div className="note-card-header">
              <span className="note-icon">ⓘ</span>
              <h4 className="note-title">Unbounded Scores</h4>
            </div>
            <p className="note-text">
              Logits are real-valued outputs from matrix multiplication. They cannot be summed or treated as odds until softmax normalization is applied.
            </p>
          </div>
        </>
      )}

      {/* ── PHASE 6: SAMPLING INSPECTOR ── */}
      {mode === "inference" && stageIndex === 5 && (
        <>
          <div className="inspector-card">
            <div className="inspector-card__header">
              <span className="inspector-card__icon">🎲</span>
              <h3 className="inspector-card__title">Sampling Dynamics</h3>
            </div>
            <div className="inspector-card__body">
              <div className="token-spec-table">
                <div className="spec-row">
                  <span className="spec-label">Decoding method</span>
                  <span className="spec-value">Nucleus (Top-P)</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Argmax (Greedy) token</span>
                  <span className="spec-value text-mono">&quot;Paris&quot;</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Greedy probability</span>
                  <span className="spec-value text-mono">85.4%</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">PRNG source</span>
                  <span className="spec-value">LCG Categorical</span>
                </div>
              </div>
            </div>
          </div>

          <div className="inspector-note-card">
            <div className="note-card-header">
              <span className="note-icon">ⓘ</span>
              <h4 className="note-title">Temperature 0 vs &gt;0</h4>
            </div>
            <p className="note-text">
              At T=0 (greedy), the model always picks the argmax token deterministically. At T&gt;0, tokens are drawn probabilistically from the filtered distribution.
            </p>
          </div>
        </>
      )}

      {/* ── PHASE 7: GENERATION INSPECTOR ── */}
      {mode === "inference" && stageIndex === 6 && (
        <>
          <div className="inspector-card">
            <div className="inspector-card__header">
              <span className="inspector-card__icon">✨</span>
              <h3 className="inspector-card__title">Autoregressive Loop</h3>
            </div>
            <div className="inspector-card__body">
              <div className="token-spec-table">
                <div className="spec-row">
                  <span className="spec-label">Total context tokens</span>
                  <span className="spec-value text-mono">{tokens.length + 1}</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">KV Cache status</span>
                  <span className="spec-value text-mono">Cached (O(1) step)</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Stop condition</span>
                  <span className="spec-value">&lt;|endoftext|&gt; / EOS</span>
                </div>
                <div className="spec-row">
                  <span className="spec-label">Step latency</span>
                  <span className="spec-value text-mono">~35 ms / tok</span>
                </div>
              </div>
            </div>
          </div>

          <div className="inspector-note-card">
            <div className="note-card-header">
              <span className="note-icon">ⓘ</span>
              <h4 className="note-title">KV Caching</h4>
            </div>
            <p className="note-text">
              Without KV caching, each step would recompute attention for all prior tokens ($O(N^2)$). Caching Keys and Values makes subsequent token generation $O(N)$.
            </p>
          </div>
        </>
      )}

      {/* ── TRAINING MODE INSPECTOR ── */}
      {mode === "training" && (
        <div className="inspector-card">
          <div className="inspector-card__header">
            <span className="inspector-card__icon">🧪</span>
            <h3 className="inspector-card__title">Training Diagnostics</h3>
          </div>
          <div className="inspector-card__body">
            <div className="token-spec-table">
              <div className="spec-row">
                <span className="spec-label">Training stage</span>
                <span className="spec-value">{stage.title}</span>
              </div>
              <div className="spec-row">
                <span className="spec-label">Objective</span>
                <span className="spec-value">Next-token prediction</span>
              </div>
              <div className="spec-row">
                <span className="spec-label">Loss function</span>
                <span className="spec-value text-mono">Cross-Entropy (-ln P)</span>
              </div>
              <div className="spec-row">
                <span className="spec-label">Optimizer</span>
                <span className="spec-value text-mono">AdamW (decay=0.01)</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
