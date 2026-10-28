import { useState, useEffect } from "react";
import ModelSelectorModal from "../model/ModelSelectorModal";
import { modelService } from "../../model/modelAdapter";
import "./TopBar.css";

/**
 * Top application bar — Futuristic AI / GPT header with glowing neural icon,
 * live model status beacon, cyber mode pill toggle, and settings gear.
 */
export default function TopBar({ mode, onModeChange }) {
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [modelInfo, setModelInfo] = useState(() => ({
    name: modelService.getModelConfig().name,
    status: modelService.status,
    isDemo: modelService.currentModelId === "demo-mode",
  }));

  useEffect(() => {
    const unsubscribe = modelService.subscribe((state) => {
      setModelInfo({
        name: state.currentModel.name,
        status: state.status,
        isDemo: state.currentModelId === "demo-mode",
      });
    });
    return unsubscribe;
  }, []);

  return (
    <header className="topbar" role="banner">
      <div className="topbar__left">
        <div className="topbar__brand">
          <div className="topbar__icon-halo">
            {/* High-Tech Neural Sparkle / Node SVG */}
            <svg
              className="topbar__logo-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
            </svg>
          </div>
          <div className="topbar__title-group">
            <div className="topbar__title-row">
              <h1 className="topbar__title text-display">LLM Pipeline Visualizer</h1>
              <span className="topbar__tag-beacon">
                <span className="beacon-dot" />
                GPT-4x Engine
              </span>
            </div>
            <p className="topbar__subtitle">Interactive Transformer & Autoregressive Neural Network Architecture</p>
          </div>
        </div>
      </div>

      <div className="topbar__right">
        {/* Cyber Mode Switch Pill */}
        <div
          className="mode-pill-toggle"
          role="radiogroup"
          aria-label="Pipeline mode selection"
        >
          <button
            type="button"
            className={`mode-pill-btn ${mode === "inference" ? "mode-pill-btn--active" : ""}`}
            role="radio"
            aria-checked={mode === "inference"}
            onClick={() => onModeChange("inference")}
          >
            <span className="mode-btn-icon">⚡</span>
            Inference
          </button>
          <button
            type="button"
            className={`mode-pill-btn ${mode === "training" ? "mode-pill-btn--active" : ""}`}
            role="radio"
            aria-checked={mode === "training"}
            onClick={() => onModeChange("training")}
          >
            <span className="mode-btn-icon">🧬</span>
            Training
          </button>
        </div>

        {/* Cyber Settings Button */}
        <button
          type="button"
          className="topbar__settings-btn"
          onClick={() => setShowSettingsModal(true)}
          aria-label="Runtime settings and models"
          title={`Runtime: ${modelInfo.isDemo ? "Demo Simulation" : modelInfo.name}`}
        >
          <svg
            className="settings-gear-svg"
            width="17"
            height="17"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>

      {/* Model Selector Modal */}
      <ModelSelectorModal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
      />
    </header>
  );
}
