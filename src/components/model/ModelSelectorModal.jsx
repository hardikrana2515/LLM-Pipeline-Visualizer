import { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import Badge from "../common/Badge";
import { SUPPORTED_MODELS, detectRuntimeCapabilities } from "../../model/modelConfig";
import { modelService } from "../../model/modelAdapter";
import "./ModelSelectorModal.css";

/**
 * ModelSelectorModal — lets the user inspect hardware runtime capabilities,
 * select between Demo Simulation and Real In-Browser / Local models, and explicitly
 * trigger model downloads with progress feedback and capability gating disclosures.
 */
export default function ModelSelectorModal({ isOpen, onClose }) {
  const [modelState, setModelState] = useState(() => ({
    currentModelId: modelService.currentModelId,
    currentModel: modelService.getModelConfig(),
    status: modelService.status,
    statusMessage: modelService.statusMessage,
    downloadProgress: modelService.downloadProgress,
    capabilities: modelService.adapter.getCapabilities(),
  }));

  const [hwCapabilities, setHwCapabilities] = useState(null);

  // Subscribe to ModelService state changes
  useEffect(() => {
    const unsubscribe = modelService.subscribe((state) => {
      setModelState(state);
    });
    return unsubscribe;
  }, []);

  // Detect hardware on mount
  useEffect(() => {
    if (isOpen) {
      detectRuntimeCapabilities().then(setHwCapabilities);
    }
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const { currentModelId, status, statusMessage, downloadProgress } = modelState;

  const handleSelectModel = (id) => {
    modelService.selectModel(id);
  };

  const handleLoadModel = () => {
    modelService.loadActiveModel();
  };

  const handleCancel = () => {
    modelService.cancel();
  };

  return createPortal(
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Model Selection & Runtime"
      onClick={onClose}
    >
      <div className="model-modal" onClick={(e) => e.stopPropagation()}>
        <div className="model-modal__header">
          <div className="model-modal__title-row">
            <span className="model-modal__icon">⚙️</span>
            <div>
              <h2 className="model-modal__title">Model Runtime & Inference Settings</h2>
              <p className="model-modal__subtitle">
                Choose between instant educational simulation and real in-browser neural network inference.
              </p>
            </div>
          </div>
          <button
            type="button"
            className="model-modal__close-btn"
            onClick={onClose}
            aria-label="Close modal"
          >
            ✕
          </button>
        </div>

        <div className="model-modal__body">
          {/* Hardware Capabilities Profile */}
          <div className="hw-profile-card">
            <span className="hw-profile-title">Client Hardware & Runtime Profile</span>
            <div className="hw-profile-grid">
              <div className="hw-item">
                <span className="hw-label">WebGPU Acceleration:</span>
                <span
                  className={`hw-val ${
                    hwCapabilities?.hasWebGPU ? "hw-val--supported" : "hw-val--unsupported"
                  }`}
                >
                  {hwCapabilities?.hasWebGPU ? "● Supported" : "○ Not detected (CPU Fallback)"}
                </span>
              </div>
              <div className="hw-item">
                <span className="hw-label">WebAssembly (WASM):</span>
                <span className="hw-val hw-val--supported">
                  {hwCapabilities?.hasWasm ? "● Available" : "○ Not available"}
                </span>
              </div>
              <div className="hw-item">
                <span className="hw-label">CPU Concurrency:</span>
                <span className="hw-val text-mono">{hwCapabilities?.cpuCores || 4} Cores</span>
              </div>
            </div>
          </div>

          {/* Model Choices List */}
          <div className="models-list">
            {SUPPORTED_MODELS.map((model) => {
              const isSelected = currentModelId === model.id;
              const isReady = isSelected && status === "ready";
              const isLoading = isSelected && (status === "loading" || status === "downloading");

              return (
                <div
                  key={model.id}
                  className={`model-card ${isSelected ? "model-card--selected" : ""}`}
                >
                  <div className="model-card__top">
                    <div className="model-card__info">
                      <div className="model-card__name-row">
                        <span className="model-card__name">{model.name}</span>
                        <Badge
                          type="provenance"
                          variant={model.capabilities.provenance}
                          label={model.capabilities.provenance.toUpperCase()}
                          size="sm"
                        />
                      </div>
                      <span className="model-card__source">
                        {model.source} • License: <strong>{model.license}</strong> • Size: <strong>{model.downloadSize}</strong>
                      </span>
                    </div>

                    <div className="model-card__actions">
                      {isSelected ? (
                        <div className="model-card__status-wrap">
                          {isReady ? (
                            <span className="status-pill status-pill--ready">● Ready</span>
                          ) : isLoading ? (
                            <button
                              type="button"
                              className="btn-cancel"
                              onClick={handleCancel}
                            >
                              Cancel
                            </button>
                          ) : (
                            <button
                              type="button"
                              className="btn-load-model"
                              onClick={handleLoadModel}
                            >
                              Connect / Load
                            </button>
                          )}
                        </div>
                      ) : (
                        <button
                          type="button"
                          className="btn-select-model"
                          onClick={() => handleSelectModel(model.id)}
                        >
                          Select Model
                        </button>
                      )}
                    </div>
                  </div>

                  <p className="model-card__desc">{model.description}</p>

                  {/* Loading / Download Progress */}
                  {isSelected && isLoading && (
                    <div className="model-progress-wrap">
                      <div className="model-progress-header">
                        <span className="model-progress-status">{statusMessage}</span>
                        <span className="model-progress-percent text-mono">{downloadProgress}%</span>
                      </div>
                      <div className="model-progress-bar-bg">
                        <div
                          className="model-progress-bar-fill"
                          style={{ width: `${downloadProgress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Capability Gating Disclosures */}
                  <div className="model-caps-row">
                    <span className="caps-label">Verified Outputs:</span>
                    <span className="cap-tag cap-tag--ok">✓ Generated Tokens</span>
                    <span className="cap-tag cap-tag--ok">✓ Token IDs</span>
                    <span className="cap-tag cap-tag--ok">✓ Latency & Throughput</span>
                    {!model.capabilities.attention && (
                      <span className="cap-tag cap-tag--gated" title="Not returned by standard in-browser ONNX compute graph">
                        ⊘ Attention Weights (Unavailable)
                      </span>
                    )}
                    {!model.capabilities.logits && (
                      <span className="cap-tag cap-tag--gated" title="Full vocabulary logits tensor not exported in browser ONNX pipeline">
                        ⊘ Vocab Logits (Gated)
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="model-modal__footer">
          <div className="model-status-indicator">
            <span className="status-label">Active Status:</span>
            <span className="status-val text-mono">{statusMessage}</span>
          </div>
          <button type="button" className="btn-close-modal" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
