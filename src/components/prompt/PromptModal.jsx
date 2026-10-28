import { useState, useEffect } from "react";
import { SAMPLE_PROMPTS } from "../../data/constants";
import "./PromptModal.css";

/**
 * PromptModal — Clean setup dialog for entering/editing prompts and selecting examples.
 */
export default function PromptModal({
  isOpen,
  onClose,
  initialPrompt,
  onSavePrompt,
  mode,
  onModeChange,
}) {
  const [text, setText] = useState(initialPrompt);

  useEffect(() => {
    setText(initialPrompt);
  }, [initialPrompt, isOpen]);

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      onSavePrompt(text.trim());
      onClose();
    }
  };

  const handleSelectSample = (sampleText) => {
    setText(sampleText);
  };

  return (
    <div
      className="prompt-modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="prompt-modal-title"
      onClick={onClose}
    >
      <div
        className="prompt-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="prompt-modal__header">
          <div>
            <h2 id="prompt-modal-title" className="prompt-modal__title">
              Set Your Input Prompt
            </h2>
            <p className="prompt-modal__subtitle">
              Every LLM generation starts with input context. Choose an educational sample or write your own.
            </p>
          </div>
          <button
            type="button"
            className="prompt-modal__close-btn"
            onClick={onClose}
            aria-label="Close dialog"
          >
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="prompt-modal__body">
          {/* Sample Prompts */}
          <div className="prompt-modal__samples">
            <span className="prompt-modal__samples-label">Curated examples:</span>
            <div className="prompt-modal__sample-chips">
              {SAMPLE_PROMPTS.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  className={`prompt-sample-chip ${text === sample.text ? "prompt-sample-chip--active" : ""}`}
                  onClick={() => handleSelectSample(sample.text)}
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>

          {/* Text Area */}
          <div className="prompt-modal__input-wrap">
            <textarea
              className="prompt-modal__textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={4}
              placeholder="Type your prompt here..."
              autoFocus
            />
          </div>

          {/* Footer Actions */}
          <div className="prompt-modal__footer">
            <button
              type="button"
              className="prompt-modal__cancel-btn"
              onClick={onClose}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="prompt-modal__submit-btn"
              disabled={!text.trim()}
            >
              Explore pipeline →
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
