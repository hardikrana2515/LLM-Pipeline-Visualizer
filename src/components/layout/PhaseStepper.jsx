import "./PhaseStepper.css";

/**
 * PhaseStepper — Clean editorial left-column component displaying
 * the current prompt summary with Edit action, and the vertical sequential stepper.
 */
export default function PhaseStepper({
  stages,
  activeStageIndex,
  onSelectStage,
  prompt,
  onEditPrompt,
  mode,
}) {
  return (
    <aside className="phase-stepper-sidebar" aria-label="Pipeline Stages Navigation">
      {/* Your Prompt Section */}
      <div className="stepper-prompt-section">
        <span className="stepper-section-label">YOUR PROMPT</span>
        <div className="stepper-prompt-card">
          <p className="stepper-prompt-text">{prompt}</p>
          <div className="stepper-prompt-actions">
            <button
              type="button"
              className="stepper-prompt-edit-btn"
              onClick={onEditPrompt}
              aria-label="Edit input prompt"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
              </svg>
              Edit
            </button>
          </div>
        </div>
      </div>

      {/* Vertical Steps List */}
      <nav className="stepper-steps-nav" aria-label="Stages">
        <ul className="stepper-steps-list">
          {stages.map((stage, idx) => {
            const isActive = idx === activeStageIndex;
            const isCompleted = idx < activeStageIndex;

            return (
              <li key={stage.id} className="stepper-step-item">
                <button
                  type="button"
                  className={`stepper-step-btn ${isActive ? "stepper-step-btn--active" : ""} ${isCompleted ? "stepper-step-btn--completed" : ""}`}
                  onClick={() => onSelectStage(idx)}
                  aria-current={isActive ? "step" : undefined}
                >
                  <span className="stepper-step-badge">
                    {idx + 1}
                  </span>
                  <div className="stepper-step-text">
                    <span className="stepper-step-title">{stage.title}</span>
                    <span className="stepper-step-desc">
                      {stage.shortDescription}
                    </span>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
