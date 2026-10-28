import "./PhaseBottomNav.css";

/**
 * PhaseBottomNav — Clean, persistent bottom navigation with Previous,
 * Center Step Progress, and Next actions.
 */
export default function PhaseBottomNav({
  currentStageIndex,
  totalStages,
  onPrevious,
  onNext,
  canPrevious,
  canNext,
}) {
  const currentStep = currentStageIndex + 1;
  const progressPercent = Math.round((currentStep / totalStages) * 100);

  return (
    <footer className="phase-bottom-nav" role="navigation" aria-label="Phase navigation">
      {/* Left: Previous Button */}
      <button
        type="button"
        className="nav-btn nav-btn--prev"
        onClick={onPrevious}
        disabled={!canPrevious}
        aria-label="Go to previous phase"
      >
        <span aria-hidden="true">←</span> Previous
      </button>

      {/* Center: Step Progress Indicator */}
      <div className="nav-step-indicator" aria-live="polite">
        <span className="nav-step-counter">
          {currentStep} of {totalStages}
        </span>
        <div
          className="nav-step-progress-track"
          role="progressbar"
          aria-valuenow={currentStep}
          aria-valuemin={1}
          aria-valuemax={totalStages}
        >
          <div
            className="nav-step-progress-fill"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Right: Next Button */}
      <button
        type="button"
        className="nav-btn nav-btn--next"
        onClick={onNext}
        disabled={!canNext}
        aria-label="Proceed to next phase"
      >
        Next <span aria-hidden="true">→</span>
      </button>
    </footer>
  );
}
