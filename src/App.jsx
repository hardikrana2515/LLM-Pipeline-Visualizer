import { useState, useCallback, useReducer, useMemo, useEffect } from "react";
import TopBar from "./components/layout/TopBar";
import PhaseStepper from "./components/layout/PhaseStepper";
import PhaseBottomNav from "./components/layout/PhaseBottomNav";
import PhaseWorkspace from "./components/workspace/PhaseWorkspace";
import PhaseInspector from "./components/workspace/PhaseInspector";
import PromptModal from "./components/prompt/PromptModal";
import {
  INFERENCE_STAGES,
  TRAINING_STAGES,
  SAMPLE_PROMPTS,
} from "./data/constants";
import {
  PIPELINE_ACTIONS,
  createInitialPipelineState,
  pipelineReducer,
} from "./engine/pipelineStateMachine";
import { getDeterministicDemoData } from "./engine/demoFixtures";
import "./App.css";

/**
 * App — Minimal, Phase-by-Phase Learning Experience for LLM Pipeline Visualizer.
 * Clean 3-column layout matching editorial scientific learning principles:
 * - Left: Prompt summary + Vertical 7-stage sequential stepper
 * - Center: Single focused phase workspace (Title, intro, visual demo, flow, why it matters, technical disclosure)
 * - Right: Contextual Inspector with live token/phase inspection and educational notes
 * - Bottom: Persistent Previous / Next navigation bar
 */
export default function App() {
  const [mode, setMode] = useState("inference");
  const [prompt, setPrompt] = useState(SAMPLE_PROMPTS[0].text); // "What is the capital of France?"
  const [showPromptModal, setShowPromptModal] = useState(false);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(5); // Default to 'France' matching reference design

  // Current stages according to active mode
  const stages = useMemo(
    () => (mode === "inference" ? INFERENCE_STAGES : TRAINING_STAGES),
    [mode]
  );

  // Pipeline State Machine via useReducer (start at index 1: Tokenization to match reference screenshot)
  const [pipelineState, dispatch] = useReducer(
    pipelineReducer,
    null,
    () => {
      const initial = createInitialPipelineState(INFERENCE_STAGES, SAMPLE_PROMPTS[0].text);
      return {
        ...initial,
        activeStageIndex: 1, // Start on Step 2 of 7 (Tokenization)
        currentRunStageIndex: 1,
      };
    }
  );

  const { activeStageIndex } = pipelineState;

  // Active Stage Object
  const activeStage = stages[activeStageIndex] || stages[0];

  // Deterministic demo fixtures for current prompt
  const demoData = useMemo(() => {
    return getDeterministicDemoData(prompt);
  }, [prompt]);

  // Mode change handler (Inference / Training)
  const handleModeChange = useCallback((newMode) => {
    setMode(newMode);
    dispatch({ type: PIPELINE_ACTIONS.RESET_RUN });
    setSelectedTokenIndex(0);
  }, []);

  // Step navigation handlers
  const handlePrevious = useCallback(() => {
    if (activeStageIndex > 0) {
      dispatch({ type: PIPELINE_ACTIONS.STEP_BACKWARD });
    }
  }, [activeStageIndex]);

  const handleNext = useCallback(() => {
    if (activeStageIndex < stages.length - 1) {
      dispatch({ type: PIPELINE_ACTIONS.STEP_FORWARD });
    }
  }, [activeStageIndex, stages.length]);

  const handleSelectStage = useCallback((idx) => {
    dispatch({
      type: PIPELINE_ACTIONS.SELECT_STAGE,
      payload: { index: idx },
    });
  }, []);

  const handleSavePrompt = useCallback((newPrompt) => {
    setPrompt(newPrompt);
    setSelectedTokenIndex(0);
  }, []);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      const activeEl = document.activeElement;
      const isInputActive =
        activeEl && (activeEl.tagName === "INPUT" || activeEl.tagName === "TEXTAREA");
      if (isInputActive) return;

      if (e.code === "ArrowRight") {
        e.preventDefault();
        handleNext();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        handlePrevious();
      } else if (e.code === "Space") {
        e.preventDefault();
        handleNext();
      } else if (e.key === "i" || e.key === "I") {
        e.preventDefault();
        handleModeChange("inference");
      } else if (e.key === "t" || e.key === "T") {
        e.preventDefault();
        handleModeChange("training");
      } else if (e.key >= "1" && e.key <= "7") {
        const stageIdx = parseInt(e.key, 10) - 1;
        if (stageIdx < stages.length) {
          e.preventDefault();
          handleSelectStage(stageIdx);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleNext, handlePrevious, handleModeChange, handleSelectStage, stages.length]);

  return (
    <div className="app-container">
      {/* Top Application Bar */}
      <TopBar mode={mode} onModeChange={handleModeChange} />

      {/* Main Learning Body: 3 Distinct Columns */}
      <div className="app-main-layout">
        {/* Left Column: Prompt Summary & Vertical Sequential Stepper */}
        <PhaseStepper
          stages={stages}
          activeStageIndex={activeStageIndex}
          onSelectStage={handleSelectStage}
          prompt={prompt}
          onEditPrompt={() => setShowPromptModal(true)}
          mode={mode}
        />

        {/* Center Column: The Primary Learning Workspace */}
        <div className="app-workspace-column">
          <PhaseWorkspace
            stage={activeStage}
            stageIndex={activeStageIndex}
            totalStages={stages.length}
            prompt={prompt}
            demoData={demoData}
            mode={mode}
            selectedTokenIndex={selectedTokenIndex}
            onSelectToken={setSelectedTokenIndex}
            onEditPrompt={() => setShowPromptModal(true)}
          />
        </div>

        {/* Right Column: Contextual Inspector (Live Token / Phase Details) */}
        <PhaseInspector
          stage={activeStage}
          stageIndex={activeStageIndex}
          prompt={prompt}
          selectedTokenIndex={selectedTokenIndex}
          onSelectToken={setSelectedTokenIndex}
          mode={mode}
        />
      </div>

      {/* Bottom Persistent Navigation Bar */}
      <PhaseBottomNav
        currentStageIndex={activeStageIndex}
        totalStages={stages.length}
        onPrevious={handlePrevious}
        onNext={handleNext}
        canPrevious={activeStageIndex > 0}
        canNext={activeStageIndex < stages.length - 1}
      />

      {/* Clean Prompt Setup / Edit Dialog */}
      <PromptModal
        isOpen={showPromptModal}
        onClose={() => setShowPromptModal(false)}
        initialPrompt={prompt}
        onSavePrompt={handleSavePrompt}
        mode={mode}
        onModeChange={handleModeChange}
      />
    </div>
  );
}
