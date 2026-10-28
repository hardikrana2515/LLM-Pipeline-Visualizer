/**
 * LLM Pipeline Visualizer — Pipeline State Machine
 *
 * Implements deterministic state transitions for pipeline runs:
 * - Supported states: IDLE, RUNNING, PAUSED, COMPLETED, ERROR, CANCELLED
 * - Pure reducer with immutable state updates
 * - History stack for reliable STEP_BACKWARD state restoration
 * - Clean RESET_RUN behavior that guarantees no stale output
 */

export const RUN_STATES = {
  IDLE: "idle",
  RUNNING: "running",
  PAUSED: "paused",
  COMPLETED: "completed",
  ERROR: "error",
  CANCELLED: "cancelled",
};

export const PIPELINE_ACTIONS = {
  START_RUN: "START_RUN",
  PAUSE_RUN: "PAUSE_RUN",
  RESUME_RUN: "RESUME_RUN",
  STEP_FORWARD: "STEP_FORWARD",
  STEP_BACKWARD: "STEP_BACKWARD",
  RESET_RUN: "RESET_RUN",
  SELECT_STAGE: "SELECT_STAGE",
  COMPLETE_RUN: "COMPLETE_RUN",
  FAIL_RUN: "FAIL_RUN",
  SET_SPEED: "SET_SPEED",
};

/**
 * Creates the initial state for the pipeline state machine.
 * @param {Array} stages - Array of stage definitions
 * @param {string} prompt - Current prompt text
 * @returns {Object} Initial state object
 */
export function createInitialPipelineState(stages = [], prompt = "") {
  return {
    runState: RUN_STATES.IDLE,
    activeStageIndex: 0,
    currentRunStageIndex: 0,
    totalStages: stages.length,
    playbackSpeed: 1,
    prompt: prompt,
    error: null,
    // History stack stores snapshots for reliable STEP_BACKWARD
    history: [],
    // Completed stage indices set
    completedStages: [],
  };
}

/**
 * Pure reducer function for pipeline state transitions.
 * @param {Object} state - Current pipeline state
 * @param {Object} action - Action with type and optional payload
 * @returns {Object} Next pipeline state
 */
export function pipelineReducer(state, action) {
  switch (action.type) {
    case PIPELINE_ACTIONS.START_RUN: {
      return {
        ...state,
        runState: RUN_STATES.RUNNING,
        activeStageIndex: 0,
        currentRunStageIndex: 0,
        completedStages: [],
        history: [],
        error: null,
      };
    }

    case PIPELINE_ACTIONS.PAUSE_RUN: {
      if (state.runState !== RUN_STATES.RUNNING) return state;
      return {
        ...state,
        runState: RUN_STATES.PAUSED,
      };
    }

    case PIPELINE_ACTIONS.RESUME_RUN: {
      if (state.runState !== RUN_STATES.PAUSED) return state;
      return {
        ...state,
        runState: RUN_STATES.RUNNING,
      };
    }

    case PIPELINE_ACTIONS.STEP_FORWARD: {
      if (state.activeStageIndex >= state.totalStages - 1) {
        // Already at last stage, mark completed
        const nextCompleted = Array.from(
          new Set([...state.completedStages, state.activeStageIndex])
        );
        return {
          ...state,
          runState: RUN_STATES.COMPLETED,
          completedStages: nextCompleted,
        };
      }

      // Snapshot current state for history
      const snapshot = {
        activeStageIndex: state.activeStageIndex,
        currentRunStageIndex: state.currentRunStageIndex,
        completedStages: [...state.completedStages],
      };

      const nextIndex = state.activeStageIndex + 1;
      const isLastStage = nextIndex === state.totalStages - 1;
      const nextCompleted = Array.from(
        new Set([...state.completedStages, state.activeStageIndex])
      );

      return {
        ...state,
        activeStageIndex: nextIndex,
        currentRunStageIndex: Math.max(state.currentRunStageIndex, nextIndex),
        completedStages: isLastStage
          ? [...nextCompleted, nextIndex]
          : nextCompleted,
        runState: isLastStage ? RUN_STATES.COMPLETED : state.runState,
        history: [...state.history, snapshot],
      };
    }

    case PIPELINE_ACTIONS.STEP_BACKWARD: {
      if (state.history.length === 0 && state.activeStageIndex === 0) {
        return state;
      }

      // Restore from history stack if available
      if (state.history.length > 0) {
        const newHistory = [...state.history];
        const previousSnapshot = newHistory.pop();

        return {
          ...state,
          activeStageIndex: previousSnapshot.activeStageIndex,
          currentRunStageIndex: previousSnapshot.currentRunStageIndex,
          completedStages: previousSnapshot.completedStages,
          runState:
            state.runState === RUN_STATES.COMPLETED
              ? RUN_STATES.PAUSED
              : state.runState,
          history: newHistory,
        };
      }

      // Fallback: simple decrement
      const prevIndex = Math.max(0, state.activeStageIndex - 1);
      return {
        ...state,
        activeStageIndex: prevIndex,
        runState:
          state.runState === RUN_STATES.COMPLETED
            ? RUN_STATES.PAUSED
            : state.runState,
      };
    }

    case PIPELINE_ACTIONS.SELECT_STAGE: {
      const targetIndex = action.payload?.index ?? 0;
      if (targetIndex < 0 || targetIndex >= state.totalStages) return state;

      return {
        ...state,
        activeStageIndex: targetIndex,
        // Only advance currentRunStageIndex if selecting ahead during active run
        currentRunStageIndex: Math.max(state.currentRunStageIndex, targetIndex),
      };
    }

    case PIPELINE_ACTIONS.RESET_RUN: {
      return {
        ...state,
        runState: RUN_STATES.IDLE,
        activeStageIndex: 0,
        currentRunStageIndex: 0,
        completedStages: [],
        history: [],
        error: null,
      };
    }

    case PIPELINE_ACTIONS.COMPLETE_RUN: {
      return {
        ...state,
        runState: RUN_STATES.COMPLETED,
        completedStages: Array.from(
          { length: state.totalStages },
          (_, i) => i
        ),
      };
    }

    case PIPELINE_ACTIONS.FAIL_RUN: {
      return {
        ...state,
        runState: RUN_STATES.ERROR,
        error: action.payload?.error || "Pipeline execution encountered an error.",
      };
    }

    case PIPELINE_ACTIONS.SET_SPEED: {
      return {
        ...state,
        playbackSpeed: action.payload?.speed || 1,
      };
    }

    default:
      return state;
  }
}
