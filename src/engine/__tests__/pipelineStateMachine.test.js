import { describe, it, expect } from "vitest";
import {
  RUN_STATES,
  PIPELINE_ACTIONS,
  createInitialPipelineState,
  pipelineReducer,
} from "../pipelineStateMachine";

describe("Pipeline State Machine", () => {
  const dummyStages = [
    { id: "prompt", title: "Prompt" },
    { id: "tokenization", title: "Tokens" },
    { id: "embeddings", title: "Embeddings" },
    { id: "attention", title: "Attention" },
    { id: "logits", title: "Logits" },
    { id: "sampling", title: "Sampling" },
    { id: "generation", title: "Generation" },
  ];

  it("creates clean initial state", () => {
    const state = createInitialPipelineState(dummyStages, "test prompt");
    expect(state.runState).toBe(RUN_STATES.IDLE);
    expect(state.activeStageIndex).toBe(0);
    expect(state.currentRunStageIndex).toBe(0);
    expect(state.totalStages).toBe(7);
    expect(state.history).toEqual([]);
    expect(state.completedStages).toEqual([]);
    expect(state.error).toBeNull();
  });

  it("starts run correctly", () => {
    const initial = createInitialPipelineState(dummyStages);
    const state = pipelineReducer(initial, { type: PIPELINE_ACTIONS.START_RUN });

    expect(state.runState).toBe(RUN_STATES.RUNNING);
    expect(state.activeStageIndex).toBe(0);
    expect(state.history).toEqual([]);
  });

  it("steps forward and saves history snapshot", () => {
    let state = createInitialPipelineState(dummyStages);
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.START_RUN });
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.STEP_FORWARD });

    expect(state.activeStageIndex).toBe(1);
    expect(state.currentRunStageIndex).toBe(1);
    expect(state.history.length).toBe(1);
    expect(state.history[0].activeStageIndex).toBe(0);
    expect(state.completedStages).toContain(0);
  });

  it("steps backward restoring previous history snapshot", () => {
    let state = createInitialPipelineState(dummyStages);
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.START_RUN });
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.STEP_FORWARD });
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.STEP_FORWARD });

    expect(state.activeStageIndex).toBe(2);
    expect(state.history.length).toBe(2);

    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.STEP_BACKWARD });
    expect(state.activeStageIndex).toBe(1);
    expect(state.history.length).toBe(1);

    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.STEP_BACKWARD });
    expect(state.activeStageIndex).toBe(0);
    expect(state.history.length).toBe(0);
  });

  it("completes run when advancing past the last stage", () => {
    let state = createInitialPipelineState(dummyStages);
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.START_RUN });

    // Step forward 6 times to reach the last stage (index 6)
    for (let i = 0; i < 6; i++) {
      state = pipelineReducer(state, { type: PIPELINE_ACTIONS.STEP_FORWARD });
    }
    expect(state.activeStageIndex).toBe(6);
    expect(state.runState).toBe(RUN_STATES.COMPLETED);

    // Stepping forward at end stays completed
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.STEP_FORWARD });
    expect(state.runState).toBe(RUN_STATES.COMPLETED);
    expect(state.activeStageIndex).toBe(6);
  });

  it("pauses and resumes run properly", () => {
    let state = createInitialPipelineState(dummyStages);
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.START_RUN });
    expect(state.runState).toBe(RUN_STATES.RUNNING);

    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.PAUSE_RUN });
    expect(state.runState).toBe(RUN_STATES.PAUSED);

    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.RESUME_RUN });
    expect(state.runState).toBe(RUN_STATES.RUNNING);
  });

  it("resets run cleanly restoring initial state and wiping history", () => {
    let state = createInitialPipelineState(dummyStages);
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.START_RUN });
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.STEP_FORWARD });
    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.STEP_FORWARD });
    expect(state.activeStageIndex).toBe(2);
    expect(state.history.length).toBe(2);

    state = pipelineReducer(state, { type: PIPELINE_ACTIONS.RESET_RUN });
    expect(state.runState).toBe(RUN_STATES.IDLE);
    expect(state.activeStageIndex).toBe(0);
    expect(state.currentRunStageIndex).toBe(0);
    expect(state.history).toEqual([]);
    expect(state.completedStages).toEqual([]);
  });

  it("selects stage without corrupting history", () => {
    let state = createInitialPipelineState(dummyStages);
    state = pipelineReducer(state, {
      type: PIPELINE_ACTIONS.SELECT_STAGE,
      payload: { index: 3 },
    });
    expect(state.activeStageIndex).toBe(3);
  });

  it("updates playback speed correctly", () => {
    let state = createInitialPipelineState(dummyStages);
    state = pipelineReducer(state, {
      type: PIPELINE_ACTIONS.SET_SPEED,
      payload: { speed: 2 },
    });
    expect(state.playbackSpeed).toBe(2);
  });
});
