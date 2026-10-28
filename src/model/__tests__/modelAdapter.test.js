import { describe, it, expect, vi } from "vitest";
import {
  SimulationModelAdapter,
  modelService,
} from "../modelAdapter";
import { SUPPORTED_MODELS, detectRuntimeCapabilities } from "../modelConfig";

describe("Model Adapter Architecture (Phase 5)", () => {
  it("registers supported models with complete metadata", () => {
    expect(SUPPORTED_MODELS.length).toBeGreaterThanOrEqual(3);
    const demo = SUPPORTED_MODELS.find((m) => m.id === "demo-mode");
    expect(demo).toBeDefined();
    expect(demo.capabilities.provenance).toBe("illustrative");
    expect(demo.downloadSize).toBe("0 MB");

    const qwen = SUPPORTED_MODELS.find((m) => m.id === "qwen-0.5b-chat");
    expect(qwen).toBeDefined();
    expect(qwen.license).toBe("Apache 2.0");
    expect(qwen.capabilities.logits).toBe(false); // capability gated
  });

  it("detectRuntimeCapabilities returns valid hardware profile", async () => {
    const caps = await detectRuntimeCapabilities();
    expect(typeof caps.hasWasm).toBe("boolean");
    expect(typeof caps.hasWebGPU).toBe("boolean");
    expect(caps.cpuCores).toBeGreaterThanOrEqual(1);
  });

  it("SimulationModelAdapter streams tokens asynchronously", async () => {
    const adapter = new SimulationModelAdapter();
    const tokenEvents = [];

    const result = await adapter.generate(
      "What is the capital of France?",
      {},
      (tok) => tokenEvents.push(tok)
    );

    expect(tokenEvents.length).toBeGreaterThan(0);
    expect(result.output).toContain("Paris");
    expect(result.elapsedMs).toBeGreaterThan(0);
    expect(result.provenance).toBe("illustrative");
  });

  it("SimulationModelAdapter supports cancellation", async () => {
    const adapter = new SimulationModelAdapter();
    const genPromise = adapter.generate("Test prompt", {}, () => {
      adapter.cancel();
    });

    await expect(genPromise).rejects.toThrow(/cancelled/i);
  });

  it("ModelService notifies subscribers on model selection change", async () => {
    const listener = vi.fn();
    const unsubscribe = modelService.subscribe(listener);

    await modelService.selectModel("gpt2");
    expect(listener).toHaveBeenCalled();
    const lastCallArg = listener.mock.calls[listener.mock.calls.length - 1][0];
    expect(lastCallArg.currentModelId).toBe("gpt2");
    expect(lastCallArg.status).toBe("idle");

    // Revert to demo mode
    await modelService.selectModel("demo-mode");
    unsubscribe();
  });
});
