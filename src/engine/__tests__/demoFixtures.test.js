import { describe, it, expect } from "vitest";
import { getDeterministicDemoData } from "../demoFixtures";

describe("getDeterministicDemoData Context-Aware Completions", () => {
  it("provides Paris completion for Capital of France", () => {
    const data = getDeterministicDemoData("What is the capital of France?");
    expect(data.logits.candidates[0].token).toBe("Paris");
    expect(data.generation.tokens[0].text).toBe("The");
    expect(data.generation.completionText).toContain("Paris");
  });

  it("provides Tokyo completion for Capital of Japan", () => {
    const data = getDeterministicDemoData("What is the capital of Japan?");
    expect(data.logits.candidates[0].token).toBe("Tokyo");
    expect(data.generation.tokens[0].text).toContain("Tokyo");
    expect(data.generation.completionText).toContain("Tokyo");
  });

  it("provides code completion for Python prompts", () => {
    const data = getDeterministicDemoData("Write a Python function that calculates the fibonacci sequence.");
    expect(data.logits.candidates.map((c) => c.token)).toContain("def");
    expect(data.generation.tokens[0].text).toBe("def");
  });

  it("provides Transformer concepts for AI prompts", () => {
    const data = getDeterministicDemoData("Explain how transformer models work in simple terms.");
    expect(data.logits.candidates[0].token).toBe("Transformers");
    expect(data.generation.tokens[0].text).toBe("Transformers");
  });

  it("provides story tokens for narrative prompt", () => {
    const data = getDeterministicDemoData("Once upon a time, in a land far away, there lived a");
    expect(data.logits.candidates[0].token).toBe("young");
    expect(data.generation.tokens[0].text).toBe("young");
  });
});
