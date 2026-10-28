import { describe, it, expect } from "vitest";
import { SubwordTokenizer } from "../subwordTokenizer";

describe("Subword BPE Tokenizer Engine", () => {
  const tokenizer = new SubwordTokenizer();

  it("tokenizes standard sentence with correct token count and IDs", () => {
    const res = tokenizer.tokenize("What is the capital of France?");
    expect(res.tokens.length).toBe(7);
    expect(res.tokens[0].text).toBe("What");
    expect(res.tokens[0].id).toBe(2061);
    expect(res.tokens[1].text).toBe(" is");
    expect(res.tokens[1].id).toBe(318);
    expect(res.tokens[5].text).toBe(" France");
    expect(res.tokens[5].id).toBe(4881);
    expect(res.tokens[6].text).toBe("?");
    expect(res.tokens[6].id).toBe(30);
  });

  it("tracks exact character spans in prompt", () => {
    const text = "What is";
    const res = tokenizer.tokenize(text);

    expect(res.tokens[0].startChar).toBe(0);
    expect(res.tokens[0].endChar).toBe(4);
    expect(text.slice(res.tokens[0].startChar, res.tokens[0].endChar)).toBe("What");

    expect(res.tokens[1].startChar).toBe(4);
    expect(res.tokens[1].endChar).toBe(7);
    expect(text.slice(res.tokens[1].startChar, res.tokens[1].endChar)).toBe(" is");
  });

  it("decomposes compound words into subwords", () => {
    const res = tokenizer.tokenize("Unbelievable");
    expect(res.tokens.length).toBe(3);
    expect(res.tokens[0].text).toBe("Un");
    expect(res.tokens[0].type).toBe("prompt");
    expect(res.tokens[1].text).toBe("believ");
    expect(res.tokens[1].type).toBe("subword");
    expect(res.tokens[2].text).toBe("able");
    expect(res.tokens[2].type).toBe("subword");
  });

  it("produces correct visible whitespace markers (Ġ)", () => {
    const res = tokenizer.tokenize("Hello world");
    expect(res.tokens[0].cleanText).toBe("Hello");
    expect(res.tokens[1].cleanText).toBe("Ġworld");
  });

  it("computes accurate UTF-8 byte lengths and hex strings", () => {
    const res = tokenizer.tokenize("AI");
    expect(res.tokens[0].byteLength).toBe(2);
    expect(res.tokens[0].hexBytes).toBe("0x41 0x49");
  });

  it("identifies special tokens with distinctive type", () => {
    const res = tokenizer.tokenize("<|im_start|>system<|im_end|>");
    expect(res.tokens[0].text).toBe("<|im_start|>");
    expect(res.tokens[0].isSpecial).toBe(true);
    expect(res.tokens[0].type).toBe("special");
    expect(res.tokens[0].id).toBe(100264);

    expect(res.tokens[2].text).toBe("<|im_end|>");
    expect(res.tokens[2].isSpecial).toBe(true);
    expect(res.tokens[2].id).toBe(100265);
  });

  it("detokenizes back to original text", () => {
    const text = "What is the capital of France?";
    const res = tokenizer.tokenize(text);
    const reconstructed = tokenizer.detokenize(res.tokenIds);
    expect(reconstructed).toBe(text);
  });

  it("handles empty or whitespace-only inputs without crashing", () => {
    const emptyRes = tokenizer.tokenize("");
    expect(emptyRes.tokens).toEqual([]);
    expect(emptyRes.tokenCount).toBe(0);

    const spaceRes = tokenizer.tokenize("   ");
    expect(spaceRes.tokens.length).toBe(1);
    expect(spaceRes.tokens[0].text).toBe("   ");
  });

  it("handles Unicode characters and emojis properly", () => {
    const emojiText = "Hello 🚀";
    const res = tokenizer.tokenize(emojiText);
    expect(res.tokens.length).toBe(2);
    expect(res.tokens[1].byteLength).toBeGreaterThan(1);
  });
});
