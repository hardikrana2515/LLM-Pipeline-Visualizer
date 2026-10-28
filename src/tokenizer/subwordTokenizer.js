/**
 * LLM Pipeline Visualizer — Subword BPE Tokenizer (Educational Illustrative Engine)
 *
 * Implements TokenizerAdapter for Phase 3:
 * - Subword decomposition & vocabulary mapping
 * - Whitespace tracking (Ġ marker for leading whitespace)
 * - Exact character span tracking [startChar, endChar]
 * - UTF-8 byte length & Hex byte representation
 * - Special token detection (<|im_start|>, <|im_end|>, <|endoftext|>, etc.)
 * - Token categorization: 'prompt' | 'subword' | 'special' | 'punctuation' | 'number'
 * - Detokenization round-trip
 */

import { TokenizerAdapter } from "./tokenizerAdapter";

// Curated vocabulary mapping for standard educational examples
export const KNOWN_VOCAB = {
  // Special Tokens
  "<|im_start|>": 100264,
  "<|im_end|>": 100265,
  "<|endoftext|>": 50256,
  "<s>": 1,
  "</s>": 2,
  "<pad>": 0,
  "<unk>": 3,

  // Common prompt tokens (without leading space)
  "What": 2061,
  "Explain": 34354,
  "Once": 7454,
  "Write": 14352,
  "The": 464,
  "In": 554,
  "A": 32,
  "How": 1374,
  "Un": 451,
  "believ": 9821,
  "able": 317,
  "token": 11241,
  "ization": 3280,
  "transform": 9032,
  "er": 254,

  // Common tokens with leading space
  " is": 318,
  " the": 262,
  " capital": 3139,
  " of": 286,
  " France": 4881,
  " upon": 2402,
  " a": 257,
  " time": 640,
  " in": 287,
  " land": 1956,
  " far": 1400,
  " away": 1419,
  " there": 612,
  " lived": 4410,
  " how": 703,
  " transformer": 39446,
  " models": 4981,
  " work": 670,
  " simple": 2829,
  " terms": 2524,
  " Python": 12444,
  " function": 1407,
  " that": 326,
  " calculates": 37412,
  " fibonacci": 31191,
  " sequence": 8245,
  " Paris": 6342,
  " Europe": 5931,
  " young": 1862,
  " king": 3797,
  " who": 508,
  " dreamed": 14751,
  " exploring": 12560,
  " stars": 5364,

  // Punctuation
  "?": 30,
  ".": 13,
  ",": 11,
  "!": 0,
  ":": 25,
  ";": 26,
  "(": 7,
  ")": 8,
  "[": 9,
  "]": 10,
};

// Known subword decompositions for educational demonstration
const SUBWORD_PATTERNS = [
  { word: "Unbelievable", pieces: ["Un", "believ", "able"] },
  { word: "unbelievable", pieces: ["un", "believ", "able"] },
  { word: "tokenization", pieces: ["token", "ization"] },
  { word: "Tokenization", pieces: ["Token", "ization"] },
  { word: "transformers", pieces: ["transform", "ers"] },
  { word: "Transformers", pieces: ["Transform", "ers"] },
  { word: "superconductivity", pieces: ["super", "conduct", "ivity"] },
];

// Special tokens regex pattern
const SPECIAL_TOKEN_REGEX = /<\|[^|]+\|>|<[^>]+>/g;

export class SubwordTokenizer extends TokenizerAdapter {
  constructor() {
    super();
    this.vocab = { ...KNOWN_VOCAB };
    this.reverseVocab = new Map();
    for (const [text, id] of Object.entries(this.vocab)) {
      this.reverseVocab.set(id, text);
    }
  }

  getInfo() {
    return {
      name: "Illustrative Byte-Level BPE",
      type: "BPE (Byte-Pair Encoding)",
      vocabSize: 32000,
      provenance: "illustrative",
      description:
        "Deterministic educational BPE tokenizer demonstrating subword splitting, byte tracking, and whitespace markers.",
    };
  }

  /**
   * Deterministically hashes a token string into vocabulary integer ID
   * when not in predefined dictionary.
   */
  hashToken(tokenText) {
    if (this.vocab[tokenText] !== undefined) {
      return this.vocab[tokenText];
    }
    let hash = 0;
    for (let i = 0; i < tokenText.length; i++) {
      hash = (hash << 5) - hash + tokenText.charCodeAt(i);
      hash |= 0;
    }
    const id = Math.abs(hash % 31000) + 100;
    this.vocab[tokenText] = id;
    this.reverseVocab.set(id, tokenText);
    return id;
  }

  /**
   * Encodes a string into UTF-8 hex byte string (e.g. "0x46 0x72")
   */
  getHexBytes(text) {
    const bytes = new TextEncoder().encode(text);
    return Array.from(bytes)
      .map((b) => "0x" + b.toString(16).toUpperCase().padStart(2, "0"))
      .join(" ");
  }

  /**
   * Formats a token with visible whitespace symbol (Ġ).
   */
  formatVisibleText(text) {
    if (text.startsWith(" ")) {
      return "Ġ" + text.slice(1);
    }
    if (text === " ") {
      return "Ġ";
    }
    return text;
  }

  /**
   * Determines token category.
   */
  determineTokenType(text, isSpecial) {
    if (isSpecial) return "special";
    if (/^[.,!?:;()[\]{}"'`\-\/\\@#$%^&*+=<>~]+$/.test(text.trim())) {
      return "punctuation";
    }
    if (/^\s*\d+\s*$/.test(text)) {
      return "number";
    }
    return "prompt";
  }

  /**
   * Main tokenization algorithm.
   */
  tokenize(rawText, options = {}) {
    if (!rawText || typeof rawText !== "string") {
      return {
        tokens: [],
        tokenIds: [],
        tokenCount: 0,
        charCount: 0,
        text: "",
        provenance: "illustrative",
        tokenizerName: this.getInfo().name,
      };
    }

    const tokens = [];
    const textLen = rawText.length;
    let cursor = 0;

    // Regex to extract words, punctuation, whitespace chunks, or special tokens
    // Uses Unicode flag 'u' to handle emojis and multi-byte code points properly
    const tokenRegex = /(<\|[^|]+\|>|<[^>]+>)|(\s*[\p{L}\p{N}]+)|(\s*[^\s\p{L}\p{N}]+)|(\s+)/gu;
    let match;

    while ((match = tokenRegex.exec(rawText)) !== null) {
      const matchText = match[0];
      const matchStart = match.index;
      const matchEnd = matchStart + matchText.length;
      const isSpecial = Boolean(match[1]);

      // Check if word decomposes into known subwords
      const cleanWord = matchText.trim();
      const subwordMatch = !isSpecial && SUBWORD_PATTERNS.find(
        (p) => p.word.toLowerCase() === cleanWord.toLowerCase()
      );

      if (subwordMatch) {
        // Decompose into subwords
        const leadingSpace = matchText.startsWith(" ") ? " " : "";
        let subCursor = matchStart;

        subwordMatch.pieces.forEach((piece, pIdx) => {
          const pieceText = pIdx === 0 && leadingSpace ? `${leadingSpace}${piece}` : piece;
          const pieceStart = subCursor;
          const pieceEnd = pieceStart + pieceText.length;
          subCursor = pieceEnd;

          const id = this.hashToken(pieceText);
          const byteLen = new TextEncoder().encode(pieceText).length;

          tokens.push({
            id,
            text: pieceText,
            cleanText: this.formatVisibleText(pieceText),
            pos: tokens.length,
            startChar: pieceStart,
            endChar: pieceEnd,
            byteLength: byteLen,
            hexBytes: this.getHexBytes(pieceText),
            type: pIdx === 0 ? "prompt" : "subword",
            isSpecial: false,
          });
        });
      } else {
        const id = this.hashToken(matchText);
        const byteLen = new TextEncoder().encode(matchText).length;

        tokens.push({
          id,
          text: matchText,
          cleanText: this.formatVisibleText(matchText),
          pos: tokens.length,
          startChar: matchStart,
          endChar: matchEnd,
          byteLength: byteLen,
          hexBytes: this.getHexBytes(matchText),
          type: this.determineTokenType(matchText, isSpecial),
          isSpecial: isSpecial,
        });
      }
    }

    return {
      tokens,
      tokenIds: tokens.map((t) => t.id),
      tokenCount: tokens.length,
      charCount: textLen,
      text: rawText,
      provenance: "illustrative",
      tokenizerName: this.getInfo().name,
    };
  }

  /**
   * Reconstruct original string from token IDs.
   */
  detokenize(tokenIds) {
    if (!tokenIds || !Array.isArray(tokenIds)) return "";
    return tokenIds
      .map((id) => this.reverseVocab.get(id) || `[${id}]`)
      .join("");
  }
}

// Singleton export
export const defaultTokenizer = new SubwordTokenizer();

/**
 * Convenience helper to tokenize prompt text and return token array.
 * @param {string} text
 * @returns {Array<object>}
 */
export function tokenizePrompt(text) {
  return defaultTokenizer.tokenize(text).tokens;
}
