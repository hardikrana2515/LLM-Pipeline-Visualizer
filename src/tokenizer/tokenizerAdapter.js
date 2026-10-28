/**
 * LLM Pipeline Visualizer — Tokenizer Adapter Contract
 *
 * Defines the standard interface for tokenizers (both educational illustrative
 * tokenizers and real browser-based model tokenizers in later phases).
 */

/**
 * Standard token structure emitted by all tokenizers in this project.
 * @typedef {Object} TokenItem
 * @property {number} id - Vocabulary integer ID
 * @property {string} text - Raw token text piece
 * @property {string} cleanText - Text piece formatted with visible whitespace (e.g. Ġ or ·)
 * @property {number} pos - 0-indexed position in sequence
 * @property {number} startChar - Starting character index in original prompt
 * @property {number} endChar - Ending character index in original prompt
 * @property {number} byteLength - Length in bytes (UTF-8)
 * @property {string} hexBytes - Hexadecimal byte sequence (e.g. "0x46 0x72")
 * @property {string} type - 'prompt' | 'special' | 'subword' | 'punctuation' | 'number'
 * @property {boolean} isSpecial - Whether this is a control/special token
 */

/**
 * Standard Tokenizer Result.
 * @typedef {Object} TokenizationResult
 * @property {TokenItem[]} tokens - Array of token items
 * @property {number[]} tokenIds - Array of pure integer token IDs
 * @property {number} tokenCount - Total token count
 * @property {number} charCount - Total character count in original text
 * @property {string} text - Original input text
 * @property {string} provenance - 'illustrative' | 'real'
 * @property {string} tokenizerName - Identifier of tokenizer used
 */

export class TokenizerAdapter {
  /**
   * Tokenize an input string into TokenItems and token IDs.
   * @param {string} text
   * @param {Object} [options]
   * @returns {TokenizationResult}
   */
  tokenize(text, options = {}) {
    throw new Error("tokenize() must be implemented by adapter subclass.");
  }

  /**
   * Decode an array of token IDs back into a reconstructed string.
   * @param {number[]} tokenIds
   * @returns {string}
   */
  detokenize(tokenIds) {
    throw new Error("detokenize() must be implemented by adapter subclass.");
  }

  /**
   * Return metadata about the tokenizer.
   * @returns {{ name: string, type: string, vocabSize: number, provenance: string }}
   */
  getInfo() {
    throw new Error("getInfo() must be implemented by adapter subclass.");
  }
}
