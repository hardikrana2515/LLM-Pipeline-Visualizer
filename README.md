# 🧠 LLM Pipeline Visualizer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-llm--pipeline--visualizer-10b981?style=for-the-badge&logo=vercel)](https://llm-pipeline-visualizer-eight.vercel.app/)

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![React 19](https://img.shields.io/badge/React-19.3-06b6d4.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.3-8b5cf6.svg)](https://vitejs.dev/)
[![Vitest](https://img.shields.io/badge/Tests-49%20Passed-10b981.svg)](https://vitest.dev/)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX-WASM%20%2F%20WebGPU-38bdf8.svg)](https://huggingface.co/docs/transformers.js)

> 🚀 **Live Interactive Web App**: [https://llm-pipeline-visualizer-eight.vercel.app/](https://llm-pipeline-visualizer-eight.vercel.app/)

An interactive, phase-by-phase scientific learning tool designed to demystify how **Large Language Models (LLMs)** process text, transform vectors, compute multi-head self-attention, and autoregressively generate answers.

Designed with an obsidian cyber-tech aesthetic, the visualizer bridges the gap between high-level AI concepts and raw transformer linear algebra, giving students and engineers complete transparency into every tensor.

---

## ✨ Features at a Glance

- **🎯 Phase-by-Phase Interactive Learning**: Step through the neural pipeline sequentially without being overwhelmed by cluttered dashboard panels.
- **⚡ Dual-Engine Architecture**:
  - **Educational Simulation Engine**: Deterministic, mathematically grounded JavaScript simulation that exposes internal attention query-key matrices, RoPE rotations, embeddings, and candidate logits.
  - **Real Neural Network Inference**: Run actual in-browser ONNX models (**GPT-2 Small 124M**, **Qwen 1.5 0.5B**) via Hugging Face `@huggingface/transformers` or connect to a native **Local Ollama** instance.
- **🎛️ Live Parameter Control**: Tweak Temperature ($T$), Top-$K$, and Top-$P$ (Nucleus) sampling sliders and inspect how the probability curve flattens or sharpens in real time.
- **🧪 Dual Pipeline Modes**:
  - **Inference Mode**: From raw prompt input to tokenization, embedding, transformer attention, logits projection, sampling, and autoregressive generation.
  - **Training Lab Mode**: Next-token shifting ($X \to Y$), Cross-Entropy loss calculation ($-\ln P$), Perplexity ($e^{\mathcal{L}}$), gradient backpropagation, and AdamW optimizer parameter updates.
- **⌨️ Keyboard-Driven Navigation**: Seamlessly step through stages and reroll samples using arrow keys and hotkeys.
- **♿ Inclusive & Accessible**: WCAG 2.1 AA compliant color contrast, ARIA landmarks, screen-reader status beacons, and reduced-motion support.

---

## 🔬 The 7-Stage Inference Pipeline

The visualizer breaks down autoregressive text generation into 7 distinct sequential phases:

```
[1. Prompt] ➔ [2. Tokenization] ➔ [3. Embeddings & RoPE] ➔ [4. Transformer Attention]
                                                                      │
[7. Generation] ⮜ [6. Sampling & Temp] ⮜ [5. LM Head Logits] ⮜─────────┘
```

| Phase | Name | What Happens Under the Hood | Interactive Controls |
| :---: | :--- | :--- | :--- |
| **01** | **Input Prompt** | Raw user text is formatted with special template tokens (e.g. ChatML `<\|im_start\|>`) to define role boundaries. | Prompt selector modal, curated presets (Geography, Python coding, Transformers, Story). |
| **02** | **Tokenization** | Subword Byte-Pair Encoding (BPE) splits text into token pieces, mapping each to a vocabulary ID. | Click any token chip to inspect its token ID, character span, and byte length. |
| **03** | **Embeddings & RoPE** | Token IDs index a weight matrix to produce 768-dimensional dense vectors. Rotary Position Embeddings (RoPE) rotate 2D vector pairs by $\theta_i = 10000^{-2i/d}$. | Token vector selector, vector slice preview, Cosine Similarity tab, 2D RoPE rotation visualizer. |
| **04** | **Transformer Attention** | Computes Scaled Dot-Product Attention across tokens: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ with lower-triangular causal masking. | Multi-head selector (Heads 1–4), interactive attention heatmap grid with row-normalized weights. |
| **05** | **LM Head Logits** | The final hidden state vector is projected through the LM Head unembedding matrix onto the 50,257-token vocabulary. | Candidate logits table, unscaled score meters ($z_i$), candidate ID inspection. |
| **06** | **Sampling & Softmax** | Logits are scaled by temperature $z_i / T$, filtered by Top-$K$ and Top-$P$, and converted to probabilities: $P(w_i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$. | Temperature slider ($0.0$ greedy to $1.2$ creative), Top-$K$/Top-$P$ sliders, live probability bar chart, 🎲 Sample/Reroll. |
| **07** | **Autoregressive Loop** | The sampled token is appended to the prompt sequence, creating the input context for the next forward pass. | **+ Generate Next Token** button, **Reset Sequence** button, autoregressive loop animation. |

---

## 🧪 Training Mode Lab

Toggle the **Training** pill in the top bar to inspect how models learn:

1. **Token Shifting ($X \to Y$)**: Visualizes causal sequence masking where input tokens $t_1 \dots t_{n-1}$ predict target targets $t_2 \dots t_n$.
2. **Cross-Entropy Loss**:
   $$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$
   Interactive probability slider demonstrating how low confidence on the target token penalizes the network exponentially.
3. **Perplexity (PPL)**:
   $$\text{PPL} = \exp(\mathcal{L})$$
   Calculates the model's effective branching factor.
4. **AdamW Optimizer Simulation**:
   Tracks moving averages of gradients ($m_t$) and squared gradients ($v_t$) to demonstrate decoupled weight decay updates.

---

## 🤖 Supported Models & Runtime Matrix

Click the **⚙️ Settings Gear** in the top bar to choose between the simulation engine and real neural models:

| Model | Category | License | Format & Size | Requirements | Verified Outputs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Educational Simulation** *(Default)* | In-Memory Engine | MIT | Pure JS (0 MB) | Any browser | ✓ Transparent Attention Matrices<br>✓ RoPE Rotations<br>✓ Logits & Embeddings |
| **GPT-2 Small (124M)** | Real In-Browser | MIT | ONNX `q4` (~135 MB) | WASM / WebGPU | ✓ Real Generated Tokens<br>✓ Exact Token IDs<br>✓ Latency & Throughput |
| **Qwen 1.5 (0.5B Chat)** | Real In-Browser | Apache 2.0 | ONNX `q4` (~310 MB) | WebGPU Recommended | ✓ Instruction-Tuned Output<br>✓ Multilingual Tokens |
| **Local Ollama Endpoint** | Local Host API | Dependent on Model | HTTP JSON (`localhost:11434`) | Ollama server running | ✓ Llama 3 / Mistral Output<br>✓ Zero Browser Memory Load |

> **Note on Browser WASM Memory**:
> 32-bit WebAssembly linear memory enforces contiguous allocation boundaries. For larger models (0.5B+), WebGPU acceleration or a local Ollama endpoint is recommended to prevent browser heap exhaustion (`std::bad_alloc`).

---

## 🚀 Quickstart Guide

### Prerequisites
- [Node.js](https://nodejs.org/) (version 18.0 or higher)
- `npm` or `pnpm`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/llm-pipeline-visualizer.git
   cd llm-pipeline-visualizer/llm-pipeline-visualizer
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the local development server**:
   ```bash
   npm run dev
   ```
   Open your browser and navigate to `http://localhost:5173/` (or the port indicated in your terminal).

4. **Run unit tests**:
   ```bash
   npm test
   ```

5. **Build for production**:
   ```bash
   npm run build
   ```
   The optimized production bundle will be generated in the `dist/` directory.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :---: | :--- |
| <kbd>←</kbd> / <kbd>→</kbd> | Navigate to Previous / Next pipeline phase |
| <kbd>Space</kbd> | Generate next token (when on Phase 7) |
| <kbd>P</kbd> or <kbd>/</kbd> | Open Input Prompt modal |
| <kbd>R</kbd> | Reroll categorical sampling (Phase 6) |
| <kbd>Esc</kbd> | Close active modal dialog |

---

## 📁 Project Architecture

```
llm-pipeline-visualizer/
├── index.html                     # HTML5 entry point with cyber styling tokens
├── package.json                   # Dependencies: React 19, Vite 8, Vitest 5, Transformers.js
├── vite.config.js                 # Vite bundler configuration
└── src/
    ├── App.jsx                    # Core app layout: TopBar, Stepper, Workspace, Inspector
    ├── App.css                    # App layout grid styles
    ├── index.css                  # Design system: obsidian color tokens, typography, glassmorphism
    ├── components/
    │   ├── common/
    │   │   ├── Badge.jsx          # Provenance & status badge component
    │   │   └── Badge.css
    │   ├── layout/
    │   │   ├── TopBar.jsx         # Header: mode toggle, model beacon, settings gear
    │   │   ├── PhaseStepper.jsx   # Left vertical 7-stage sequential navigation stepper
    │   │   └── PhaseBottomNav.jsx # Bottom navigation bar with keyboard hints
    │   ├── model/
    │   │   ├── ModelSelectorModal.jsx # Runtime selection, WebGPU check, model download
    │   │   └── ModelSelectorModal.css
    │   ├── prompt/
    │   │   ├── PromptModal.jsx    # Prompt input & preset library dialog
    │   │   └── PromptModal.css
    │   └── workspace/
    │       ├── PhaseWorkspace.jsx # Central workspace for each active phase
    │       ├── PhaseWorkspace.css
    │       ├── PhaseInspector.jsx # Right contextual inspector & mathematical notes
    │       └── PhaseInspector.css
    ├── data/
    │   └── constants.js           # Stage definitions, prompt presets, status enums
    ├── engine/
    │   ├── demoFixtures.js        # Deterministic fixtures & smart contextual completions
    │   ├── numericalUtils.js      # Stable softmax, RoPE, RMSNorm, cross-entropy, AdamW
    │   ├── pipelineStateMachine.js# Stage transition reducer & step validation
    │   └── __tests__/             # Unit test suites (fixtures, math, state machine)
    ├── model/
    │   ├── modelAdapter.js        # Unified adapter interface (Transformers.js, Ollama, Simulation)
    │   ├── modelConfig.js         # Supported model registry and capability gating
    │   └── __tests__/             # Asynchronous model streaming unit tests
    └── tokenizer/
        ├── subwordTokenizer.js    # BPE subword tokenizer & vocabulary mapper
        ├── tokenizerAdapter.js    # Tokenizer abstraction layer
        └── __tests__/             # Tokenizer unit tests
```

---

## 🧪 Testing & Validation

The codebase includes comprehensive unit testing with **Vitest**:

```bash
npm test
```

### Verified Test Suites (49 tests):
- `numericalUtils.test.js`: Verified numerical stability for Softmax, temperature scaling, Top-K/Top-P filtering, Cosine Similarity, RoPE angles, Cross-Entropy Loss, and AdamW gradient updates.
- `subwordTokenizer.test.js`: Validates subword segmentation, character offsets, byte lengths, and token ID reconstruction.
- `pipelineStateMachine.test.js`: Enforces sequential step transitions, stage boundaries, and mode resets.
- `demoFixtures.test.js`: Validates context-aware deterministic completions across diverse prompt categories (Geography, Python coding, Transformers, Story).
- `modelAdapter.test.js`: Confirms asynchronous streaming token callbacks and non-blocking model orchestration.

---

## 📚 References & Further Reading

- **Attention Is All You Need**: Vaswani et al., 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **RoFormer: Enhanced Transformer with Rotary Position Embedding**: Su et al., 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **Language Models are Unsupervised Multitask Learners (GPT-2)**: Radford et al., 2019. [OpenAI](https://openai.com/research/better-language-models)
- **Hugging Face Transformers.js**: In-browser machine learning runtime powered by ONNX Runtime Web. [Hugging Face](https://huggingface.co/docs/transformers.js)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
