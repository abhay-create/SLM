# Technical Specification: Adaptive Curriculum SLM

This document provides a highly detailed breakdown of the Small Language Model (SLM) architecture and the state-of-the-art curriculum learning pipeline deployed for its training. The methodologies included here represent a shift from naive sequence-length training toward dynamic, semantic, and structurally-aware pacing, adopting critical insights from recent research (e.g., the Gemma 4 MoE / Curriculum Addendums).

---

## 1. Core Model Architecture (`src/model.py`)
The foundation of the project is a highly optimized, causal decoder-only transformer (~50M parameters).

* **Activation Function:** Replaced standard GELU with **SwiGLU** (Swish-Gated Linear Units). SwiGLU inherently performs better for language modeling by enforcing stricter zero-mean thresholds across linear scaling components.
* **Normalization:** Uses **RMSNorm** (Root Mean Square Normalization) exclusively over standard LayerNorm. This strips mean-centering from the computation, executing faster without degrading representation performance.
* **Positional Strategy:** Learnable generic positional embeddings combined with a **256 Context Window** configuration (`configs/stage0_curriculum.yaml`).
* **Tokenizer:** A custom Byte-Pair Encoding (BPE) setup (`tokenizers/tokenizer_corpus.json`) resulting in a finely tuned ~40,000 token vocabulary limit perfectly sized for this scale.

---

## 2. Difficulty Scoring Logic (`src/score_difficulty.py`)
The curriculum mathematically ranks every token trajectory prior to training. Instead of sorting blindly by text length, it generates a composite semantic score via `curriculum_scores.npy`.

* **55% GPT-2 Perplexity:** We run offline reference models directly over the TinyStories dataset. High perplexity acts as a raw proxy indicating linguistic complexity.
* **20% Type-Token Ratio (TTR):** Lexical diversity. A high ratio indicates dense, non-repetitive vocabulary, escalating difficulty.
* **15% Token Length Penalty:** Pure geometric padding.
* **-10% Discourse Coherence Discount (Gemma 4 Insight):** We implemented a regex engine to hunt for discourse markers (*first, because, however, explicitly*). Research reveals that long documentation featuring structured causal transitions is actually **easier** for modern Multi-Head Attention to bind than medium-length abstract text. Stories heavily featuring these markers are discounted and placed earlier in the curriculum pipeline.

---

## 3. Data Division & Shared Expert Proxies (`src/curriculum_dataset.py`)
Training data handles chunk serving and bounds scaling. The file organizes the tokenized cached tensors and divides them progressively base on the `CompetenceScheduler`.

**Method: Anchor Batches (Catastrophic Forgetting Safeguard)**
* **Rationale:** In standard progressive curriculums, models quickly "forget" basic primitive syntaxes (like simple subject-verb-objects) once they advance into ultra-hard data (a vulnerability addressed by fixed routing-experts in Gemma 4 MoE blocks).
* **Implementation:** The `CurriculumStageDataset` forces **10% of every sampled query** to originate explicitly from the absolute easiest 5% pool of all available tokens. This serves as a perpetual syntax anchor no matter how far the curriculum advances.

---

## 4. The Competence Scheduler & Internal Delays (`src/curriculum_dataset.py`)
Rather than blindly expanding data access every 5,000 steps, the `CompetenceScheduler` expands dataset fraction (`initial_fraction = 0.15`) purely based on **demonstrated algorithmic competence**.

* Expansion logic executes only if validation baseline drops by a set threshold (`min_delta: 0.01`).
* **Stagnation Backoff:** `add_patience(penalty)`. If the internal metrics flag the model as unstable, the expansion freezes seamlessly regardless of how the validation loss looks. 

---

## 5. Live Training Augmentations (`train_curriculum.py`)
The actual runtime loop tracks advanced, hyper-localized metrics inspired by the newest structural training standards.

### A. Dynamic Soft-Token Context Budget
* **Rationale:** Forcing an inexperienced model to parse 256 tokens immediately results in overwhelming attention distribution. 
* **Implementation:** `get_dynamic_block_size()` strictly slices both generic inputs and labels at 64 tokens at early curriculum fractions. By explicitly removing the contextual field-of-view, we force the SLM to prioritize heavy local syntactic dependency before expanding strictly geometrically (64 → 128 → 192 → 256 max).

### B. Attention Uncertainty Signal (KV Divergence)
* **Rationale:** The global attention heads on the deepest layer handle abstract conceptual distance. If they collapse, gradients fail.
* **Implementation:** A PyTorch `register_forward_hook` sits atop the final block's `qkv_proj` module. It extracts the raw Keys and Values and measures their Cosine Similarity in real-time. If K and V diverge dramatically during a batch update (`cosine distance > 0.4`), it triggers an immediate *Attention Sink Chaos* flag, aggressively delaying curriculum expansion.

### C. Layer-Aware Gradient Stability Tracking
* **Rationale:** Spiky updates down the deepest layers permanently derail curriculum trajectory long before a validation script catches them.
* **Implementation:** We implemented an active parameter norm tracker targeting only the lowest 2 Layers immediately after `loss.backward()`. If the coefficient of variation over the timeline pool breaches `0.15`, the algorithm fires a severe freeze (`frozen_unstable_layers`), halting dataset expansion until internal backward variance stabilizes.

---

## 6. Directory Map (Where is what)

Your repository was refactored exactly to match this curriculum suite layout cleanly:

* **`/src`**
  * `model.py` (Core SLM configuration & execution)
  * `score_difficulty.py` (Perplexity mapping & Dataset preparation script)
  * `curriculum_dataset.py` (Pacing Scheduler, Dataloading hooks, and Anchor logic)
  * `logger.py` (Saves multi-tier curriculum fraction matrices gracefully)
  * `dataset.py` / `tokenizer.py` (BPE infrastructure)

* **`train_curriculum.py`** (The main Execution Script residing safely in your Root Directory)

* **`/configs`**
  * `stage0_curriculum.yaml` (Active file handling thresholds, sizes, and context parameters).

* **`/logs`** & **`/checkpoints`**
  * Outputs from active training runs. Look inside `.csv` files here to actively monitor the CV trajectory.

* **`/scripts`** & **`/tests`**
  * Profiling tools and localized component verification functions (e.g., verifying context shifts directly).
