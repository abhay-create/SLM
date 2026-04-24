# SLM Project File Architecture

This document provides a detailed breakdown of the files in this repository, including implementation details, metadata, and dependencies.

---

## 🏗️ Core Library (`src/`)

The `src/` directory contains the foundational logic for the model, data processing, and curriculum scoring.

### [src/model.py](file:///home/user21/slm/src/model.py)
- **Implements**: The `SLM` (Small Language Model) class and `SLMConfig`. A custom Transformer architecture featuring:
    - RMSNorm for stability.
    - SwiGLU activation functions.
    - Configurable Attention (Standard + learnable/RoPE positional embeddings).
- **Dependencies**: `torch`, `torch.nn`, `math`.
- **Use**: Defines the neural network structure. This file is the primary dependency for all training and inference scripts.

### [src/curriculum_dataset.py](file:///home/user21/slm/src/curriculum_dataset.py)
- **Implements**: `CurriculumStageDataset` and `CompetenceScheduler`. It manages:
    - Sorting data by pre-computed difficulty.
    - Dynamic pool expansion (Competence-based scaling).
    - Anchor batch injection to prevent catastrophic forgetting.
- **Dependencies**: `torch`, `numpy`, `pickle`, `src.dataset`.
- **Use**: Handles the adaptive curriculum data delivery during training.

### [src/score_difficulty.py](file:///home/user21/slm/src/score_difficulty.py)
- **Implements**: The scoring engine for TinyStories. It computes a composite difficulty score:
    - `0.55 * Perplexity` (via reference GPT-2).
    - `0.20 * Type-Token Ratio (TTR)`.
    - `0.15 * Token Length`.
    - `-0.10 * Coherence Bonus`.
- **Dependencies**: `torch`, `transformers` (GPT-2), `numpy`, `datasets`, `src.tokenizer`.
- **Use**: Pre-computing difficulty scores for the 2.1M story dataset.

### [src/dataset.py](file:///home/user21/slm/src/dataset.py)
- **Implements**: Standard token-level chunking, streaming iterators for TinyStories, FineWeb, and SimpleWiki, and the `StreamingStageDataset`.
- **Dependencies**: `torch`, `datasets`, `tokenizers`, `pickle`.
- **Use**: Provides the base data loading infrastructure for non-curriculum stages or val sets.

### [src/logger.py](file:///home/user21/slm/src/logger.py)
- **Implements**: `TrainingLogger`. A CSV-based logging utility that tracks loss, perplexity, and curriculum state at every evaluation step.
- **Dependencies**: `os`, `csv`, `datetime`.
- **Use**: Records training history for visualization and evaluation.

### [src/writingprompts_dataset.py](file:///home/user21/slm/src/writingprompts_dataset.py)
- **Implements**: Dataset streaming from HuggingFace (`euclaise/writingprompts`), difficulty scoring using the GPT-2 reference method, and tier buckling (Easy/Medium/Hard).
- **Dependencies**: `torch`, `datasets`, `numpy`, `transformers`.
- **Use**: Provides complex narrative data for post-expansion curriculum stages.

---

## 🚀 Training & Orchestration

### [train_curriculum.py](file:///home/user21/slm/train_curriculum.py)
- **Implements**: The primary curriculum-aware training loop. Features:
    - Adaptive pacing based on validation loss.
    - KV-Divergence monitoring for tracking representation drift.
    - Dynamic sequence length scaling.
- **Dependencies**: `torch`, `yaml`, `src.*` (model, logger, curriculum_dataset).
- **Use**: The main entry point for running the 50M parameter SLM training experiments.

### [train_expansion.py](file:///home/user21/slm/train_expansion.py)
- **Implements**: Orchestrator for multi-stage model expansion training. Features:
    - Calls `expand_model.py` operators between stages.
    - Differential learning rates for new vs pre-trained parameters.
    - Forgetting monitoring over previous datasets.
- **Dependencies**: `torch`, `src.*`, `expand_model.py`.
- **Use**: Executes the 50M → 100M layer-by-layer growth protocol.

### [start_pipeline.sh](file:///home/user21/slm/start_pipeline.sh)
- **Implements**: An automation script that sequences the scoring of data and the start of the training process.
- **Dependencies**: `bash`, `python3`.
- **Use**: Used to launch the entire project pipeline with a single command.

---

## 📈 Evaluation & Utility

### [generate.py](file:///home/user21/slm/generate.py)
- **Implements**: Lightweight inference logic for sampling text from a saved model checkpoint.
- **Dependencies**: `torch`, `src.model`, `tokenizers`.
- **Use**: Quick qualitative testing of the model's storytelling capability.

### [evaluate_curriculum.py](file:///home/user21/slm/evaluate_curriculum.py)
- **Implements**: Quantitative evaluation of curriculum checkpoints across difficulty tiers (Easy/Medium/Hard).
- **Dependencies**: `torch`, `numpy`, `src.model`.
- **Use**: Validating that the model performs well on complex samples after curriculum training.

### [analyse_scores.py](file:///home/user21/slm/analyse_scores.py)
- **Implements**: Distribution analysis of the `curriculum_scores.npy` file.
- **Dependencies**: `numpy`, `datasets`, `matplotlib` (optional).
- **Use**: Auditing the scoring algorithm to ensure it correctly segments the dataset tiers.

### [expand_model.py](file:///home/user21/slm/expand_model.py)
- **Implements**: Math-verified architectural growth scripts, including:
    - Depth expansion (layer cloning with symmetry-breaking noise).
    - FFN widening (zero-padded weight downcast for exact function preservation).
    - Context expansion via learnable interpolations.
- **Use**: Generates expanded checkpoints before the next training stage.

### [evaluate_expansion.py](file:///home/user21/slm/evaluate_expansion.py)
- **Implements**: Expansion-specific evaluations like parameter counting, validation logic (ensuring expansion correctness), output similarity, and cross-dataset metrics.
- **Use**: Run between and after growth stages to certify stability.

---

## 📂 Key Directories

- **[configs/](file:///home/user21/slm/configs/)**: Contains YAML files (e.g., `stage0_full.yaml`) defining hyperparameters, learning rate schedules, and curriculum modes.
- **[tokenizers/](file:///home/user21/slm/tokenizers/)**: Stores pre-trained BPE tokenizer binaries and configurations.
- **[Curriculum_scores/](file:///home/user21/slm/Curriculum_scores/)**: Persistent storage for high-precision difficulty scores (`.npy`).
- **[Logs/](file:///home/user21/slm/Logs/)**: Stores CSV training logs and console output for every experiment run.

---
*Created on: 2026-04-17*
