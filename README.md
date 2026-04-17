# Adaptive Curriculum Training for SLM

This repository implements an advanced, competence-aware adaptive curriculum for training a **50M-parameter Small Language Model (SLM)** on the TinyStories dataset. The pipeline is designed to accelerate convergence and improve final model quality by dynamically scaling data difficulty.

## 🚀 Key Features

*   **Difficulty Scoring**: Uses a composite metric combining GPT-2 Perplexity (PPL), Type-Token Ratio (TTR), Narrative Length, and a Discourse Coherence Bonus.
*   **Adaptive Scheduling**: Implements a `CompetenceScheduler` that monitors validation loss in real-time to expand the training pool complexity.
*   **Forgetting Prevention**: Uses **Anchor Batch Injection** (5% easiest samples) and **Stability Monitoring** (KV-Divergence tracking) to ensure stable representation learning.
*   **Optimized Pipeline**: Vectorized scoring and unbuffered background logging for high-throughput training.

## 📂 Repository Structure

- `src/`: Core modules (Model, Tokenizer, Dataset, Scoring logic).
- `configs/`: YAML configurations for training stages.
- `Curriculum_scores/`: Pre-computed dataset difficulty mappings.
- `Logs/`: Training metrics and validation logs.
- `checkpoints/`: Model state dictionaries and curriculum metadata.
- `scripts/`: Utility scripts for evaluation and analysis.

## 🛠️ Quick Start

### 1. Scoring the Dataset
To generate or update difficulty scores for the corpus:
```bash
python src/score_difficulty.py --output curriculum_scores.npy --resume
```

### 2. Launching Training
To start the adaptive curriculum training loop:
```bash
python train_curriculum.py --config configs/stage0_full.yaml
```

### 3. Text Generation
To query the trained model:
```bash
python generate.py
```

---
*Developed by the Google DeepMind team for Advanced Agentic Coding.*
