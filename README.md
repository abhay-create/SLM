# SLM Curriculum & Expansion (50M → 100M)

This repository contains the infrastructure for a **multi-stage curriculum-driven expansion** of a Small Language Model (SLM). The project bridges the domain gap between preschool-level narratives (TinyStories) and complex adult fiction (WritingPrompts) while doubling the model's parameter count through function-preserving growth.

---

## 🏗️ System Architecture

### 1. The SLM Model (`src/model.py`)
A decoder-only Transformer optimized for stability and small-scale training.
- **RMSNorm**: Used for pre-normalization in every decoder block.
- **SwiGLU**: Activation function for the FFN blocks (gate/up/down projection).
- **Rotary Embeddings (RoPE)**: Supports flexible context extension (defaulting to learnable positional embeddings in early stages).
- **Weight Tying**: The `tok_emb` and `lm_head` share parameters.
- **Base Config (50M)**: 6 Layers | 512 $d_{model}$ | 2048 $d_{ff}$ | 256 ctx | 50k Vocab.

### 2. Tokenization (`tokenizers/`)
- **Type**: Byte-Pair Encoding (BPE).
- **Vocab**: 40,000 tokens (Standardized across all stages).
- **Files**: `tokenizer_corpus.json` is the primary tokenizer used for all curriculum data.

---

## 📈 Expansion Strategy

The model grows using **Function-Preserving Model Growth (FPMG)** to minimize retraining spikes.

| Expansion Type | Technical Logic | File Reference |
| :--- | :--- | :--- |
| **Depth (Layers)** | Clones specific layers and adds Gaussian noise ($\sigma=0.01$) to break symmetry. | `expand_model.py:expand_depth` |
| **Width (FFN)** | Widens $d_{ff}$ by zero-padding the `w_down` projection. New neurons start "dormant." | `expand_model.py:expand_ffn_width` |
| **Context** | Linearly interpolates learnable positional embeddings to extend window size. | `expand_model.py:expand_context_length` |

---

## 🛣️ 6-Stage Curriculum Roadmap

The curriculum uses "Bridge Datasets" to ramp up vocabulary and logic complexity.

| Stage | Goal | Dataset Mix | Expansion Action |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Baseline | 100% TinyStories | None (Start @ 50M) |
| **Stage 2** | Causal Chains | 60% ROC / 40% SimpleStories | **9L** (+3 layers), **384 ctx** |
| **Stage 3** | Logic & Scene | 40% Simple / 40% Child / 20% Replay | None (Stability Phase) |
| **Stage 4** | Paragraphing | 60% Child / 30% Wiki / 10% Replay | **12L** (+3 layers), **512 ctx** |
| **Stage 5** | Conditioning | 80% WP Easy / 20% Replay | None (Domain Alignment) |
| **Stage 6** | Full Fiction | 70% WP / 30% Replay | **3584 FFN**, **768 ctx** |

---

## 🔬 Evaluation & Logging

### 1. Cross-Domain Tracking (`src/capability_logger.py`)
At the end of every stage, the model is evaluated on **6 simultaneous validation sets** to track growth and forgetting:
- `s0` (TinyStories), `s1` (SimpleWiki), `roc` (ROCStories), `simple` (SimpleStories), `child` (Children-Stories), `wp` (WritingPrompts).
- Results are appended to `docs/curriculum_capabilities.md`.

### 2. Plateau Detection
- **Mechanism**: Monitors `val_key` (stage-specific).
- **Exit Logic**: If loss doesn't improve by `min_delta` (0.002) for `patience` (15) intervals, the stage concludes.

### 3. Monitoring, Benchmarks and Anti-Forgetting

We now include a focused benchmarking and monitoring strategy to detect capability gains and catastrophic forgetting during expansion and curriculum training.

- **Anchor Baseline**: compute and persist TinyStories (`s0`) validation loss at start-of-stage and save it in checkpoints as `anchor_val`. Use this as the primary "forgetting" baseline.
- **Per-domain metrics (evaluated every `eval_interval`)**: validation loss and perplexity for `s0`, `s1`, `roc`, `simple`, `child`, `wp`.
- **Tiered evaluation**: `easy|medium|hard` losses via `evaluate_by_tier()` to track where learning occurs.
- **Forgetting metric**: relative forgetting = (current_anchor - anchor_val) / anchor_val. Log and map to `replay_frac` if positive.
- **Backward/Forward Transfer (BWT/FWT)**: log improvements or regressions on earlier domains after each stage as in continual learning literature (Kirkpatrick et al. 2017; Lopez-Paz & Ranzato 2017).
- **Stability signals**: per-layer (deep-layer) grad norms, global grad norm, and `kv_divergence_metric()` (uncertainty signal). If instability detected, slow curriculum expansion.
- **Operational metrics**: tokens/sec, step wall-time, GPU mem, checkpoint size.
- **Logging outputs**: a per-stage CSV in `logs/` and an appended human-friendly report in `docs/curriculum_capabilities.md` (see `src/capability_logger.py`).

YAML additions (optional): include in stage configs or training configs:

```
replay_sources:          # list of cached train_<name>_seq{seq_len}.pkl to use as replay pool
	- tinystories
	- stage0_anchor.pkl    # or absolute path
initial_replay_fraction: 0.0   # starting fraction of replay samples (0.0-1.0)
```

Implementation notes:
- The dataset supports a replay pool and `set_replay_fraction()` to dynamically control how often replay samples are drawn.
- An adaptive policy increases `replay_frac` (bounded, default ≤0.3) when forgetting is detected, and decays it when the anchor recovers.
- Checkpoints now include `anchor_val` so resumed runs keep consistent forgetting baselines.

---

## 🛠️ Execution Manual

### Running a Stage
To initiate an expansion stage (e.g., Stage 3):
```bash
python train_expansion.py --stage 3 --tokenizer tokenizers/tokenizer_corpus.json
```
*Note: The script automatically handles checkpoint loading from the previous stage, architectural expansion, and final capability logging.*

### Benchmarking
To manually run the capability logger on a specific checkpoint:
```bash
python src/capability_logger.py --checkpoint checkpoints/stage_2_best.pt --stage "Manual Check"
```

---

## 📂 File Map (Granular)

| Path | Primary Use |
| :--- | :--- |
| `src/model.py` | Neural network architecture and `SLMConfig`. |
| `src/dataset.py` | Token-level chunking and infinite dataset iterators. |
| `src/curriculum_dataset.py` | Manages mixed-dataset streaming and deterministic caching. |
| `src/capability_logger.py` | Multi-domain PPL evaluator and stylistic prober. |
| `configs/expansion_stages.yaml` | The "Source of Truth" for the 6-stage roadmap. |
| `train_expansion.py` | The main orchestrator for the expansion pipeline. |
| `expand_model.py` | Math utilities for function-preserving model growth. |
| `checkpoints/` | Stores `stage_X_best.pt` and `stage_X_final.pt`. |
| `docs/figures/` | Visualizations of learning curves and capability trade-offs. |

---

## 💡 AI Agent "Lessons Learned"
- **Plateau Detection**: Always track the **target domain** validation key, not the baseline, to prevent premature exits due to stagnation on the old domain.
- **Dataset Streaming**: For small/medium datasets, use a **local-loop iterator** (loading without `streaming=True`) to avoid network-fetch overhead and "Repo card not found" logging loops.
- **Expansion Symmetry**: When cloning layers, **Gaussian noise is mandatory**. Without it, gradients remain identical across cloned layers, and the model gains no functional capacity.

## Known Issues & Troubleshooting

1) Stage mapping confusion
- Symptom: running `python train_expansion.py --stage 2 --tokenizer tokenizers/tokenizer_corpus.json` but not finding "stage 2" hard-coded in the script.
- Explanation: `train_expansion.py` reads the per-stage configuration from `configs/expansion_stages.yaml` using the key `stage_{args.stage.lower()}` (e.g. `--stage 2` → `stage_2`). See `configs/expansion_stages.yaml` for the canonical stage definitions and hyperparameters.

2) Catastrophic forgetting observed during Stage 3 (ROCStories)
- Symptom: ROCStories validation loss increases during Stage 3 runs (examples seen in logs: val_roc ≈ 2.77 @1k → ≈3.13 @4k steps).
- Context: the ROCStories validation set is small (~278 chunks) and therefore noisy; resumed runs can misalign baselines if `anchor_val` was not persisted.
- Immediate mitigations applied in codebase:
	- Anchor baseline: the TinyStories (`s0`) loss is computed at stage start and saved to checkpoints as `anchor_val` so resumed runs measure forgetting consistently.
	- Adaptive replay: training dataset exposes `set_replay_fraction()` and the training loop maps measured forgetting → `replay_frac` (bounded, default cap 0.3) to mix replay samples into batches.
	- EMA smoothing: a short-term EMA (`forgetting_ema`) is computed from raw forgetting to avoid reacting to high-variance evals. Configurable via stage YAML keys: `forgetting_ema_alpha`, `replay_cap`, `replay_scale`, `min_replay`.
	- Logging: CSVs in `Logs/` now include `ts_forgetting` and `ts_forgetting_ema` per-eval, as well as `replay_frac` for post-hoc analysis.

3) Next diagnostic steps (recommended)
- Parse the per-stage CSV (`Logs/stage_<N>_*.csv`) and compute the time-series of `ts_forgetting` vs `anchor_val`. Compute an EMA to verify the smoothing parameter is appropriate.
- Run a short dry-run (reduced `max_tokens`) with `forgetting_ema_alpha` tuned (recommend 0.05–0.3) to observe stable `replay_frac` ramps.
- If replay is insufficient, consider stronger regularizers (EWC or online KD) as a secondary safeguard.

4) Quick commands
- Inspect the YAML stage keys:

```bash
sed -n '1,240p' configs/expansion_stages.yaml
```

- List recent logs and CSVs:

```bash
ls -lah Logs/ | sed -n '1,200p'
tail -n 120 Logs/train_stage3.log
```


