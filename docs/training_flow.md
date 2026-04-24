Training Flow (detailed)
=========================

This document traces the full training flow starting from `train_expansion.py` and lists every module/function that is invoked during a single expansion-stage run.

High-level entry
----------------
- Caller: run from CLI
  - `python train_expansion.py --stage 2 --tokenizer tokenizers/tokenizer_corpus.json`
  - Entrypoint: [train_expansion.py](train_expansion.py)

Detailed step-by-step flow
--------------------------
1. CLI parsing and dispatch
   - `parse_args()` constructs an `argparse.Namespace`.
   - `train_expansion(parse_args())` is invoked. See [train_expansion.py](train_expansion.py).

2. Load stage config
   - YAML file: `configs/expansion_stages.yaml` is loaded.
   - Key used: `stage_key = f"stage_{args.stage.lower()}"` → loads `stage_2` for `--stage 2`.
   - File: [train_expansion.py](train_expansion.py)

3. Tokenizer & checkpoint load
   - Tokenizer: `Tokenizer.from_file(args.tokenizer)` loads `tokenizers/tokenizer_corpus.json`.
   - Source checkpoint: `torch.load(cfg['source_checkpoint'])` loads previous-stage checkpoint.
   - The checkpoint contains `config` and `model_state` used to instantiate `SLM`.
   - Files: [train_expansion.py](train_expansion.py), [src/model.py](src/model.py)

4. Model expansion (if configured)
   - Depth expansion: `expand_model.expand_depth()` clones layers and adds symmetry-breaking noise.
   - FFN widening: `expand_model.expand_ffn_width()` widens FFN with function-preserving init.
   - Context extension: `expand_model.expand_context_length()` interpolates positional embeddings.
   - Each expansion is validated via `expand_model.validate_expansion()`.
   - Files: [expand_model.py](expand_model.py)

5. Training setup
   - Build optimizer with differential learning rates: `build_expansion_optimizer()` (in `train_expansion.py`).
   - Mixed precision setup: `GradScaler`, `autocast`.
   - Curriculum dataset: `CurriculumStageDataset().build(...)` constructs chunked dataset, ordering, and optionally loads replay pool.
     - Under the hood this uses tokenization and chunking utilities from `src/dataset.py` when caches are built.
   - Competence scheduler: `CompetenceScheduler()` (from `src/curriculum_dataset.py`) is initialized for `adaptive` mode.
   - Validation loaders: `load_all_val_sets(tokenizer)` builds DataLoaders for `s0`, `s1`, `roc`, `simple`, `child`, `wp`.
   - Files: [train_expansion.py](train_expansion.py), [src/curriculum_dataset.py](src/curriculum_dataset.py), [src/dataset.py](src/dataset.py)

6. Training loop (main)
   - Outer loop rebuilds curriculum dataloader each epoch via `make_curriculum_dataloader()` (in `train_expansion.py` → wrapper in `src/curriculum_dataset.py`).
   - Per-batch steps:
     - Compute dynamic context size: `get_dynamic_block_size()` (from `train_curriculum.py`).
     - Clamp token IDs and prepare tensors.
     - Update learning rates: `update_lr_groups()` (train_expansion uses differential LR groups).
     - Forward pass inside `autocast`, compute loss via `model(x, y)` — model defined in `src/model.py`.
     - Backward: `scaler.scale(loss).backward()` (non-bf16) or `loss.backward()` (bf16 path), gradient clipping.
     - Track gradient statistics (deep-layer grad norms, global grad norm).
     - Optimizer step: `scaler.step(optimizer)` or `optimizer.step()`.
   - Files: [train_expansion.py](train_expansion.py), [src/model.py](src/model.py), [train_curriculum.py](train_curriculum.py)

7. Periodic evaluation & curriculum update
   - Every `eval_interval` steps:
     - Compute validation losses for all val loaders via `evaluate(model, loader)` (in `train_curriculum.py`).
     - Per-tier evaluation: `evaluate_by_tier(model, dataset)` (train_curriculum).
     - Compute `kv_divergence_metric(model, x)` to detect representation instability.
     - Update `CompetenceScheduler.update_competence()` and call `train_ds.set_eligible_fraction()` to expand difficulty.
     - Adaptive replay policy: compute TinyStories forgetting vs `anchor_val` and adjust `train_ds.set_replay_fraction()` (if replay pool available).
     - Save best checkpoint via `save_checkpoint()` (now includes optional `anchor_val`).
     - Log metrics via `TrainingLogger.log()` (writes CSV with extended columns: `replay_frac`, `ts_forgetting`, `grad_norm`, ...).
   - Files: [train_expansion.py](train_expansion.py), [train_curriculum.py](train_curriculum.py), [src/logger.py](src/logger.py), [src/curriculum_dataset.py](src/curriculum_dataset.py)

8. Exit and capability logging
   - When training ends (token budget, plateau, spike, etc.) the script calls `logger.log_exit()` and prints final summary.
   - If a best checkpoint exists, `train_expansion.py` calls `run_capability_logging(best_ckpt_path, tokenizer, stage_name)` to record cross-domain metrics and generations in `docs/curriculum_capabilities.md`.
   - Files: [train_expansion.py](train_expansion.py), [src/capability_logger.py](src/capability_logger.py)

Supporting utilities called during flow
-------------------------------------
- `src/model.py`: SLM model, forward, generation, `.num_params()` — fundamental network computations.
- `expand_model.py`: expansion helpers and checkpoint creation for function-preserving growth.
- `src/dataset.py`: tokenization, chunking, val-set builders, replay chunk loaders.
- `src/curriculum_dataset.py`: curriculum ordering, CompetenceScheduler, CurriculumStageDataset sampling logic (anchor injection & replay sampling).
- `src/logger.py`: CSV and console logging. Files written to `logs/`.
- `train_curriculum.py`: evaluation helpers, LR schedule, early-exit detectors, checkpoint helpers.
- `src/capability_logger.py`: end-of-stage capability reports and stylized generation probes.

Notes on replay & forgetting
---------------------------
- Replay pool sources should be provided as cached `train_<name>_seq{seq}.pkl` files (see `cache/`).
- The adaptive policy increases a controlled `replay_frac` (bounded by default to 0.3) when TinyStories forgetting is observed, then decays when the anchor recovers.
- Checkpoints include `anchor_val` where available to provide consistent baselines across resumed runs.

Files modified to reflect this flow
----------------------------------
- `train_expansion.py` — orchestrator and expansion training loop
- `train_curriculum.py` — evaluation helpers, anchor/replay policies, checkpoint anchor storage
- `src/curriculum_dataset.py` — replay pool loading and `set_replay_fraction()` API
- `src/logger.py` — added columns and host/pid filename behaviour
- `src/capability_logger.py` — forgetting report vs `anchor_val`

If you want, I can now:
- Add line-numbered references for key call sites inside each file, or
- Implement EMA smoothing for the forgetting signal before mapping to `replay_frac`, or
- Add a small plotting utility to visualize `ts_forgetting`, `replay_frac`, and `grad_norm` from the CSV logs.

