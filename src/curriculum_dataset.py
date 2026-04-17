"""
curriculum_dataset.py — Curriculum-aware dataset and competence scheduler.

Components:
  CompetenceScheduler  — Adaptive difficulty expansion based on val loss
  CurriculumStageDataset — Dataset that serves stories filtered by difficulty

Supports 4 modes:
  random     — No curriculum (baseline, standard random shuffle)
  length     — Sort by token length (ipynb's naive approach)
  perplexity — Sort by perplexity-based composite score (fixed linear schedule)
  adaptive   — Perplexity-based with competence-aware pacing (recommended)

Usage:
  from curriculum_dataset import CurriculumStageDataset, CompetenceScheduler
"""

import os
import random
import pickle
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─── Competence Scheduler ─────────────────────────────────────────────────────

class CompetenceScheduler:
    """
    Tracks model competence via validation loss and controls the difficulty
    fraction of training data exposed to the model.

    The scheduler adapts how quickly it expands the difficulty pool:
    - Strong improvement (>1% loss decrease) → fast expansion (+5%)
    - Moderate improvement (>0.1%)           → normal expansion (+2%)
    - Stagnating                             → hold (force +1% after 3 stalls)
    """

    def __init__(
        self,
        initial_fraction: float = 0.15,
        max_fraction: float = 1.0,
        min_fraction: float = 0.05,
        strong_threshold: float = 0.01,
        moderate_threshold: float = 0.001,
        strong_expansion: float = 0.05,
        moderate_expansion: float = 0.02,
        force_expansion: float = 0.01,
        patience: int = 3,
    ):
        self.current_fraction = initial_fraction
        self.max_fraction = max_fraction
        self.min_fraction = min_fraction
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold
        self.strong_expansion = strong_expansion
        self.moderate_expansion = moderate_expansion
        self.force_expansion = force_expansion
        self.patience_limit = patience

        # Internal state
        self.val_loss_history = []
        self.improvement_rate = 0.0
        self.patience_counter = 0
        self.fraction_history = [initial_fraction]

    def add_patience(self, penalty: int = 2):
        """Externally delay curriculum expansion (e.g. from KV divergence spike)."""
        self.patience_counter += penalty

    def update_competence(self, val_loss: float, deep_layers_stable: bool = True) -> dict:
        """
        Update curriculum fraction based on current validation loss.
        Returns a dict with status information for logging.
        """
        self.val_loss_history.append(val_loss)

        if len(self.val_loss_history) < 3:
            status = "warming_up"
            expansion = 0.0
        else:
            # Compute improvement rate: recent 3 vs older 3
            recent_avg = np.mean(self.val_loss_history[-3:])
            if len(self.val_loss_history) >= 6:
                older_avg = np.mean(self.val_loss_history[-6:-3])
            else:
                older_avg = self.val_loss_history[0]

            self.improvement_rate = (older_avg - recent_avg) / max(older_avg, 1e-6)

            if self.improvement_rate > self.strong_threshold:
                expansion = self.strong_expansion
                self.patience_counter = 0
                status = "strong_improvement"
            elif self.improvement_rate > self.moderate_threshold:
                expansion = self.moderate_expansion
                self.patience_counter = 0
                status = "moderate_improvement"
            else:
                expansion = 0.0
                self.patience_counter += 1
                status = "stagnating"

            # Force expansion if stagnating too long
            if self.patience_counter >= self.patience_limit:
                expansion = self.force_expansion
                self.patience_counter = 0
                status = "forced_expansion"

            if not deep_layers_stable and status in ("strong_improvement", "moderate_improvement"):
                # Freeze expansion because internal representations are still chaotic
                expansion = 0.0
                status = "frozen_unstable_layers"
                self.patience_counter += 1

        # Apply expansion
        old_fraction = self.current_fraction
        self.current_fraction = min(
            self.current_fraction + expansion,
            self.max_fraction,
        )
        self.fraction_history.append(self.current_fraction)

        return {
            "status": status,
            "fraction": self.current_fraction,
            "old_fraction": old_fraction,
            "expansion": expansion,
            "improvement_rate": self.improvement_rate,
            "patience_counter": self.patience_counter,
            "val_loss": val_loss,
        }

    def get_current_fraction(self) -> float:
        """Return the current eligible fraction (0.0 to 1.0)."""
        return self.current_fraction

    def state_dict(self) -> dict:
        """Save scheduler state for checkpoint resume."""
        return {
            "current_fraction": self.current_fraction,
            "val_loss_history": self.val_loss_history,
            "improvement_rate": self.improvement_rate,
            "patience_counter": self.patience_counter,
            "fraction_history": self.fraction_history,
        }

    def load_state_dict(self, state: dict):
        """Restore scheduler state from checkpoint."""
        self.current_fraction = state["current_fraction"]
        self.val_loss_history = state["val_loss_history"]
        self.improvement_rate = state["improvement_rate"]
        self.patience_counter = state["patience_counter"]
        self.fraction_history = state.get("fraction_history", [self.current_fraction])


# ─── Fixed Schedule (for 'length' and 'perplexity' modes without adaptivity) ─

def get_linear_fraction(current_step: int, max_steps: int,
                        initial: float = 0.15) -> float:
    """Linear expansion from initial_fraction to 1.0 over training."""
    return min(initial + (current_step / max(max_steps, 1)) * (1.0 - initial), 1.0)


def get_phase_fraction(current_step: int, max_steps: int) -> float:
    """Fixed 3-phase schedule (matches ipynb's approach for comparison)."""
    progress = current_step / max(max_steps, 1)
    if progress < 0.25:
        return 0.30
    elif progress < 0.60:
        return 0.60
    else:
        return 1.00


# ─── Curriculum Dataset ───────────────────────────────────────────────────────

class CurriculumStageDataset(Dataset):
    """
    Wraps pre-chunked TinyStories data with curriculum-aware sampling.

    The dataset loads pre-cached token chunks (from dataset.py's cache) and
    pre-computed difficulty scores (from score_difficulty.py). It filters
    the available chunks based on the current curriculum fraction.

    Modes:
      random     — Serve all chunks in random order (no curriculum)
      length     — Sort chunks by token length, apply linear schedule
      perplexity — Sort chunks by perplexity composite, apply linear schedule
      adaptive   — Sort by perplexity composite, expand via CompetenceScheduler
    """

    def __init__(self, anchor_frac: float = 0.10):
        self.chunks = []           # List of token-id lists
        self.scores = None         # (N, 5) array from score_difficulty.py
        self.sorted_indices = []   # Chunk indices sorted by difficulty
        self.eligible_count = 0    # How many chunks are currently eligible
        self.mode = "random"
        self.anchor_frac = anchor_frac
        self.anchor_count = 1

    def build(
        self,
        dataset_name: str,
        tokenizer,
        seq_len: int,
        max_tokens: int,
        cache_dir: str = "cache",
        scores_path: str = "curriculum_scores.npy",
        mode: str = "adaptive",
        initial_fraction: float = 0.15,
    ) -> "CurriculumStageDataset":
        """
        Load or build the chunked dataset, then set up curriculum ordering.
        """
        self.mode = mode

        # ── Load or build chunks (reuse existing cache logic) ─────────────────
        cache_path = os.path.join(cache_dir, f"train_{dataset_name}_seq{seq_len}.pkl")

        if os.path.exists(cache_path):
            print(f"[curriculum] Loading cached chunks from {cache_path}")
            with open(cache_path, "rb") as f:
                self.chunks = pickle.load(f)
            print(f"[curriculum] Loaded {len(self.chunks):,} chunks")
        else:
            from src.dataset import tokenize_and_chunk, get_train_iter
            print(f"[curriculum] Building chunks for '{dataset_name}' "
                  f"(seq={seq_len}, max_tokens={max_tokens:,})...")
            text_iter = get_train_iter(dataset_name)
            self.chunks = tokenize_and_chunk(text_iter, tokenizer, seq_len, max_tokens)
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.chunks, f)
            print(f"[curriculum] Saved {len(self.chunks):,} chunks → {cache_path}")

        n_chunks = len(self.chunks)

        # ── Set up difficulty ordering ────────────────────────────────────────
        if mode == "random":
            # Random order — no curriculum
            self.sorted_indices = list(range(n_chunks))
            random.shuffle(self.sorted_indices)
            self.eligible_count = n_chunks
            print(f"[curriculum] Mode: RANDOM (no curriculum, all {n_chunks:,} chunks)")

        elif mode == "length":
            # Sort by chunk token length (ipynb approach)
            lengths = [len(c) for c in self.chunks]
            self.sorted_indices = sorted(range(n_chunks), key=lambda i: lengths[i])
            self.eligible_count = max(1, int(n_chunks * initial_fraction))
            print(f"[curriculum] Mode: LENGTH-BASED")
            print(f"[curriculum] Initial: {self.eligible_count:,}/{n_chunks:,} "
                  f"({initial_fraction*100:.0f}%) chunks")

        elif mode in ("perplexity", "adaptive"):
            # Sort by pre-computed difficulty scores
            if not os.path.exists(scores_path):
                raise FileNotFoundError(
                    f"Difficulty scores not found at '{scores_path}'. "
                    f"Run score_difficulty.py first."
                )

            self.scores = np.load(scores_path)
            print(f"[curriculum] Loaded {len(self.scores):,} difficulty scores")

            # Map story indices to chunk indices.
            # scores are per-story, chunks are sequential windows.
            # We need to map each chunk back to its source story.
            # Strategy: build a chunk→difficulty mapping based on the
            # sequential ordering of stories in the dataset.
            chunk_difficulties = self._map_scores_to_chunks(n_chunks)

            self.sorted_indices = sorted(
                range(n_chunks), key=lambda i: chunk_difficulties[i]
            )
            self.eligible_count = max(1, int(n_chunks * initial_fraction))

            mode_label = "ADAPTIVE" if mode == "adaptive" else "PERPLEXITY (fixed)"
            print(f"[curriculum] Mode: {mode_label}")
            print(f"[curriculum] Initial: {self.eligible_count:,}/{n_chunks:,} "
                  f"({initial_fraction*100:.0f}%) chunks")
            
        self.anchor_count = max(1, int(n_chunks * 0.05)) # bottom 5%
        print(f"[curriculum] Anchor pool: {self.anchor_count:,} chunks (freq: {self.anchor_frac*100:.0f}%)")

        return self

    def _map_scores_to_chunks(self, n_chunks: int) -> list:
        """
        Map pre-scored story-level difficulties to chunk-level.

        The scores array has one entry per story (sorted by difficulty).
        The chunks are sequential overlapping windows from concatenated stories.
        We assign each chunk a difficulty based on the rough position in the
        dataset: earlier chunks (from earlier stories) get lower difficulty.

        For simplicity, we use the score's difficulty ranking directly:
        rank the chunks by their position and assign difficulties proportionally.
        """
        if self.scores is None:
            return [0.0] * n_chunks

        n_scores = len(self.scores)

        # Simple approach: assume chunks are roughly in story order
        # Map chunk index → approximate difficulty via interpolation
        # The scores are already sorted by difficulty, and story indices
        # tell us which stories are easy/hard
        story_to_difficulty = {}
        for row in self.scores:
            story_idx = int(row[0])
            difficulty = row[1]
            story_to_difficulty[story_idx] = difficulty

        # Assign each chunk a difficulty score based on its position.
        # Since chunks are created sequentially from streaming stories,
        # chunk i roughly corresponds to story i * (n_scores / n_chunks).
        chunk_difficulties = []
        for chunk_idx in range(n_chunks):
            # Map chunk position to approximate story index
            approx_story = int(chunk_idx * n_scores / max(n_chunks, 1))
            approx_story = min(approx_story, n_scores - 1)
            # Get difficulty from the original (unsorted) story ordering
            if approx_story in story_to_difficulty:
                chunk_difficulties.append(story_to_difficulty[approx_story])
            else:
                # Fallback: use position-based interpolation from sorted scores
                chunk_difficulties.append(self.scores[approx_story, 1])

        return chunk_difficulties

    def set_eligible_fraction(self, fraction: float):
        """Update how many chunks (by difficulty rank) are eligible for sampling."""
        n = len(self.chunks)
        self.eligible_count = max(1, int(n * min(fraction, 1.0)))

    def set_eligible_from_step(self, current_step: int, max_steps: int):
        """For non-adaptive modes: compute fraction from step."""
        if self.mode == "length":
            fraction = get_phase_fraction(current_step, max_steps)
        elif self.mode == "perplexity":
            fraction = get_linear_fraction(current_step, max_steps)
        else:
            return  # adaptive mode uses CompetenceScheduler externally
        self.set_eligible_fraction(fraction)

    def __len__(self):
        return self.eligible_count

    def __getitem__(self, idx):
        # Anchor injection: chance to fetch from the easiest 5% pool
        if self.mode != "random" and random.random() < self.anchor_frac:
            actual_idx = self.sorted_indices[random.randint(0, self.anchor_count - 1)]
        else:
            # Map idx through the difficulty-sorted ordering
            if self.mode == "random":
                actual_idx = self.sorted_indices[idx % len(self.sorted_indices)]
            else:
                actual_idx = self.sorted_indices[idx % self.eligible_count]

        chunk = torch.tensor(self.chunks[actual_idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]  # (input, target)


# ─── DataLoader factory (compatible with repo's interface) ────────────────────

def make_curriculum_dataloader(
    dataset: CurriculumStageDataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from a CurriculumStageDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
