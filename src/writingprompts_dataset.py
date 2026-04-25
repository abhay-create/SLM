"""
writingprompts_dataset.py — WritingPrompts text iterator.

HuggingFace dataset: "euclaise/writingprompts"
Format: {"prompt": str, "story": str}

Supports both a looping train split and a one-pass validation split.
The validation split uses a hash-based deterministic document split (S-5)
so the same documents are always held out regardless of iterator order.
"""

import hashlib
from typing import Iterator, Optional


def _wp_hash_is_val(text: str, val_fraction: float = 0.05,
                    seed: str = "slm_wp_val") -> bool:
    """Returns True if this document belongs in the validation set (deterministic hash split)."""
    key = (seed + text[:64]).encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16)
    return (h % 10000) < int(val_fraction * 10000)


def iter_writingprompts(
    split: str = "train",
    max_length_filter: Optional[int] = None,
    val_fraction: float = 0.05,
) -> Iterator[str]:
    """
    Yields WritingPrompts examples formatted as '[WP] <prompt>\\n\\n<story>'.

    Args:
        split: 'train' loops indefinitely; 'validation' iterates once.
        max_length_filter: If set, skips stories longer than this many words.
        val_fraction: Fraction of documents held out for validation (hash-based).
    """
    from datasets import load_dataset
    ds = load_dataset("euclaise/writingprompts", split="train")

    def _format(ex: dict) -> Optional[str]:
        prompt = ex.get("prompt", "").strip()
        story = ex.get("story", ex.get("text", "")).strip()
        if not story:
            return None
        if max_length_filter and len(story.split()) > max_length_filter:
            return None
        return f"[WP] {prompt}\n\n{story}" if prompt else story

    if split == "train":
        # Loop indefinitely; exclude validation documents
        while True:
            for ex in ds:
                text = _format(ex)
                if text is None:
                    continue
                if not _wp_hash_is_val(text, val_fraction):
                    yield text
    else:
        # Validation: iterate once, only yield val-split documents
        for ex in ds:
            text = _format(ex)
            if text is None:
                continue
            if _wp_hash_is_val(text, val_fraction):
                yield text
