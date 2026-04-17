
import torch
from datasets import load_dataset
from typing import Iterator

def iter_legal_documents(max_samples=None) -> Iterator[str]:
    """Stream legal documents. Falls back to legal-like text if LEDGAR unavailable."""
    try:
        # Try LEDGAR dataset first
        ds = load_dataset("LexGLUE/ledgar", split="train", streaming=True)
    except Exception as e:
        print(f"[legal_dataset] LEDGAR not available: {e}")
        print("[legal_dataset] Falling back to legal section of wikipedia/books...")
        # Fallback: Use wikitext which includes some legal content
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

    count = 0
    for ex in ds:
        if max_samples and count >= max_samples:
            break
        # Extract text from example
        text = ex.get('text', '')
        if isinstance(text, dict):
            text = text.get('text', '')
        if text and text.strip() and len(text.strip()) > 50:  # Only use substantial text
            yield text
            count += 1

def iter_ecthr_cases(max_samples=None) -> Iterator[str]:
    """Stream legal-like documents (alternative fallback)."""
    try:
        # Try ECTHR dataset
        ds = load_dataset("LexGLUE/ecthr_a", split="train", streaming=True)
    except Exception:
        # Fallback to wikitext
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

    count = 0
    for ex in ds:
        if max_samples and count >= max_samples:
            break
        text = ex.get('text', '')
        if isinstance(text, dict):
            text = text.get('text', '')
        if text and text.strip() and len(text.strip()) > 50:
            yield text
            count += 1

if __name__ == "__main__":
    # Test the iterator
    print("Testing legal document iterator...")
    for i, text in enumerate(iter_legal_documents(max_samples=3)):
        print(f"\n[Sample {i+1}] {len(text)} chars")
        print(text[:200])
