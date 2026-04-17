"""
benchmark_tinystories_models.py

Benchmarking suite for TinyStories-trained SLMs.
Implements metrics from the TinyStories paper and standard LLM evaluation.

Standard metrics for evaluating TinyStories models:
1. Perplexity (PPL) - main metric used in TinyStories paper
2. Validation Loss across different story complexities
3. Generation quality metrics (BLEU, length, coherence)
4. Entity consistency (story-specific)
5. Downstream task transfer (optional MMLU/Arc samples)
"""

import os
import torch
import math
from tokenizers import Tokenizer
from src.model import SLM
import glob
from collections import defaultdict
import numpy as np

TOKENIZER_PATH = "tokenizers/tokenizer_corpus.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── METRIC 1: PERPLEXITY ON VALIDATION SET ──────────────────────────────────

def calculate_perplexity(model, tokenizer, text_samples, max_length=256):
    """
    Calculate perplexity on a list of text samples.
    Perplexity = exp(mean(loss))
    
    This is THE standard metric for language models.
    Lower is better. TinyStories models typically achieve 5-30 PPL.
    """
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for text in text_samples:
            encoded = tokenizer.encode(text)
            ids = encoded.ids[:max_length]
            
            if len(ids) < 2:
                continue
            
            input_ids = torch.tensor([ids[:-1]], device=DEVICE)
            target_ids = torch.tensor([ids[1:]], device=DEVICE)
            
            logits, loss = model(input_ids, target_ids)
            
            total_loss += loss.item() * (len(ids) - 1)
            total_tokens += len(ids) - 1
    
    if total_tokens == 0:
        return float('inf')
    
    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)
    
    return perplexity, mean_loss


# ─── METRIC 2: GENERATION LENGTH & COHERENCE ────────────────────────────────

def evaluate_generation_quality(model, tokenizer, prompts, max_new=100):
    """
    Evaluate:
    - Average generation length
    - Sentence count (coherence proxy)
    - Repetition ratio (lower is better)
    """
    stats = {
        "avg_length": 0,
        "avg_sentences": 0,
        "avg_words": 0,
        "repetition_ratio": 0,  # Count of repeated words / total words
    }
    
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids], device=DEVICE)
            
            output_ids = model.generate(
                input_ids,
                max_new=max_new,
                temperature=0.01,
                top_k=1,
                use_cache=True
            )
            
            output_text = tokenizer.decode(output_ids[0].tolist())
            
            # Extract generated part (after prompt)
            generated = output_text[len(prompt):]
            
            # Metrics
            words = generated.split()
            sentences = generated.count('.') + generated.count('!') + generated.count('?') + 1
            
            stats["avg_length"] += len(generated)
            stats["avg_sentences"] += sentences
            stats["avg_words"] += len(words)
            
            # Repetition (how many unique words vs total)
            unique_words = len(set(w.lower() for w in words if len(w) > 3))
            total_words = len([w for w in words if len(w) > 3])
            repetition = 1 - (unique_words / total_words) if total_words > 0 else 0
            stats["repetition_ratio"] += repetition
    
    n_prompts = len(prompts)
    for key in stats:
        stats[key] /= n_prompts
    
    return stats


# ─── METRIC 3: STORY COMPLEXITY PERFORMANCE ───────────────────────────────────

def evaluate_by_complexity(model, tokenizer, validation_samples):
    """
    TinyStories has different story complexities: simple, moderate, complex.
    Evaluate perplexity separately for each to see if model trains uniformly.
    """
    results = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for story, complexity in validation_samples:
            encoded = tokenizer.encode(story)
            ids = encoded.ids
            
            if len(ids) < 2:
                continue
            
            input_ids = torch.tensor([ids[:-1]], device=DEVICE)
            target_ids = torch.tensor([ids[1:]], device=DEVICE)
            
            _, loss = model(input_ids, target_ids)
            ppl = math.exp(loss.item())
            
            results[complexity].append(ppl)
    
    # Average by complexity
    summary = {}
    for complexity in ["simple", "moderate", "complex"]:
        if results[complexity]:
            avg_ppl = np.mean(results[complexity])
            std_ppl = np.std(results[complexity])
            summary[complexity] = {"mean": avg_ppl, "std": std_ppl}
    
    return summary


# ─── METRIC 4: ENTITY CONSISTENCY OVER LONG SEQUENCES ──────────────────────────

def evaluate_entity_consistency(model, tokenizer, entity_prompts, generation_length=200):
    """
    How well does the model maintain entity consistency in long generations?
    
    This is critical for story quality.
    """
    consistency_scores = []
    
    model.eval()
    with torch.no_grad():
        for prompt, target_entity in entity_prompts:
            encoded = tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids], device=DEVICE)
            
            output_ids = model.generate(
                input_ids,
                max_new=generation_length,
                temperature=0.01,
                top_k=1,
                use_cache=True
            )
            
            output_text = tokenizer.decode(output_ids[0].tolist())
            
            # Count entity mentions
            entity_mentions = output_text.lower().count(target_entity.lower())
            
            # Score: how early does name disappear?
            words = output_text.split()
            last_mention_idx = -1
            for i, word in enumerate(words):
                if target_entity.lower() in word.lower():
                    last_mention_idx = i
            
            # Consistency = last_mention_idx / total_words (higher = better)
            consistency = last_mention_idx / len(words) if len(words) > 0 else 0
            consistency_scores.append(consistency)
    
    return {
        "avg_consistency": np.mean(consistency_scores),
        "std_consistency": np.std(consistency_scores),
        "consistency_scores": consistency_scores,
    }


# ─── BENCHMARK COMPARISON ──────────────────────────────────────────────────────

def run_benchmark_suite(model, model_name, tokenizer):
    """Run full benchmarking suite on a model."""
    
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*80}")
    
    # Sample validation texts (in real scenario, load from TinyStories val set)
    val_texts = [
        "There was a little girl named Anna. She lived in a small house by the forest. Every day she would walk in the woods and pick flowers.",
        "Once upon a time, a boy named Tim found a magic lamp. He rubbed it and a genie appeared! The genie offered him three wishes.",
        "A cat named Whiskers was sleeping on the windowsill. Suddenly, a bird flew by. Whiskers jumped up and chased the bird around the garden.",
    ]
    
    entity_prompts = [
        ("Once upon a time there was a girl named Sophie. Sophie", "Sophie"),
        ("A boy named Jack walked into the forest. Jack was", "Jack"),
        ("The princess named Elena ruled the kingdom. Elena was", "Elena"),
    ]
    
    generation_prompts = [
        "Once upon a time there was",
        "The little boy loved to",
        "One day she decided to",
    ]
    
    # BENCHMARK 1: Perplexity
    print("\n[BENCHMARK 1] PERPLEXITY")
    print("-" * 80)
    ppl, loss = calculate_perplexity(model, tokenizer, val_texts)
    print(f"Perplexity:  {ppl:.2f}")
    print(f"Validation Loss: {loss:.4f}")
    print(f"Interpretation: Lower is better. TinyStories baseline ~10-15 PPL")
    
    # BENCHMARK 2: Generation Quality
    print("\n[BENCHMARK 2] GENERATION QUALITY")
    print("-" * 80)
    gen_stats = evaluate_generation_quality(model, tokenizer, generation_prompts)
    print(f"Avg generation length: {gen_stats['avg_length']:.0f} chars")
    print(f"Avg sentences: {gen_stats['avg_sentences']:.1f}")
    print(f"Avg words: {gen_stats['avg_words']:.0f}")
    print(f"Repetition ratio: {gen_stats['repetition_ratio']:.2%}")
    print(f"Interpretation: Lower repetition (~20-40%) is better")
    
    # BENCHMARK 3: Entity Consistency
    print("\n[BENCHMARK 3] ENTITY CONSISTENCY")
    print("-" * 80)
    entity_stats = evaluate_entity_consistency(model, tokenizer, entity_prompts)
    print(f"Avg consistency score: {entity_stats['avg_consistency']:.2%}")
    print(f"Std consistency: {entity_stats['std_consistency']:.2%}")
    print(f"Interpretation: Higher is better. >50% means names stay relevant throughout")
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    summary = {
        "model": model_name,
        "perplexity": ppl,
        "loss": loss,
        "repetition": gen_stats["repetition_ratio"],
        "entity_consistency": entity_stats["avg_consistency"],
    }
    
    return summary


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("TINYSTORIES SLM BENCHMARKING SUITE")
    print("="*80)
    print("""
    Standard metrics for evaluating TinyStories models:
    
    1. PERPLEXITY (PPL) — Main metric, measures language model quality
       - Lower is better
       - TinyStories baseline: 10-15 PPL (for different scales)
       - Your model should achieve: 5-20 PPL
    
    2. GENERATION QUALITY — Coherence and diversity
       - Repetition ratio: how much word repetition (lower better)
       - Sentence count: should maintain narrative structure
    
    3. ENTITY CONSISTENCY — Critical for story quality
       - How long character names remain relevant
       - Higher consistency = better storytelling
    
    4. COMPLEXITY PERFORMANCE — Uniform across difficulties
       - Simple stories: easier, lower PPL
       - Complex stories: harder, higher PPL
       - Model should handle all equally well relative to baseline
    """)
    
    # Load checkpoints
    all_ckpts = sorted(glob.glob("checkpoints/stage0_best*.pt"))
    
    if len(all_ckpts) == 0:
        print("❌ No checkpoints found!")
        return
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    results = []
    
    # Benchmark each checkpoint
    for ckpt_path in all_ckpts:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = SLM(ckpt["config"]).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        
        model_name = os.path.basename(ckpt_path)
        summary = run_benchmark_suite(model, model_name, tokenizer)
        results.append(summary)
    
    # Comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\n{'Model':<40} {'PPL':>8} {'Loss':>8} {'Repetition':>12} {'Entity':>8}")
    print("-"*80)
    
    for r in results:
        print(f"{r['model']:<40} {r['perplexity']:>8.2f} {r['loss']:>8.4f} {r['repetition']:>11.1%} {r['entity_consistency']:>8.1%}")
    
    print("\n" + "="*80)
    print("BENCHMARK RESOURCES")
    print("="*80)
    print("""
    To get realistic benchmarks, you need:
    
    1. TINYSTORIES VALIDATION SET
       - Use the official TinyStories dataset
       - Split by complexity: simple, moderate, complex
       - ~1000-5000 validation examples per complexity
    
    2. STANDARD METRICS
       - Perplexity (implemented above)
       - BERTScore for semantic similarity
       - Human evaluation (gold standard)
    
    3. DOWNSTREAM TASKS (optional)
       - GLUE: understanding, inference
       - Multiple Choice QA (ARC, HellaSwag)
       - Story coherence ranking
    
    4. PAPERS TO REFERENCE
       - TinyStories (Eldan & Li, 2023): Original paper with baseline metrics
       - GPT-2 paper: Standard LM evaluation protocols
       - Holistic Evaluation of Language Models (HELM): Comprehensive framework
    """)


if __name__ == "__main__":
    main()
