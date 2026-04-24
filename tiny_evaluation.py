import json
from tqdm.auto import tqdm

# Import the generation engine from the generate.py file
from generate import generate_text

def calculate_diversity_score(text: str) -> tuple[float, float]:
    """
    Computes lexical diversity via unigram and bigram uniqueness ratios.
    A low score strongly correlates with autoregressive repetitive loops
    (e.g., "the cat ran the cat ran").
    """
    tokens = text.lower().split()
    if not tokens:
        return 0.0, 0.0
    
    unigrams = tokens
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    
    unique_unigrams = len(set(unigrams)) / len(unigrams)
    unique_bigrams = len(set(bigrams)) / max(1, len(bigrams))
    
    return unique_unigrams, unique_bigrams

def run_benchmarks(num_samples: int = 50, output_file: str = "generation_metrics.json"):
    """
    Executes a suite of benchmark prompts against the imported generation pipeline
    and computes aggregate statistical coherence metrics.
    """
    # Standardised TinyStories evaluation prompts
    base_prompts = [
        "Once upon a time, there was a little girl named Lily. She loved to",
        "A big brown bear walked into the forest and saw",
        "Tom was very sad because he lost his favorite toy. He decided to",
        "The small dog barked at the loud truck. The truck driver",
        "One sunny day, a magic frog jumped out of the pond and said",
    ]
    
    # Expand the prompt list to meet the requested sample size
    test_prompts = base_prompts * (num_samples // len(base_prompts) + 1)
    test_prompts = test_prompts[:num_samples]

    results = []
    total_unigram_div = 0.0
    total_bigram_div = 0.0
    
    print(f"\n[evaluate] Running {len(test_prompts)} generation benchmarks via generate.py...")
    
    for prompt in tqdm(test_prompts, desc="Evaluating"):
        # Execute the imported generation function
        generated_text = generate_text(prompt, max_new_tokens=150, temperature=0.8)
        
        # Isolate the model's continuation by stripping the prompt
        if generated_text.startswith(prompt):
            continuation = generated_text[len(prompt):].strip()
        else:
            continuation = generated_text.strip()
        
        # Compute structural diversity
        uni_div, bi_div = calculate_diversity_score(continuation)
        total_unigram_div += uni_div
        total_bigram_div += bi_div
        
        # Detect proper sequence termination
        hit_eos = "<|endoftext|>" in generated_text or "[EOS]" in generated_text
        
        results.append({
            "prompt": prompt,
            "continuation": continuation,
            "full_text": generated_text,
            "unigram_diversity": round(uni_div, 3),
            "bigram_diversity": round(bi_div, 3),
            "hit_eos": hit_eos
        })

    # Compute macro-averages
    avg_uni = total_unigram_div / num_samples
    avg_bi = total_bigram_div / num_samples
    eos_completion_rate = sum(1 for r in results if r["hit_eos"]) / num_samples

    print("\n" + "="*50)
    print("BENCHMARK AGGREGATION RESULTS")
    print("="*50)
    print(f"Average Unigram Diversity: {avg_uni:.3f}  (Threshold: > 0.45)")
    print(f"Average Bigram Diversity:  {avg_bi:.3f}  (Threshold: > 0.75)")
    print(f"EOS Completion Rate:       {eos_completion_rate*100:.1f}%")
    print("="*50)

    # Serialize results to disk for manual inspection or secondary LLM grading
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate] Saved full generations and metrics to {output_file}")

if __name__ == "__main__":
    # Execute benchmark suite
    run_benchmarks(num_samples=50)