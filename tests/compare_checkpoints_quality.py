"""
compare_checkpoints_quality.py
Test generation quality across different checkpoints to verify loss vs quality correlation
"""
import os
import torch
from tokenizers import Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER_PATH = "tokenizers/tokenizer_corpus.json"

TEST_PROMPTS = [
    "Lions are large carnivorous",
    "The Earth is the third planet from",
]

CHECKPOINTS = [
    # (name, path, step, val_loss)
    ("Stage0 Trial (13K steps, loss=1.68)", "checkpoints/stage0_best _1st_trial.pt", 13000, 1.6842),
    ("Stage0 Final (6.5K steps, loss=5.97)", "checkpoints/stage0_best.pt", 6500, 5.9717),
    ("Stage1 Final (12K steps, loss=3.56)", "checkpoints/stage1_best.pt", 12000, 3.5629),
]

def test_checkpoint(name, path, step, loss):
    """Test a single checkpoint"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"  Path: {path}")
    print(f"  Step: {step}, Validation Loss: {loss:.4f}")
    print(f"{'='*80}")
    
    if not os.path.exists(path):
        print(f"[SKIP] Not found")
        return
    
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        from model import SLM
        
        model = SLM(ckpt["config"]).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        
        for prompt in TEST_PROMPTS:
            print(f"\n[Prompt] {repr(prompt)}")
            print("-" * 80)
            
            input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new=80,
                    temperature=1.0,
                    top_k=200,
                    use_cache=True
                )
            
            text = tokenizer.decode(output[0].tolist())
            print(text)
            
            # Analyze quality
            tokens = tokenizer.encode(text).ids
            unique_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
            print(f"\n[METRICS] Total tokens: {len(tokens)}, Unique: {len(set(tokens))}, Diversity: {unique_ratio:.2%}")
            
    except Exception as e:
        print(f"[ERROR] {e}")


def main():
    print("LOSS vs GENERATION QUALITY VERIFICATION")
    print()
    
    for name, path, step, loss in CHECKPOINTS:
        test_checkpoint(name, path, step, loss)
    
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    print()
    print("If better loss = better generation:")
    print("  Stage0 Trial (loss 1.68) should generate BEST text")
    print("  Stage1 Final (loss 3.56) should generate WORST text")
    print()
    print("If loss doesn't correlate with generation quality:")
    print("  All might generate similarly poor text (suggests different root cause)")
    print()


if __name__ == "__main__":
    main()
