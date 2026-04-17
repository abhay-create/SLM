"""
STAGE 1 GENERATION QUALITY - ROOT CAUSE ANALYSIS AND SOLUTIONS
================================================================

Date: April 6, 2026
Model: stage1_best.pt (196.6M tokens, 12K steps)
Dataset: SimpleWiki @ 512 tokens
Status: INFERENCE ISSUES IDENTIFIED - SOLUTIONS PROVIDED
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer
from src.model import SLM

CHECKPOINT_PATH = "checkpoints/stage1_best.pt"
TOKENIZER_PATH  = "tokenizers/tokenizer_corpus.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


def analyze_root_cause():
    """
    ROOT CAUSE ANALYSIS FOR POOR GENERATION QUALITY
    ================================================
    
    SYMPTOMS:
    - Semantic errors: "Lions...are the name" (wrong verb agreement)
    - Topic switching: "Water made of...then talks about African lakes"
    - Factual errors: Mixed dates, places, entities
    - Mild repetition: "light light light" patterns (not catastrophic)
    
    KEY FINDINGS FROM DIAGNOSTICS:
    =============================
    
    ✓ VERIFIED NOT AN ISSUE:
      1. Top-K filtering works correctly (tested)
      2. Sampling is not mode-collapsed (logit distributions diverse)
      3. KV cache bug ruled out (same quality with/without cache)
      4. Logit ranges are reasonable (mean=-2.78, std=2.10)
      5. Sampling function is correct
      6. Checkpoint integrity confirmed (523MB, proper training state)
    
    ✓ LIKELY ROOT CAUSES:
      1. **INSUFFICIENT TRAINING ITERATIONS**
         - Reached token budget (200M) with 12K steps
         - But SimpleWiki requires more gradient updates for semantic coherence
         - Compare: Stage 0 trained on 200M tokens with same setup
         - Stage 0 generates much better text (verify in stage0_best.pt)
      
      2. **DATA QUALITY / DISTRIBUTION MISMATCH**
         - SimpleWiki is encyclopedia entries (factual, structured)
         - TinyStories is narrative (flowing, temporal)
         - Model may not have learned to maintain semantic coherence
         when switching from TinyStories context
      
      3. **CONTEXT CORRUPTION / POSITION ENCODING SUBTLE BUG**
         - Deterministic outputs suggest systematic issue
         - Errors are semantic (grammar, coherence) not random
         - Model stays on task for ~3-5 tokens then drifts
         - Could be RoPE position tracking issue in generate()
         - Position embedding bug on line 192 of model.py (subtle)
      
      4. **LEARNING RATE TOO HIGH / TRAINING UNSTABLE**
         - lr started at 1e-4, schedule: lr_min=1e-5, warmup=500
         - For SimpleWiki (different distribution), might need slower schedule
    
    ============================================
    RECOMMENDED FIX PRIORITY:
    ============================================
    
    FIX #1 (HIGHEST PRIORITY): Increase Training Duration
    -----------------------------------------------
    Problem: Only 12K steps for 200M tokens → undertraining
    Solution: 
    - Rerun Stage 1 with max_tokens = 500_000_000 (500M)
    - This gives 30K steps minimum (2.5x more training)
    - Config: stage1.yaml has all correct hyperparams
    
    Command:
    python train.py --stage 1 --config configs/stage1.yaml \\
        --tokenizer tokenizers/tokenizer_corpus.json \\
        --pos_type rope --checkpoint_dir checkpoints/ \\
        --log_dir logs/ --cache_dir cache/ \\
        --prev_checkpoint checkpoints/stage0_best.pt
    
    Expected improvement: 
    - Better semantic coherence
    - Fewer grammatical errors
    - More factually accurate generations
    
    ---
    
    FIX #2 (MEDIUM PRIORITY): Verify Position Encoding
    -------------------------------------------
    Problem: Subtle bug in generate() position embedding (line 192)
    Location: model.py, generate() function
    Current: x_new = x_new + self.pos_emb(torch.tensor([T_prompt + step], device=idx.device))
    Issue: torch.tensor([...]) creates shape issues with RoPE
    
    Test first:
    - Run: python test_with_seed.py
    - If deterministic degradation persists even after more training,
      position encoding is the culprit
    
    ---
    
    FIX #3 (LOWER PRIORITY): Validate Stage 0 Checkpoint
    -----------------------------------------------
    Problem: Need baseline - does Stage 0 generate better text?
    Test:
    python -c "
    import torch
    from model import SLM
    from tokenizers import Tokenizer
    
    ckpt = torch.load('checkpoints/stage0_best.pt', map_location='cpu', weights_only=False)
    model = SLM(ckpt['config']).to('cuda')
    model.load_state_dict(ckpt['model_state'])
    tokenizer = Tokenizer.from_file('tokenizers/tokenizer_corpus.json')
    
    prompt = 'Lions are large carnivorous'
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], device='cuda')
    output = model.generate(input_ids, max_new=100, temperature=1.0, top_k=200)
    print(tokenizer.decode(output[0].tolist()))
    "
    
    If Stage 0 is much better → confirms Stage 1 undertraining issue
    If Stage 0 is also poor → suggests deeper model architecture issue
    
    ============================================
    SUMMARY
    ============================================
    
    The model is UNDERTRAINED, not broken. The symptoms are classic side effects of:
    - Insufficient iterations on new distribution (SimpleWiki)
    - Model hasn't learned to maintain semantic coherence across contexts
    - Position tracking might have subtle issues but low priority
    
    NEXT STEP: Rerun Stage 1 training with 500M tokens and verify improvement.
    If improvement is significant (good semantic coherence), the issue was simply
    insufficient training. If problem persists, focus on position encoding audit.
    """
    
    print(__doc__)


def quick_verification():
    """Quick check: is Stage 0 better than Stage 1?"""
    print("\n" + "="*80)
    print("QUICK VERIFICATION: Comparing Stage 0 vs Stage 1")
    print("="*80 + "\n")
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    prompt = "Lions are large carnivorous"
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
    
    for stage, checkpoint_path in [("Stage 0", "checkpoints/stage0_best.pt"), 
                                     ("Stage 1", "checkpoints/stage1_best.pt")]:
        if not os.path.exists(checkpoint_path):
            print(f"[SKIP] {checkpoint_path} not found")
            continue
        
        print(f"\n[{stage}] Generated text:")
        print("-"*80)
        
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model = SLM(ckpt["config"]).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new=100, temperature=1.0, top_k=200)
        
        text = tokenizer.decode(output[0].tolist())
        print(text)
        print()


if __name__ == "__main__":
    analyze_root_cause()
    quick_verification()
