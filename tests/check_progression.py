"""
test_checkpoint_progression.py
Test generation quality at different training steps
"""
import torch
from tokenizers import Tokenizer
from src.model import SLM
import os

CHECKPOINT_PATH = "checkpoints/stage1_best.pt"
TOKENIZER_PATH = "tokenizers/tokenizer_corpus.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPT = 'The lion is the king of the'

def test_checkpoint(path, label):
    """Test a single checkpoint"""
    if not os.path.exists(path):
        print(f"[SKIP] {label}: {path} not found")
        return None
    
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = SLM(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    input_ids = torch.tensor([tokenizer.encode(TEST_PROMPT).ids], device=DEVICE)
    
    with torch.no_grad():
        output = model.generate(input_ids, max_new=80, temperature=1.0, top_k=200, use_cache=True)
    
    text = tokenizer.decode(output[0].tolist())
    tokens = tokenizer.encode(text).ids
    unique_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
    
    return {
        'label': label,
        'step': ckpt.get('step'),
        'loss': ckpt.get('best_val_loss'),
        'text': text,
        'diversity': unique_ratio
    }

print("="*80)
print("STAGE 1 GENERATION QUALITY PROGRESSION")
print("="*80)
print()

# Test current checkpoint
result = test_checkpoint("checkpoints/stage1_best.pt", "CURRENT (Step 4500)")

if result:
    print(f"[{result['label']}]")
    print(f"  Step: {result['step']}, Loss: {result['loss']:.4f}, Diversity: {result['diversity']:.1%}")
    print()
    print(f"  Generated text:")
    print(f"  {result['text']}")
    print()
    
    # Quality assessment
    if result['diversity'] > 0.6:
        print("  ✓ GOOD: Diverse output (>60% unique tokens)")
    else:
        print("  ⚠️  POOR: Repetitive output (<60% unique tokens)")
        
    # Check for specific repetition patterns
    if "V V V" in result['text'] or " V " in result['text']:
        print("  ✗ Found 'V' repetition pattern")
    elif result['text'].count('the') > 8:
        print(f"  ⚠️  High 'the' repetition ({result['text'].count('the')} times)")
    else:
        print("  ✓ No obvious repetition patterns")

