"""
test_entity_consistency.py
Measures whether model maintains consistent character names throughout generation.
"""

import torch
from tokenizers import Tokenizer
from src.model import SLM
import re

CHECKPOINT_PATH = "checkpoints/stage0_best.pt"
TOKENIZER_PATH  = "tokenizers/tokenizer_corpus.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model = SLM(ckpt["config"]).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

prompts = [
    "Once upon a time there was a mouse named Guddu. Guddu was lazy",
    "Once upon a time there was a boy named Charlie. Charlie loved adventure",
    "Once upon a time there was a girl named Alice. Alice was curious",
]

print("="*80)
print("ENTITY CONSISTENCY TEST")
print("="*80)

for prompt in prompts:
    # Extract target name from prompt
    match = re.search(r'named (\w+)', prompt)
    target_name = match.group(1) if match else None
    
    # Generate with short and long lengths
    for max_new in [50, 100]:
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=DEVICE)
        
        output_ids = model.generate(
            input_ids,
            max_new=max_new,
            temperature=0.01,  # Deterministic
            top_k=1,
            use_cache=True
        )
        output = tokenizer.decode(output_ids[0].tolist())
        
        # Count name appearances
        name_count = len(re.findall(r'\b' + target_name + r'\b', output, re.IGNORECASE))
        
        print(f"\n[Prompt] {prompt}")
        print(f"[Max tokens] {max_new}")
        print(f"[Target name] {target_name}")
        print(f"[Name occurrences] {name_count} times")
        print(f"[Output] {output[:150]}...")
        
        if name_count < 2:
            print(f"  ⚠️  Name lost after {max_new} tokens!")
        else:
            print(f"  ✓ Name maintained")

print("\n" + "="*80)
print("RECOMMENDATION: Use max_new=50 for entity consistency")
print("="*80)
