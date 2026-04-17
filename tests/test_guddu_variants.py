"""Quick test: Does 'this' vs 'a' affect name preservation?"""
import torch
from tokenizers import Tokenizer
from src.model import SLM

CHECKPOINT_PATH = "checkpoints/stage0_best.pt"
TOKENIZER_PATH  = "tokenizers/tokenizer_corpus.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model = SLM(ckpt["config"]).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Test both versions
prompts = [
    "Once upon a time there was this mouse named Guddu",  # Original
    "Once upon a time there was a mouse named Guddu",     # Test version
]

print("Testing: Does 'this' vs 'a' affect name preservation?\n")

for i, prompt in enumerate(prompts, 1):
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device=DEVICE)
    
    output_ids = model.generate(input_ids, max_new=80, temperature=0.7, top_k=50, use_cache=True)
    output = tokenizer.decode(output_ids[0].tolist())
    
    # Count how many times Guddu appears
    guddu_count = output.lower().count('guddu')
    
    print(f"Prompt {i}: '{prompt}'")
    print(f"  Tokens: {encoded.ids[-5:]}")
    print(f"  Guddu appearances: {guddu_count}/2+")
    print(f"  Output: {output[:120]}...\n")
