import torch
from tokenizers import Tokenizer
from model import *

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[generate] Loading tokenizer...")
tokenizer = Tokenizer.from_file("tokenizers/tokenizer_corpus.json")
vocab_size = tokenizer.get_vocab_size()

ckpt_path = 'checkpoints/stage0_best_1stfull.pt'
print(f"[generate] Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# Rebuild Model Config
cfg = ckpt["config"]
model = SLM(cfg).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("Tokenizer vocab size:", tokenizer.get_vocab_size())
for name, param in model.named_parameters():
    print(name, param.shape)

# 1. Dynamically retrieve your tokenizer's EOS ID (e.g., usually 2, 3, or 50256)
# If your tokenizer uses a different special token string, replace "<|endoftext|>"
eos_id = tokenizer.token_to_id("<|endoftext|>") 
if eos_id is None:
    eos_id = tokenizer.token_to_id("[EOS]") # Fallback for other standard formats

def generate_text(prompt, max_new_tokens=250, temperature=0.7):
    tokens = tokenizer.encode(prompt).ids
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        out_tokens = model.generate(x, max_new=max_new_tokens, temperature=temperature, top_k=50)[0].tolist()
    
    # 2. Slice the token list exactly where the EOS token occurs
    if eos_id in out_tokens:
        out_tokens = out_tokens[:out_tokens.index(eos_id)]
    
    # 3. Decode the truncated list
    text = tokenizer.decode(out_tokens)
    return text.replace('\n\n', ' ').strip()

prompts = [
    "Once upon a time, a brave tiny mouse",
    "Lily was crying because she lost her",
    "The big friendly giant wanted to",
    "Timmy saw a completely dark cave, so he",
    "Suddenly, a magic fairy appeared and said,"
]

print("\n" + "="*50)
print("  GENERATING RESPONSES (TinyStories)")
print("="*50)

for p in prompts:
    print(f"\nPrompt:  \"{p}\"")
    output = generate_text(p)
    print(f"Output: \"{output}\"")

print("\n" + "="*50)
print("Done.")
