import torch
from tokenizers import Tokenizer
from src.model import SLM, SLMConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[generate] Loading tokenizer...")
tokenizer = Tokenizer.from_file("tokenizers/tokenizer_corpus.json")
vocab_size = tokenizer.get_vocab_size()

ckpt_path = "checkpoints/stage0_curriculum_adaptive_best.pt"
print(f"[generate] Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# Rebuild Model Config
cfg = ckpt["config"]
model = SLM(cfg).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

def generate_text(prompt, max_new_tokens=150, temperature=0.7):
    tokens = tokenizer.encode(prompt).ids
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        out_tokens = model.generate(x, max_new=max_new_tokens, temperature=temperature, top_k=50)
    
    text = tokenizer.decode(out_tokens[0].tolist())
    # Clean up any excessive newlines
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
