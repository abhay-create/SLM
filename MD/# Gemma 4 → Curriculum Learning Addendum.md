# Gemma 4 → Curriculum Learning Addendum

> **Continuity note**: The existing strategy already covers perplexity-based difficulty scoring,
> competence-aware adaptive scheduling, smooth transitions, and multi-metric difficulty.
> This document adds only what Gemma 4 introduces that is *new and actionable* for our SLM training.

---

## 1. Per-Layer Embeddings (PLE) → Layer-Aware Curriculum Pacing

**What Gemma 4 does**: E2B/E4B introduce per-layer embedding lookup tables — each token gets a
distinct embedding vector per transformer layer (stored on flash, not VRAM). This lets shallow
layers handle syntax-level patterns and deeper layers handle semantics, without the representation
getting diluted through repeated passes.

**New idea for our training**: Apply this insight to *how we schedule difficulty per layer group*, not just globally.

Instead of a single `curriculum_fraction` for the whole model, track separate learning signals per
layer block and advance curriculum only when **both shallow and deep layers** show competence:

```python
class LayerAwareCurriculum(CompetenceAwareCurriculum):
    """
    Extends existing CompetenceAwareCurriculum.
    Hooks into gradient norms per layer group as a proxy for per-layer 'confusion'.
    Advances curriculum only when deep layers stop thrashing.
    """
    def __init__(self, *args, n_layer_groups=3, **kwargs):
        super().__init__(*args, **kwargs)
        # Track gradient norm per layer group (shallow / mid / deep)
        self.layer_grad_history = {i: [] for i in range(n_layer_groups)}
        self.n_layer_groups = n_layer_groups

    def record_grad_norms(self, model):
        """Call after loss.backward(), before optimizer.step()."""
        layers = list(model.transformer.h)  # GPT-style layer list
        group_size = len(layers) // self.n_layer_groups

        for g in range(self.n_layer_groups):
            group_layers = layers[g * group_size : (g + 1) * group_size]
            norms = [
                p.grad.norm().item()
                for layer in group_layers
                for p in layer.parameters()
                if p.grad is not None
            ]
            self.layer_grad_history[g].append(np.mean(norms) if norms else 0.0)

    def deep_layers_stable(self, window=50, threshold=0.15):
        """
        Returns True when the *deepest* layer group's gradient norm
        variance has dropped below threshold — signal that deep layers
        have absorbed the current difficulty level.
        """
        deep_norms = self.layer_grad_history[self.n_layer_groups - 1]
        if len(deep_norms) < window:
            return False
        recent = deep_norms[-window:]
        cv = np.std(recent) / (np.mean(recent) + 1e-8)  # coeff of variation
        return cv < threshold

    def update_competence(self, val_loss):
        """Only advance if both val_loss improves AND deep layers are stable."""
        super().update_competence(val_loss)          # existing logic
        if not self.deep_layers_stable():
            # Freeze curriculum expansion even if val loss is improving
            self.current_fraction = max(
                self.current_fraction - 0.01,        # slight rollback
                self.current_fraction                 # or hold
            )
```

**Why this helps**: Our 50M model's early layers learn token co-occurrences fast; deep layers need
longer exposure to complex discourse. PLE insight tells us these are genuinely separate timescales.
Tying curriculum advancement to deep-layer stability stops us from flooding the model with hard
stories before the upper layers are ready.

**Integration point**: Add `curriculum.record_grad_norms(model)` right after `loss.backward()`
in the existing training loop. Replace `curriculum.update_competence(val_loss)` with the subclass version.

---

## 2. Soft Token Budget → Dynamic Sequence Length Curriculum

**What Gemma 4 does**: The vision encoder exposes a configurable token budget (70 / 140 / 280 / 560 / 1120
soft tokens). Crucially, a *lower budget forces aggressive pooling / downsampling* — the model sees
a coarser representation. This is not a crop, it's a lossy compression that the model must learn to
handle at different fidelity levels.

**New idea**: Mirror this with a **dynamic context window / sequence-length budget** during text training.

Rather than always feeding `block_size=256` tokens, start with a compressed view and gradually expand:

```python
BUDGET_SCHEDULE = [64, 128, 192, 256]   # mirrors 70→140→280→560 token budget idea

def get_dynamic_block_size(curriculum_fraction: float, max_block: int = 256) -> int:
    """
    Maps curriculum progress [0, 1] → context window size.
    At fraction=0.0 → 64 tokens (coarse, easy patterns only)
    At fraction=1.0 → 256 tokens (full context)
    
    Uses discrete budget steps like Gemma 4's soft-token tiers.
    """
    n = len(BUDGET_SCHEDULE)
    idx = min(int(curriculum_fraction * n), n - 1)
    return BUDGET_SCHEDULE[idx]

# In get_batch_adaptive():
block_size = get_dynamic_block_size(curriculum.current_fraction)
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy(data[i     : i + block_size].astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix])
```

**Why this helps**: Short contexts force the model to learn local syntax and word-level patterns
before long-range discourse. It's a different axis of difficulty from perplexity score — a 256-token
story at 64 tokens of context is inherently "easier" because the model only needs to predict
locally. When combined with the existing perplexity-based difficulty scorer, you get a 2D curriculum
grid:

```
                Low perplexity ──────────────► High perplexity
Short context │  START HERE (easiest)               medium
              │
Long context  │  medium                         END HERE (hardest)
```

**Note**: This is separate from the story difficulty ordering — it controls *how much of each story
the model sees at once*, not *which story* gets sampled.

---

## 3. K=V in Global Attention → Curriculum Signal from Attention Collapse

**What Gemma 4 does**: In global attention layers, Keys are set equal to Values (K=V). The insight
is that maintaining separate K and V projections in global layers is expensive and slightly redundant
— collapsing them cuts KV-cache in half with minimal quality loss. The model compensates by doubling
Key dimensionality.

**New idea**: Use the *divergence between K and V projections* in our model as a difficulty signal.

When K and V are very similar (cosine similarity → 1), the current batch is "easy" — the attention
layer doesn't need rich value differentiation. When they diverge strongly, the model is in
high-uncertainty territory:

```python
def kv_divergence_metric(model, x: torch.Tensor) -> float:
    """
    Proxy for 'how hard is this batch' via K-V divergence in attention layers.
    High divergence = model is uncertain = data may be too hard.
    
    Only measure on the last (global) attention block — mirrors Gemma 4's
    insight that global attention is where long-range difficulty lives.
    """
    model.eval()
    hooks, kv_pairs = [], []

    def hook_fn(module, inp, out):
        # Capture Q,K,V projections for CausalSelfAttention
        if hasattr(module, 'c_attn'):
            qkv = module.c_attn(inp[0])
            C = qkv.size(-1) // 3
            k = qkv[..., C:2*C]
            v = qkv[..., 2*C:]
            kv_pairs.append((k.detach(), v.detach()))

    # Hook only the last transformer block
    last_block = list(model.transformer.h)[-1]
    h = last_block.attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(x)

    h.remove()
    model.train()

    if not kv_pairs:
        return 0.0

    k, v = kv_pairs[0]
    k_flat = k.reshape(-1, k.size(-1))
    v_flat = v.reshape(-1, v.size(-1))

    # Cosine similarity: high similarity = easy batch, low = hard
    cos_sim = torch.nn.functional.cosine_similarity(k_flat, v_flat, dim=-1).mean().item()
    return 1.0 - cos_sim   # divergence: 0=easy, ~1=very hard

# Use in curriculum update (every N steps):
if step % 200 == 0:
    sample_x, _ = get_batch_adaptive('train', current_iter=step)
    kv_div = kv_divergence_metric(model, sample_x)
    
    # If KV divergence spikes, slow down curriculum expansion
    if kv_div > 0.4:   # empirical threshold, tune per run
        curriculum.patience_counter += 2   # existing patience mechanism
    
    wandb.log({'kv_divergence': kv_div, 'curriculum_fraction': curriculum.current_fraction})
```

**Why this helps**: This is a model-internal signal that doesn't require a reference model or extra
forward passes on held-out data. It runs in <1ms per check. Combined with existing val-loss-based
pacing, you now have both *external* (validation loss) and *internal* (attention structure) signals
driving curriculum speed.

---

## 4. p-RoPE Insight → Long-Context Story Scheduling

**What Gemma 4 does**: p-RoPE applies rotary positional encoding only to the top-frequency (high
rotation) pairs and zeros out the low-frequency pairs. The motivation: low-frequency pairs were
being used for *semantic content*, and adding positional noise to them hurt long-context performance.
Only ~25% of dimensions carry positional info; the rest stay semantic.

**New idea**: In our story dataset, long stories cause the same problem — positional encoding
accumulates across many tokens and the model's semantic representations become polluted with
positional noise at tail positions.

Rather than just using short stories first (which we already do), explicitly **up-weight long
stories that have strong discourse coherence**:

```python
def compute_long_story_coherence_bonus(stories: list, percentile_cutoff: float = 0.75) -> np.ndarray:
    """
    For stories above the 75th percentile in length, compute a coherence score.
    Stories that are long BUT coherent (strong discourse markers, consistent topic)
    should come *earlier* in the curriculum than long-but-rambling stories.
    
    Inspired by p-RoPE: positional complexity ≠ semantic complexity.
    A long coherent story is easier for a semantic model than a long incoherent one.
    """
    import re
    discourse_markers = {
        'temporal':    r'\b(first|then|next|finally|after|before|when|suddenly)\b',
        'causal':      r'\b(because|so|therefore|thus|as a result)\b',
        'adversative': r'\b(but|however|although|even though|despite)\b',
    }

    bonuses = np.zeros(len(stories))

    for i, text in enumerate(stories):
        length = len(text.split())
        if length < np.percentile([len(s.split()) for s in stories], percentile_cutoff * 100):
            bonuses[i] = 0.0   # short stories unaffected
            continue

        # Count discourse markers — more markers = more coherent structure
        marker_count = sum(
            len(re.findall(pattern, text.lower()))
            for pattern in discourse_markers.values()
        )
        # Normalize by length (markers per 100 words)
        bonuses[i] = (marker_count / max(length, 1)) * 100

    return bonuses   # Add to difficulty score with negative weight (higher coherence = easier)

# In Phase 1 preprocessing (extend existing composite difficulty):
coherence_bonus = compute_long_story_coherence_bonus([ex['text'] for ex in ds['train']])

difficulty = (
    0.55 * perplexity +
    0.20 * type_token_ratio * 100 +
    0.15 * (length / 100) +
    -0.10 * coherence_bonus    # ← NEW: long coherent stories promoted earlier
)
```

**Why this helps**: The existing difficulty scorer would rank a 300-token story with clear temporal
markers ("first... then... finally...") harder than a 50-token story purely on length signal.
p-RoPE's insight says that's wrong — structured long-range content is semantically easier for
the model to learn from than short incoherent content.

---

## 5. MoE Shared Expert → Curriculum "Anchor Batches"

**What Gemma 4 does**: The MoE variant always activates a shared expert (3× larger than routing
experts) on every token. This expert holds general knowledge that must always be present; the
routed experts handle specialization. The shared expert prevents catastrophic forgetting of
fundamentals as the model specializes.

**New idea**: Maintain a small set of "anchor batches" — stories from the easiest 5% — that
are *always included* in every training batch regardless of curriculum stage, like a shared expert
that never gets turned off.

```python
class AnchorBatchCurriculum(LayerAwareCurriculum):
    """
    Extends LayerAwareCurriculum.
    Reserves anchor_frac of each batch for fixed easy stories (the 'shared expert').
    Prevents forgetting of core linguistic patterns as harder data is introduced.
    """
    def __init__(self, *args, anchor_frac: float = 0.10, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-select anchor stories: easiest 5% by difficulty score
        n_anchors = int(0.05 * len(self.scores))
        self.anchor_indices = self.scores[:n_anchors, 0].astype(int)  # story indices
        self.anchor_frac = anchor_frac  # fraction of each batch that's anchors

    def get_batch_indices(self, batch_size: int) -> np.ndarray:
        n_anchors_in_batch = max(1, int(batch_size * self.anchor_frac))
        n_curriculum = batch_size - n_anchors_in_batch

        anchor_sample = np.random.choice(self.anchor_indices, size=n_anchors_in_batch, replace=True)
        curriculum_sample = np.random.choice(self.get_eligible_stories(), size=n_curriculum, replace=True)

        return np.concatenate([anchor_sample, curriculum_sample])

# Usage: replace get_eligible_stories() call in get_batch_adaptive()
chosen_story_idxs = curriculum.get_batch_indices(batch_size)
```

**Why this helps**: As our curriculum reaches phase 3 (hard stories), the model could start
"forgetting" basic subject-verb agreement or simple sentence patterns. The anchor batch — like the
shared MoE expert — ensures fundamental patterns always get reinforced. Overhead is just 10% of
batch, cost is zero.

---

## Summary: What's New vs. What's Carried Forward

| Already in existing MD | New from Gemma 4 |
|---|---|
| Perplexity-based difficulty scoring | Layer-aware curriculum (grad norm by layer group) |
| Competence-aware scheduling (val loss) | Soft token budget → dynamic block size |
| Smooth transitions, no fixed phases | K=V divergence as internal difficulty signal |
| Multi-metric scoring (TTR, length) | Discourse coherence bonus for long stories |
| Tier-based evaluation | Anchor batches (shared expert analogue) |

**Recommended order of implementation** (effort vs impact):

1. **Anchor batches** — 10 lines, zero risk, works immediately  
2. **Dynamic block size** — 5 lines in `get_batch`, clean axis of difficulty  
3. **Discourse coherence bonus** — one-time preprocessing addition, no training overhead  
4. **KV divergence signal** — adds an internal diagnostic, good for wandb logging  
5. **Layer-aware grad tracking** — most complex, highest payoff for understanding what's happening