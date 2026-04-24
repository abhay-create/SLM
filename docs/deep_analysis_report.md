# Deep Analysis: Why the Expansion Strategy Struggled

## The Core Problem in One Sentence

The model was asked to learn an entirely different language domain (adult fiction) after being deeply specialized on children's stories, with only 14% of the training budget that was used for the original specialization, and a replay buffer too small to prevent overwriting.

---

## 1. The Two Datasets Are Fundamentally Different Domains

### Content Comparison

| Property | TinyStories | WritingPrompts |
|---|---|---|
| **Target audience** | Children (ages 3-6) | Adult fiction writers |
| **Vocabulary** | Simple, repetitive | Complex, diverse, includes profanity |
| **Narrative structure** | Linear, single event, moral lesson | Multi-scene, dialogue-heavy, abstract |
| **Point of view** | Third person ("the little girl") | Mixed (first person "I", second person "you") |
| **Sentence complexity** | Simple/compound | Complex/compound-complex |
| **Typical content** | "Lily played in the park" | "Death appeared before him" |

### Actual Samples from the Data

**TinyStories** (what the model knows well):
> Once upon a time, there was a little girl named Lily who loved to collect things. She had a big box where she kept all her treasures. One day, she went to the park and saw a pond with many ducks swimming in it.

**WritingPrompts** (what it was asked to learn):
> "Have I been here before?" "Most certainly, you come in every year around this time." "Well I'm glad to be back. Another please." He begins to pour a larger glass just for her. "Are you sure? These drinks are expensive."

These are not two variants of the same task. They are **two completely different languages** from the perspective of a small model.

---

## 2. Quantitative Dataset Comparison

### Token-Level Statistics

| Metric | TinyStories | WritingPrompts | Gap |
|---|---|---|---|
| **Token entropy** | 8.58 bits | 10.02 bits | +1.44 bits (17% more diverse) |
| **Unique tokens** (50K sample) | ~8,900 | ~14,800 | WP uses 66% more vocabulary |
| **Bigram repetition** | 0.170 | 0.109 | TS is 56% more repetitive |
| **Trigram repetition** | 0.065 | 0.034 | TS is 91% more repetitive |
| **Difficulty score mean** | 18.0 | 36.7 | WP is 2× harder |
| **Difficulty score range** | 5.9 – 67.6 | 2.2 – 679.7 | WP has 10× wider range |
| **Difficulty p90** | 20.9 | 48.3 | WP hard tail is 2.3× harder |

### What This Means

- WritingPrompts has **1.44 bits more entropy per token**. For a sequence of 384 tokens, that's ~553 more bits of information the model must encode per sample. This is a massive complexity jump for a 50M-parameter model.
- TinyStories is **highly repetitive** — the same phrases ("once upon a time", "the little girl", "was very happy") appear constantly. The model can achieve low loss by memorizing a small set of patterns. WritingPrompts offers no such shortcut.
- The difficulty score distributions barely overlap: TS clusters around 15-21, while WP spans 27-48 at the 10th-90th percentile range.

### Tokens That Exist Only in WritingPrompts

The model encountered tokens it had **never seen during 377M tokens of TinyStories training**:

| Token | Frequency in WP | Category |
|---|---|---|
| `"` (smart quotes) | 44,827 | Formatting |
| `]` | 28,014 | Reddit formatting |
| `\n\n\n` | 14,162 | Paragraph breaks |
| `hell` | 4,776 | Adult vocabulary |
| `shit` | 4,149 | Adult vocabulary |
| `fucking` | 3,659 | Adult vocabulary |
| `**` | 4,725 | Markdown formatting |
| `Death` | 2,284 | Thematic |
| `humanity` | 2,146 | Thematic |
| `species` | 1,718 | Thematic |

These aren't edge cases — they represent the fundamental vocabulary of the new domain.

---

## 3. The Training Budget Mismatch

This is arguably the biggest problem:

| Stage | Dataset | Tokens Used | Dataset Size | Coverage |
|---|---|---|---|---|
| **Baseline** | TinyStories | 376.8M | 450M | **83.7%** |
| **Stage B** | WritingPrompts | 27.7M | 200M | **13.9%** |
| **Stage C** | WritingPrompts | 28.7M | 200M | **14.4%** |

The baseline model saw each TinyStories example approximately **0.84 times on average**. Stages B and C saw each WritingPrompts example approximately **0.14 times**. The model was given **6× less exposure** to WritingPrompts than it had to TinyStories, while being asked to learn a fundamentally harder domain.

Combined, the expansion stages consumed 81M tokens total — only **21.5%** of what the baseline used on a simpler dataset.

---

## 4. Perplexity Results Tell the Full Story

| Checkpoint | TS Val Loss | TS PPL | WP Val Loss | WP PPL |
|---|---|---|---|---|
| **Baseline (50M)** | 1.5531 | 4.73 | 6.4656 | 642.63 |
| **Stage A (58M)** | 1.6295 | 5.10 | 6.5225 | 680.31 |
| **Stage B (71M)** | 2.1704 | 8.76 | 4.9473 | 140.80 |
| **Stage C (99M)** | 2.2188 | 9.20 | 4.8546 | 128.33 |

### Key Observations

1. **The baseline model has zero understanding of WritingPrompts** (PPL 643). It's essentially random on this domain.
2. **Stage A did not help with WP at all** (PPL 680). Training on more TinyStories with extra layers did nothing for WP — confirming these are separate domains.
3. **Stages B/C reduced WP PPL from 643 → 128** — a 5× improvement. The model *did* learn significant WP structure.
4. **But the cost was TS PPL doubling** from 4.73 → 9.20 — an unacceptable level of forgetting.

### The Trade-off Ratio

- **WP gain**: PPL 643 → 128 = **5.0× improvement**
- **TS cost**: PPL 4.73 → 9.20 = **1.9× degradation**

The model traded ~50% of its TinyStories fluency for a partial understanding of WritingPrompts. Whether this trade-off is acceptable depends on the use case, but it was not the intended outcome.

---

## 5. Generation Quality Evidence

### What the Baseline Does Well (and Stages B/C Lost)

**Baseline** on "The dog ran to the park because":
> The dog ran to the park because he loved to play in his yard. One day, the dog was playing in his yard when he saw a cat. The cat was very frightened and ran away. The dog felt sorry for the cat and wanted to help him.

✅ Clear narrative, consistent characters, logical cause-and-effect, age-appropriate.

**Stage C** on the same prompt:
> The dog ran to the park because it was the only thing we can do to make the world better. It was a hard day, but we did it. In the end, we were able to catch the person who was so young, they started to cry.

❌ Incoherent, loses the dog entirely, switches to philosophical WP-style language.

### What Stages B/C Gained

**Baseline** on "Death appeared before him and said":
> Death appeared before him and said, "Oh! I'm sorry, I didn't mean to make you angry. I just want you to know that you are just a little boy."

❌ Completely wrong register — turns a dark fantasy prompt into a children's story.

**Stage C** on the same prompt:
> Death appeared before him and said it. The man walked over to the man and looked at her. He was very surprised. The man had a red watch on his wrist.

⚠️ At least attempts adult prose and doesn't infantilize the prompt, but coherence is poor.

### The "Mode Collapse" Problem

Stage C exhibits a characteristic symptom: it generates text that is **neither good children's stories nor good adult fiction**. It sits in an uncomfortable middle ground — using WP-style vocabulary ("I never thought", "in that moment") but with TS-style simplistic structure and frequent repetition.

---

## 6. Root Cause Analysis

### Why the Strategy Failed

```
Root cause tree:

1. DOMAIN GAP TOO LARGE
   ├── TS and WP share only basic English function words
   ├── WP entropy is 17% higher (1.44 bits/token)
   ├── WP vocabulary is 66% larger
   └── WP never-seen tokens include core domain vocabulary

2. TRAINING BUDGET TOO SMALL
   ├── 28M tokens vs 377M for baseline (7.4% ratio)
   ├── Plateau detector triggered at 3000-3500 steps
   ├── Curriculum only reached 34-35% of WP data
   └── Model never saw medium/hard WP tiers in meaningful proportion

3. ANTI-FORGETTING MECHANISMS INSUFFICIENT
   ├── 15% replay buffer ≪ needed for this domain gap
   ├── No EWC, knowledge distillation, or parameter freezing
   ├── Differential LR (0.5×) too aggressive for pretrained params
   └── TS val was not used as an early stopping criterion

4. PLATEAU DETECTION TOO AGGRESSIVE
   ├── patience=5 with eval_interval=500 → exit after 2500 no-improvement steps
   ├── Learning rate may have been too low for the complexity jump
   └── The model needed 10-50× more training to properly learn WP
```

### The Fundamental Strategic Error

The plan assumed WritingPrompts would be a **gradual extension** of TinyStories — harder stories, but the same basic skill. In reality, they are **different tasks entirely**:

- TinyStories = generate formulaic children's narratives with simple vocabulary
- WritingPrompts = generate diverse adult fiction with complex dialogue, varied POV, and abstract themes

This is closer to training an English model and then asking it to learn French, rather than training on easy English and then introducing hard English.

---

## 7. What Would Work Better

### Option A: Bridge the Domain Gap
Instead of jumping from TinyStories → WritingPrompts, introduce intermediate datasets:
1. **FanFiction for Kids** (simple vocabulary but longer narratives)
2. **SimpleWiki stories** (factual but with narrative structure)
3. **Young Adult fiction** (bridges the vocabulary gap)

### Option B: Much More Training + Stronger Forgetting Prevention
- **50-70% replay buffer** during domain transition
- **EWC** (Elastic Weight Consolidation) to penalize large changes to important weights
- **Knowledge distillation** from the frozen baseline to regularize outputs
- **10× longer training** (200-300M tokens per stage, not 28M)
- **Less aggressive plateau detection** (patience=15, eval_interval=250)

### Option C: Multi-task from the Start
Instead of sequential training, **mix both datasets from the beginning** with the expanded model:
- 60% TinyStories + 40% WritingPrompts initially
- Gradually shift to 30%/70%
- This prevents the model from ever fully specializing in one domain

### Option D: Train from Scratch at 100M
The expansion plan correctly identified this as a necessary comparison. A scratch-trained 100M model on a mixed TS+WP curriculum would likely outperform the expanded model, because it would never develop the domain-specific biases that are so hard to overwrite.

---

## 8. Summary of All Test Results

### Perplexity Matrix

| Model | TS Val | WP Val | TS PPL | WP PPL |
|---|---|---|---|---|
| Baseline (50M) | **1.553** | 6.466 | **4.73** | 642.6 |
| Stage A (58M) | 1.630 | 6.523 | 5.10 | 680.3 |
| Stage B (71M) | 2.170 | **4.947** | 8.76 | 140.8 |
| Stage C (99M) | 2.219 | **4.855** | 9.20 | **128.3** |

### Generation Quality Matrix

| Prompt Style | Baseline | Stage A | Stage B | Stage C |
|---|---|---|---|---|
| TinyStories | ✅ Excellent | ✅ Good | ❌ Broken | ❌ Broken |
| WritingPrompts | ❌ Wrong register | ❌ Wrong register | ⚠️ Attempts adult prose | ⚠️ Attempts adult prose |
| Neutral | ✅ Coherent (childish) | ✅ Coherent (childish) | ⚠️ Repetitive | ⚠️ Repetitive |

### Bottom Line

The expansion infrastructure is solid. The model successfully grew from 45.8M to 99.4M parameters using function-preserving operators. But the **training strategy** — sequential domain transfer with minimal replay and insufficient training budget — produced a model that is worse at its original task and only partially competent at the new one.

---

## Appendix: Raw Output Files

All raw evaluation outputs are saved in the project:
- [deep_data_analysis_output.txt](file:///home/user21/slm/docs/deep_data_analysis_output.txt) — Full dataset comparison
- [full_eval_output.txt](file:///home/user21/slm/docs/full_eval_output.txt) — All generation + perplexity results
- [evaluation_output.txt](file:///home/user21/slm/docs/evaluation_output.txt) — Cross-stage comparison
- [docs/figures/](file:///home/user21/slm/docs/figures/) — All 6 analysis plots
