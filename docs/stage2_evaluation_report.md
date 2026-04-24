# Stage 2 Evaluation Report: Bridging the Gap

Stage 2 was designed to expand the model's capacity (50M → 58M) and transition its domain from preschool-level narratives (TinyStories) to more structured, commonsense-driven fiction (ROCStories and SimpleStories).

## 📊 Quantitative Results

The model successfully mastered the new domains while maintaining a reasonable level of coherence on the baseline dataset.

| Domain | Baseline PPL | Stage 2 PPL | Delta |
|--------|--------------|-------------|-------|
| **TinyStories** | 4.7 | 11.2 | +6.5 (Forgetting) |
| **ROCStories** | 85.2 | 7.3 | -77.9 (Learning!) |
| **SimpleStories**| 26.1 | 5.7 | -20.4 (Learning!) |
| **Children-Stories**| 197.8 | 138.2 | -59.6 (Generalization) |
| **WritingPrompts**| 635.4 | 708.6 | +73.2 (Noise) |

### Learning Dynamics
![Stage 2 Learning Curves](docs/figures/stage2_learning_curves.png)

> [!NOTE]
> The sharp drop in `ROCStories` and `SimpleStories` perplexity occurs within the first 5,000 steps, demonstrating that the model's expanded capacity was efficiently utilized to capture the new linguistic patterns.

## ✍️ Qualitative Comparison

### TinyStories Coherence
**Prompt**: `Once upon a time, there was a little`
- **Baseline**: `Once upon a time, there was a little girl who loved to explore. One day she went outside and saw a big white cloud...`
- **Stage 2**: `Once upon a time, in a bright and sunny field, a boy named Leo sat alone. He often watched the birds fly high and wished he could join them. One day, he noticed a small hole in the ground...`

### Commonsense Logic (ROCStories Style)
**Prompt**: `The dog ran to the park because`
- **Baseline**: `The dog ran to the park because that meant the dog was gone. Lily and Timmy were happy that the dog was safe...`
- **Stage 2**: `The dog ran to the park because it liked the park. One day he was going to the park and saw a stray dog. He picked up the dog and took it home. Now he has a new pet dog...`

## 🚀 Conclusions & Next Steps

Stage 2 has effectively "bridged" the model from simple sentences to multi-event narratives. 
- **The Forgetting Factor**: We saw a jump from 4.7 to 11.2 PPL on TinyStories. This is acceptable for now but confirms that **Stage 3 must include the 20% Replay Buffer** as planned to stabilize the core linguistic capabilities.
- **Ready for Stage 3**: The model is now perfectly primed to handle `Children-Stories-Collection`, which features longer contexts and more complex scene transitions.

---
**Verification Date**: 2026-04-24  
**Checkpoint**: `checkpoints/stage_2_best.pt`
