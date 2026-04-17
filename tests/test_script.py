import numpy as np
import torch
import random
from src.curriculum_dataset import CurriculumStageDataset, CompetenceScheduler

print("=== CurriculumStageDataset Test ===")
ds = CurriculumStageDataset(anchor_frac=0.10)
ds.mode = "adaptive"

# 1000 chunks total
ds.chunks = [np.arange(10, dtype=np.int64) for _ in range(1000)]

# Mock difficulty scores
mock_scores = []
for i in range(100):
    mock_scores.append([i, float(i), 10.0, 0.5, 200.0])
ds.scores = np.array(mock_scores)

n_chunks = len(ds.chunks)
chunk_difficulties = ds._map_scores_to_chunks(n_chunks)
ds.sorted_indices = sorted(range(n_chunks), key=lambda i: chunk_difficulties[i])

initial_fraction = 0.20
ds.set_eligible_fraction(initial_fraction)
ds.anchor_count = max(1, int(n_chunks * 0.05)) # 5% is 50 chunks

print(f"Total chunks: {len(ds.chunks)}")
print(f"Eligible fraction: {initial_fraction*100:.0f}% -> {ds.eligible_count} chunks")
print(f"Anchor pool size: {ds.anchor_count} chunks (bottom 5%)")

anchor_hits = 0
curriculum_hits = 0
random.seed(42)

for _ in range(1000):
    val = random.random()
    if val < ds.anchor_frac:
        actual_idx = ds.sorted_indices[random.randint(0, ds.anchor_count - 1)]
    else:
        idx = random.randint(0, 100000)
        actual_idx = ds.sorted_indices[idx % ds.eligible_count]
        
    diff = chunk_difficulties[actual_idx]
    if diff < (100 * 0.05):
        anchor_hits += 1
    else:
        curriculum_hits += 1

print("\n=== Sampling Results (1000 trials) ===")
print(f"Anchor items (easiest 5%): {anchor_hits} out of 1000 hits")
print(f"Curriculum items (the rest): {curriculum_hits} hits")

print("\n=== CompetenceScheduler Test ===")
s = CompetenceScheduler(initial_fraction=0.15)
print(f"Initial fraction: {s.get_current_fraction()*100:.0f}%")
print("Simulating warming up (2 epochs):")
for val in [5.0, 4.5]:
    info = s.update_competence(val)
    print(f"  val={val:.2f} -> fraction={info['fraction']*100:.1f}% ({info['status']})")

print("Simulating strong improvements:")
for val in [4.0, 3.5, 3.0, 2.5]:
    info = s.update_competence(val)
    print(f"  val={val:.2f} -> fraction={info['fraction']*100:.1f}% ({info['status']})")

print("Simulating deep layer instability block:")
for val in [2.0, 1.5]:
    info = s.update_competence(val, deep_layers_stable=False)
    print(f"  val={val:.2f} -> fraction={info['fraction']*100:.1f}% ({info['status']})")
