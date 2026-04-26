"""
Dataset handling for curriculum and direct learning
Supports TinyStories and other text datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional, Dict
import pickle
import os
from pathlib import Path


class TextDataset(Dataset):
    """Simple text dataset for token sequences"""
    def __init__(self, tokens: List[int], seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class CurriculumDataset:
    """
    Curriculum learning dataset manager
    Supports multiple stages with different data mixes
    """
    def __init__(self, data_dir: str, seq_len: int = 512):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stages = []
        self.current_stage = 0
        
    def add_stage(self, name: str, files: List[str], mix_ratio: Optional[List[float]] = None):
        """Add a curriculum stage with data files and optional mixing ratios"""
        if mix_ratio is None:
            mix_ratio = [1.0 / len(files)] * len(files)
        
        assert len(files) == len(mix_ratio), "Files and ratios must match"
        assert abs(sum(mix_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.stages.append({
            'name': name,
            'files': files,
            'mix_ratio': mix_ratio
        })
    
    def get_stage_dataloader(self, stage_idx: int, batch_size: int, shuffle: bool = True):
        """Get dataloader for a specific stage"""
        if stage_idx >= len(self.stages):
            raise ValueError(f"Stage {stage_idx} does not exist")
        
        stage = self.stages[stage_idx]
        
        # Load and mix data according to ratios
        all_tokens = []
        for file_path, ratio in zip(stage['files'], stage['mix_ratio']):
            full_path = self.data_dir / file_path
            if not full_path.exists():
                print(f"Warning: {full_path} not found, skipping...")
                continue
            
            with open(full_path, 'rb') as f:
                tokens = pickle.load(f)
            
            # Sample according to ratio
            n_samples = int(len(tokens) * ratio)
            if n_samples > 0:
                all_tokens.extend(tokens[:n_samples])
        
        if not all_tokens:
            raise ValueError(f"No data loaded for stage {stage_idx}")
        
        dataset = TextDataset(all_tokens, self.seq_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


class SimpleTextDataset:
    """
    Simple dataset for direct learning (no curriculum)
    Loads a single text file and tokenizes it
    """
    def __init__(self, text_file: str, tokenizer, seq_len: int = 512):
        self.seq_len = seq_len
        
        # Read text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize
        self.tokens = tokenizer.encode(text)
        print(f"Loaded {len(self.tokens):,} tokens from {text_file}")
    
    def get_dataloader(self, batch_size: int, shuffle: bool = True):
        """Get dataloader for training"""
        dataset = TextDataset(self.tokens, self.seq_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def prepare_tinystories_data(output_dir: str, tokenizer, max_samples: int = 100000):
    """
    Download and prepare TinyStories dataset
    Creates tokenized pickle files for fast loading
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already prepared
    train_file = output_dir / "train_tokens.pkl"
    val_file = output_dir / "val_tokens.pkl"
    
    if train_file.exists() and val_file.exists():
        print("TinyStories data already prepared!")
        return str(train_file), str(val_file)
    
    print("Preparing TinyStories dataset...")
    
    # For this example, we'll create synthetic data
    # In production, you'd download from HuggingFace datasets
    # from datasets import load_dataset
    # dataset = load_dataset("roneneldan/TinyStories")
    
    # Generate synthetic simple stories for demonstration
    sample_stories = [
        "Once upon a time, there was a little girl named Lucy. She loved to play in the garden. One day, she found a beautiful butterfly.",
        "Tom and his dog went to the park. They played with a ball. The dog was very happy.",
        "The sun was shining bright. Birds were singing in the trees. It was a perfect day for a picnic.",
        "Emma had a red balloon. The wind was strong. Her balloon flew away into the sky.",
        "A small cat sat on the wall. It watched the birds fly by. The cat wanted to catch them.",
    ]
    
    # Repeat to create larger dataset
    train_stories = sample_stories * (max_samples // len(sample_stories))
    val_stories = sample_stories * 100  # Smaller validation set
    
    # Tokenize
    train_text = " ".join(train_stories)
    val_text = " ".join(val_stories)
    
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    
    # Save
    with open(train_file, 'wb') as f:
        pickle.dump(train_tokens, f)
    
    with open(val_file, 'wb') as f:
        pickle.dump(val_tokens, f)
    
    print(f"Prepared {len(train_tokens):,} training tokens")
    print(f"Prepared {len(val_tokens):,} validation tokens")
    
    return str(train_file), str(val_file)


def create_curriculum_stages(data_dir: str, tokenizer):
    """
    Create a 3-stage curriculum for demonstration:
    Stage 1: Simple stories (100%)
    Stage 2: Mixed complexity (70% simple, 30% complex)
    Stage 3: Complex stories (100%)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic data for different complexity levels
    simple_stories = [
        "The cat sat. The dog ran. The bird flew.",
        "I like cake. You like pie. We eat food.",
        "Red ball. Blue car. Green tree.",
    ] * 1000
    
    complex_stories = [
        "In the bustling metropolis, where towering skyscrapers pierced the clouds and the ceaseless hum of humanity filled the air, there lived a young woman named Isabella who harbored dreams of becoming a renowned artist.",
        "The intricate mechanisms of quantum entanglement have puzzled physicists for decades, challenging our fundamental understanding of reality and causality in ways that continue to inspire both theoretical and experimental research.",
        "Through the winding corridors of time, civilizations have risen and fallen, each leaving behind echoes of their achievements and failures, testament to humanity's eternal struggle between ambition and humility.",
    ] * 500
    
    # Tokenize and save
    simple_tokens = tokenizer.encode(" ".join(simple_stories))
    complex_tokens = tokenizer.encode(" ".join(complex_stories))
    
    with open(data_dir / "simple_tokens.pkl", 'wb') as f:
        pickle.dump(simple_tokens, f)
    
    with open(data_dir / "complex_tokens.pkl", 'wb') as f:
        pickle.dump(complex_tokens, f)
    
    print(f"Created curriculum data:")
    print(f"  Simple: {len(simple_tokens):,} tokens")
    print(f"  Complex: {len(complex_tokens):,} tokens")
    
    # Create curriculum
    curriculum = CurriculumDataset(data_dir, seq_len=512)
    curriculum.add_stage("stage1_simple", ["simple_tokens.pkl"], [1.0])
    curriculum.add_stage("stage2_mixed", ["simple_tokens.pkl", "complex_tokens.pkl"], [0.7, 0.3])
    curriculum.add_stage("stage3_complex", ["complex_tokens.pkl"], [1.0])
    
    return curriculum


if __name__ == "__main__":
    # Test dataset creation
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    
    # Create dummy tokenizer for testing
    tokenizer = Tokenizer(BPE())
    
    # Test curriculum creation
    curriculum = create_curriculum_stages("data/curriculum", tokenizer)
    print(f"Created curriculum with {len(curriculum.stages)} stages")