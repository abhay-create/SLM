"""
Evaluation utilities for the SLM project
Compute perplexity, loss, and generate sample text
"""

import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm


class Evaluator:
    """Evaluation utilities for language models"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    @torch.no_grad()
    def compute_perplexity(self, dataloader) -> float:
        """Compute perplexity on a dataset"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for x, y in tqdm(dataloader, desc="Computing perplexity"):
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    @torch.no_grad()
    def evaluate_all(self, dataloader) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        self.model.eval()
        total_loss = 0.0
        num_tokens = 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            
            batch_size, seq_len = x.shape
            total_loss += loss.item() * batch_size * seq_len
            num_tokens += batch_size * seq_len
        
        avg_loss = total_loss / num_tokens
        perplexity = np.exp(avg_loss)
        bits_per_byte = avg_loss / np.log(2)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'bits_per_byte': bits_per_byte,
        }
    
    @torch.no_grad()
    def generate_samples(self, prompts: List[str], max_length: int = 100, temperature: float = 1.0, top_k: int = 50) -> List[str]:
        """Generate text samples from prompts"""
        self.model.eval()
        samples = []
        
        for prompt in prompts:
            # Encode prompt
            tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
            # Decode
            generated_text = self.tokenizer.decode(output_ids[0].tolist())
            samples.append(generated_text)
        
        return samples
    
    def evaluate_and_sample(self, val_loader, prompts: List[str] = None) -> Dict:
        """Run full evaluation with metrics and samples"""
        # Compute metrics
        metrics = self.evaluate_all(val_loader)
        
        # Generate samples if prompts provided
        if prompts:
            samples = self.generate_samples(prompts, max_length=50)
            metrics['samples'] = list(zip(prompts, samples))
        
        return metrics


def print_evaluation_results(metrics: Dict, stage_name: str = "Evaluation"):
    """Pretty print evaluation results"""
    print(f"\n{'='*60}")
    print(f"{stage_name} Results")
    print(f"{'='*60}")
    print(f"Loss:          {metrics['loss']:.4f}")
    print(f"Perplexity:    {metrics['perplexity']:.2f}")
    print(f"Bits/Byte:     {metrics['bits_per_byte']:.4f}")
    
    if 'samples' in metrics:
        print(f"\nGenerated Samples:")
        print(f"{'-'*60}")
        for i, (prompt, sample) in enumerate(metrics['samples'], 1):
            print(f"\n{i}. Prompt: {prompt}")
            print(f"   Output: {sample[:200]}...")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Evaluator module loaded successfully")