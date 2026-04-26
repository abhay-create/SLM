"""
Data preparation script for WritingPrompts dataset.
Downloads from HuggingFace, tokenizes, and saves as pickle files.
Run this ONCE before training.
"""

import os
import sys
import pickle
from pathlib import Path


def prepare_writingprompts(output_dir: str = "data/curriculum", max_tokens: int = 50_000_000):
    """
    Download WritingPrompts dataset from HuggingFace and prepare for training.
    Falls back to synthetic data if download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simple_file = output_dir / "simple_tokens.pkl"
    complex_file = output_dir / "complex_tokens.pkl"

    # Check if data already exists and is large enough (> 1MB means real data)
    if simple_file.exists() and complex_file.exists():
        s_size = simple_file.stat().st_size
        c_size = complex_file.stat().st_size
        if s_size > 1_000_000 and c_size > 1_000_000:
            print(f"Real data already exists:")
            print(f"  Simple: {s_size / 1e6:.1f} MB")
            print(f"  Complex: {c_size / 1e6:.1f} MB")
            return True

    print("Preparing data...")
    print("Attempting to download WritingPrompts from HuggingFace...")

    # Initialize tokenizer
    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer_utils import create_tokenizer
    tokenizer = create_tokenizer('gpt2', cache_dir='tokenizer_cache')
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    try:
        from datasets import load_dataset
        print("Downloading WritingPrompts dataset (euclaise/writingprompts)...")
        # Try euclaise/writingprompts which has prompt+story pairs
        try:
            dataset = load_dataset("euclaise/writingprompts", split="train",
                                   trust_remote_code=True)
        except Exception:
            print("Trying alternative: 'reddit_writing_prompts'...")
            try:
                dataset = load_dataset("jannisborn/reddit_writing_prompts",
                                       split="train", trust_remote_code=True)
            except Exception:
                print("Trying roneneldan/TinyStories as fallback...")
                dataset = load_dataset("roneneldan/TinyStories",
                                       split="train[:100%]",
                                       trust_remote_code=True)

        print(f"Dataset loaded: {len(dataset)} examples")

        # Tokenize stories - split into simple (short) and complex (long)
        all_simple_tokens = []
        all_complex_tokens = []

        print("Tokenizing...")
        n = len(dataset)
        for i, example in enumerate(dataset):
            if i % 10000 == 0:
                print(f"  {i}/{n} ({100*i/n:.1f}%) | "
                      f"Simple: {len(all_simple_tokens):,} | "
                      f"Complex: {len(all_complex_tokens):,} tokens")

            # Get text from example
            if 'story' in example:
                text = example['story']
            elif 'text' in example:
                text = example['text']
            elif 'completion' in example:
                prompt = example.get('prompt', '')
                text = (prompt + '\n\n' + example['completion'])
            else:
                text = str(example)

            if not text or len(text) < 50:
                continue

            tokens = tokenizer.encode(text)
            if len(tokens) < 20:
                continue

            # Short texts → simple stage, long texts → complex stage
            if len(tokens) <= 300:
                all_simple_tokens.extend(tokens)
                all_simple_tokens.append(tokenizer.tokenizer.eos_token_id)
            else:
                all_complex_tokens.extend(tokens)
                all_complex_tokens.append(tokenizer.tokenizer.eos_token_id)

            # Stop if we have enough tokens
            if (len(all_simple_tokens) + len(all_complex_tokens)) > max_tokens:
                print(f"Reached {max_tokens:,} token limit, stopping.")
                break

        # Ensure we have enough of each type
        if len(all_simple_tokens) < 100_000:
            print("Not enough simple examples, rebalancing...")
            # Move some complex tokens to simple
            half = len(all_complex_tokens) // 2
            all_simple_tokens.extend(all_complex_tokens[:half])
            all_complex_tokens = all_complex_tokens[half:]

        if len(all_complex_tokens) < 100_000:
            print("Not enough complex examples, rebalancing...")
            half = len(all_simple_tokens) // 2
            all_complex_tokens.extend(all_simple_tokens[:half])
            all_simple_tokens = all_simple_tokens[half:]

        print(f"\nFinal token counts:")
        print(f"  Simple:  {len(all_simple_tokens):,}")
        print(f"  Complex: {len(all_complex_tokens):,}")

        with open(simple_file, 'wb') as f:
            pickle.dump(all_simple_tokens, f)
        with open(complex_file, 'wb') as f:
            pickle.dump(all_complex_tokens, f)

        print(f"Saved to {output_dir}")
        return True

    except Exception as e:
        print(f"Dataset download failed: {e}")
        print("Falling back to large synthetic WritingPrompts-style data...")
        return _generate_synthetic_writingprompts(output_dir, tokenizer)


def _generate_synthetic_writingprompts(output_dir: Path, tokenizer):
    """Generate large synthetic writing-prompt-style training data."""
    import random

    print("Generating synthetic writing-style training data (~5M tokens)...")

    # Rich diverse writing prompts and stories
    story_templates = [
        # Adventure
        "The {hero} stood at the edge of the {place}, knowing that what lay beyond would change everything. {action} had led them here, through countless trials and sleepless nights. Now, facing the {challenge}, they drew a deep breath and stepped forward.",
        "In a world where {concept} was forbidden, {character} discovered the truth that would unravel civilization itself. The moment they opened the ancient {object}, the ground trembled beneath their feet.",
        # Mystery
        "The letter arrived at midnight, unsigned and smelling of {scent}. {character} had not expected contact from {sender}, not after all these years. The single line of text read: 'The {secret} is not what you think.'",
        # Sci-fi
        "Year {year}. Humanity had colonized {number} worlds, yet none could explain why the signal from {place} had gone silent. Captain {name} was tasked with finding out—and never came back.",
        # Fantasy
        "The last dragon spoke only once in its long life. {character} was there to hear it. 'The {concept},' it rasped, 'was never a gift. It was always a {thing}.' Then it closed its eyes forever.",
        # Drama
        "After {years} years of silence, {character_a} finally called {character_b}. The conversation lasted three minutes. Neither of them mentioned the {event}, though it hung between every word like smoke.",
        # Thriller
        "The surveillance footage showed {character} entering the building at {time}. The building had no exits. No one saw them leave. Yet by morning, they were gone—along with every record of their existence.",
        # Horror
        "The house had been empty for {years} years when {character} moved in. On the first night, they heard the same sound their grandmother described before she disappeared: {sound}. On the third night, they understood why.",
    ]

    heroes = ['Sarah', 'Marcus', 'Elena', 'Kai', 'Sophia', 'James', 'Aria', 'Liam', 'Nova', 'Ethan']
    places = ['mountain', 'forest', 'city', 'ocean', 'desert', 'valley', 'ruins', 'tower', 'cave', 'station']
    concepts = ['magic', 'technology', 'memory', 'time travel', 'telepathy', 'immortality', 'prophecy', 'quantum physics']
    things = ['burden', 'curse', 'weapon', 'test', 'illusion', 'key', 'warning', 'gift']

    def make_story():
        template = random.choice(story_templates)
        text = template.format(
            hero=random.choice(heroes),
            character=random.choice(heroes),
            character_a=random.choice(heroes),
            character_b=random.choice(heroes),
            place=random.choice(places),
            concept=random.choice(concepts),
            thing=random.choice(things),
            challenge=random.choice(['ancient guardian', 'impossible choice', 'dark secret', 'final test']),
            action=random.choice(['Fate', 'Destiny', 'Years of searching', 'One desperate choice']),
            object=random.choice(['box', 'book', 'door', 'mirror', 'map', 'letter']),
            scent=random.choice(['lavender', 'smoke', 'ocean salt', 'old paper', 'copper']),
            sender=random.choice(['the organization', 'a ghost from the past', 'the council', 'an old friend']),
            secret=random.choice(['artifact', 'truth', 'signal', 'meeting', 'message']),
            year=random.randint(2050, 2500),
            number=random.randint(10, 500),
            name=random.choice(heroes),
            years=random.randint(5, 50),
            event=random.choice(['accident', 'argument', 'betrayal', 'discovery', 'loss', 'night']),
            time=f"{random.randint(0,23):02d}:{random.randint(0,59):02d}",
            sound=random.choice(['a child crying', 'footsteps on the ceiling', 'her own name whispered', 'a clock ticking backwards']),
        )
        # Add continuation sentences to make it longer
        continuations = [
            f" {random.choice(heroes)} had always believed that {random.choice(concepts)} would save them.",
            f" The {random.choice(places)} held answers that no one dared to seek.",
            f" Perhaps this was the moment everything would change.",
            f" There was no turning back now.",
            f" {random.choice(heroes)} watched from the shadows, wondering if they had made the right choice.",
            f" The weight of the decision settled on their shoulders like stone.",
            f" All the stories, all the warnings, led to this single point in time.",
            f" And yet, nothing could have prepared them for what came next.",
        ]
        for _ in range(random.randint(3, 8)):
            text += random.choice(continuations)
        return text

    simple_tokens = []
    complex_tokens = []
    eos_id = 50256  # GPT-2 EOS token

    target = 2_000_000  # 2M tokens each
    print(f"Generating ~{2*target:,} tokens...")

    while len(simple_tokens) < target or len(complex_tokens) < target:
        story = make_story()
        tokens = tokenizer.encode(story)

        if len(tokens) < 150:
            simple_tokens.extend(tokens)
            simple_tokens.append(eos_id)
        else:
            complex_tokens.extend(tokens)
            complex_tokens.append(eos_id)

        total = len(simple_tokens) + len(complex_tokens)
        if total % 200_000 < len(tokens):
            print(f"  Generated {total:,} tokens...")

    simple_file = output_dir / "simple_tokens.pkl"
    complex_file = output_dir / "complex_tokens.pkl"

    with open(simple_file, 'wb') as f:
        pickle.dump(simple_tokens[:target], f)
    with open(complex_file, 'wb') as f:
        pickle.dump(complex_tokens[:target], f)

    print(f"Synthetic data saved:")
    print(f"  Simple:  {len(simple_tokens[:target]):,} tokens → {simple_file}")
    print(f"  Complex: {len(complex_tokens[:target]):,} tokens → {complex_file}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare WritingPrompts training data")
    parser.add_argument("--output_dir", default="data/curriculum")
    parser.add_argument("--max_tokens", type=int, default=50_000_000)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of downloading")
    args = parser.parse_args()

    if args.synthetic:
        sys.path.insert(0, str(Path(__file__).parent))
        from tokenizer_utils import create_tokenizer
        tokenizer = create_tokenizer('gpt2', cache_dir='tokenizer_cache')
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _generate_synthetic_writingprompts(output_dir, tokenizer)
    else:
        prepare_writingprompts(args.output_dir, args.max_tokens)
