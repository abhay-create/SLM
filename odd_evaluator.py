"""
OOD Evaluation script for the SLM model (writingprompts).

This script generates model outputs for a set of out-of-distribution prompts
and runs heuristic evaluations for constraint satisfaction.

Usage examples:

python eval_ood.py --checkpoint checkpoints/curriculum/best.pt --device cpu --max_new_tokens 256

Requirements:
- torch
- transformers (tokenizer)

"""

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import torch

from tokenizer_utils import create_tokenizer
from model import create_model


PROMPTS = [
    {
        "id": "spatial_state_tracking",
        "title": "Spatial State Tracking and Logical Consistency",
        "prompt": (
            "Write a narrative detailing a protagonist dismantling a bomb in a sealed 3x3x3 meter room. "
            "There are exactly four tools in the room: a wrench, a wire cutter, a chronometer, and a flashlight. "
            "After every paragraph, you must explicitly output the current spatial coordinates (x,y,z) of the protagonist "
            "and all four tools within the room's coordinate system, ensuring logical physical continuity of their placement and movement."
        ),
    },
    {
        "id": "technical_integration",
        "title": "Domain-Specific Technical Integration (Infrastructure & DevOps)",
        "prompt": (
            "Narrate the timeline of a catastrophic cascade failure within a microservices architecture hosted on AWS. "
            "Write the narrative from the first-person perspective of an Elastic Load Balancer attempting to route traffic to failing Docker containers across different Availability Zones. "
            "The narrative must accurately reflect the technical mechanics of health checks, subnet routing, and latency spikes."
        ),
    },
    {
        "id": "lipogram_strict_dialogue",
        "title": "Strict Token-Level and Syntactic Constraint Satisfaction",
        "prompt": (
            "Write an interaction between two individuals negotiating a contract. The narrative must strictly follow a lipogrammatic constraint: "
            "the letter 'e' must not appear in any word throughout the entire text. Additionally, every spoken line of dialogue must be exactly seven words long."
        ),
    },
    {
        "id": "reversed_causality",
        "title": "Reversed Causality and Thermodynamic Inversion",
        "prompt": (
            "Write a scene taking place in an isolated environment where entropy strictly decreases over time. "
            "Begin the narrative at a state of maximum thermodynamic disorder (e.g., shattered glass, dispersed heat) and progress logically toward a state of minimum disorder. "
            "Ensure the causal relationships between actions strictly adhere to this reversed physical law."
        ),
    },
    {
        "id": "rfc_format_emulation",
        "title": "Format Emulation Dislodged from Content (RFC style)",
        "prompt": (
            "Draft an incident report regarding a localized temporal paradox in a suburban kitchen. "
            "The document must be structured strictly according to the format of an IETF Request for Comments (RFC) document. "
            "It must include formal sections such as 'Abstract', 'Terminology', 'Specification', and 'Security Considerations', utilizing standard RFC 2119 requirement levels (MUST, SHOULD, MAY)."
        ),
    },
]


def safe_load_checkpoint(model: torch.nn.Module, path: str, device: torch.device):
    if not path:
        return False
    if not os.path.exists(path):
        print(f"Checkpoint path does not exist: {path}")
        return False
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    try:
        model.load_state_dict(sd)
        print("Loaded checkpoint (strict load)")
    except Exception as e:
        try:
            model.load_state_dict(sd, strict=False)
            print("Loaded checkpoint (non-strict load)")
        except Exception as e2:
            print("Failed to load checkpoint:", e2)
            return False
    return True


# ---------------------- Evaluators ----------------------

def _find_entity_coords(subtext: str, entity_names: List[str]) -> Dict[str, Tuple[float, float, float]]:
    # Pattern: entity[: -] (x,y,z) or entity at (x, y, z)
    coords = {}
    for entity in entity_names:
        # flexible matching for entity name
        pat = re.compile(rf"{re.escape(entity)}\s*(?:[:@-]*)\s*\(?\s*([+-]?[0-9]*\.?[0-9]+)\s*,\s*([+-]?[0-9]*\.?[0-9]+)\s*,\s*([+-]?[0-9]*\.?[0-9]+)\s*\)?",
                         flags=re.IGNORECASE)
        m = pat.search(subtext)
        if m:
            coords[entity.lower()] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return coords


def evaluate_spatial_tracking(text: str) -> Dict:
    result = {"pass": False, "reason": "", "details": {}}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) == 0:
        result["reason"] = "No paragraphs detected"
        return result

    entity_names = ["Protagonist", "wrench", "wire cutter", "chronometer", "flashlight"]
    missing_counts = 0
    paragraph_results = []

    cursor = 0
    for i, para in enumerate(paragraphs):
        # find the paragraph in the whole text to locate following coordinates
        pos = text.find(para, cursor)
        if pos == -1:
            pos = cursor
        end_pos = text.find(paragraphs[i + 1], pos + len(para)) if i + 1 < len(paragraphs) else len(text)
        following = text[pos + len(para):end_pos]
        coords = _find_entity_coords(following, entity_names)
        para_ok = True
        missing = [e for e in [n.lower() for n in entity_names] if e not in coords]
        if missing:
            para_ok = False
            missing_counts += 1
        else:
            # check numeric ranges (0..3 meters inclusive)
            for e, (x, y, z) in coords.items():
                if not (0.0 <= x <= 3.0 and 0.0 <= y <= 3.0 and 0.0 <= z <= 3.0):
                    para_ok = False
        paragraph_results.append({"paragraph_index": i, "coords_found": list(coords.keys()), "ok": para_ok})
        cursor = end_pos

    result["details"]["paragraphs_checked"] = len(paragraphs)
    result["details"]["paragraph_results"] = paragraph_results

    if missing_counts == 0:
        result["pass"] = True
        result["reason"] = "All paragraphs include coordinate listings for required entities within the 3x3x3 bounds."
    else:
        result["reason"] = f"{missing_counts} paragraphs missing some entity coordinates after the paragraph."
    return result


def evaluate_technical_integration(text: str) -> Dict:
    result = {"pass": False, "reason": "", "details": {}}
    text_low = text.lower()
    first_person = bool(re.search(r"\bI\b", text))
    # technical keywords heuristics
    keywords = [
        "health check",
        "healthcheck",
        "probe",
        "target group",
        "target groups",
        "availability zone",
        "availability zones",
        "subnet",
        "latency",
        "latency spike",
        "docker",
        "container",
        "elb",
        "elastic load balancer",
        "load balancer",
    ]
    found = [k for k in keywords if k in text_low]
    result["details"]["first_person"] = first_person
    result["details"]["keywords_found"] = found

    if first_person and (len(found) >= 3):
        result["pass"] = True
        result["reason"] = "First-person ELB narrator and several infrastructure keywords detected."
    else:
        result["pass"] = False
        if not first_person:
            result["reason"] = "Narrative not in first-person (no isolated 'I' detected)."
        else:
            result["reason"] = "Insufficient technical keywords detected."
    return result


def evaluate_lipogram_and_dialogue(text: str) -> Dict:
    result = {"pass": False, "reason": "", "details": {}}
    # No 'e' or 'E' anywhere
    if re.search(r"[eE]", text):
        result["reason"] = "The letter 'e' appears in the text."
        return result

    # Find quoted dialogue segments (double or single quotes)
    quotes = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    # quotes returns tuples from two groups; flatten
    dialogues = []
    for a, b in quotes:
        if a:
            dialogues.append(a.strip())
        elif b:
            dialogues.append(b.strip())

    # Also accept lines that start with '-' or with a name followed by colon as dialogue lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith('- '):
            dialogues.append(ln[2:].strip())
        elif ':' in ln and ln.count(':') == 1 and not ln.endswith(':'):
            # heuristically treat 'Name: text' as dialogue
            lhs, rhs = ln.split(':', 1)
            if lhs.isalpha():
                dialogues.append(rhs.strip())

    if len(dialogues) == 0:
        result["pass"] = False
        result["reason"] = "No dialogue lines detected to verify seven-word constraint."
        return result

    # Check each dialogue line has exactly seven words and contains no 'e'
    bad_lines = []
    for d in dialogues:
        words = [w for w in re.split(r"\s+", d) if w]
        if len(words) != 7:
            bad_lines.append({"dialogue": d, "word_count": len(words)})
        if re.search(r"[eE]", d):
            bad_lines.append({"dialogue": d, "issue": "contains 'e'"})

    if bad_lines:
        result["pass"] = False
        result["reason"] = "Some dialogue lines violate the seven-word or lipogram constraints."
        result["details"]["bad_dialogue_lines"] = bad_lines
    else:
        result["pass"] = True
        result["reason"] = "Text respects lipogram (no 'e') and each dialogue line has exactly seven words."
    return result


def evaluate_reversed_causality(text: str) -> Dict:
    result = {"pass": False, "reason": "", "details": {}}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        result["reason"] = "No paragraphs detected."
        return result

    disorder_kw = [
        "shatter", "shattered", "shards", "broken", "fragments", "dispersed", "scattered", "heat", "entropy",
        "dissipat", "chaos"
    ]
    order_kw = [
        "reassemble", "reassembled", "intact", "whole", "coalesce", "coalesced", "condense", "organized", "order",
        "decreas", "lower entropy"
    ]

    first_para = paragraphs[0].lower()
    last_para = paragraphs[-1].lower()

    has_disorder_start = any(kw in first_para for kw in disorder_kw)
    has_order_end = any(kw in last_para for kw in order_kw)

    result["details"]["has_disorder_start"] = has_disorder_start
    result["details"]["has_order_end"] = has_order_end

    if has_disorder_start and has_order_end:
        result["pass"] = True
        result["reason"] = "Starts in disorder and ends in a more ordered description (heuristic)."
    else:
        result["pass"] = False
        if not has_disorder_start and not has_order_end:
            result["reason"] = "Missing disorder at start and order at end."
        elif not has_disorder_start:
            result["reason"] = "Missing clear disorder keywords in the opening paragraph."
        else:
            result["reason"] = "Missing clear ordering keywords in the final paragraph."
    return result


def evaluate_rfc_format(text: str) -> Dict:
    result = {"pass": False, "reason": "", "details": {}}
    # Check for required RFC-like sections
    required_sections = ["Abstract", "Terminology", "Specification", "Security Considerations"]
    found_sections = [sec for sec in required_sections if re.search(rf"^\s*{re.escape(sec)}\s*[:\n]", text, flags=re.IGNORECASE | re.MULTILINE)]
    rfc2119_terms = ["MUST", "SHOULD", "MAY"]
    found_terms = [t for t in rfc2119_terms if t in text]

    result["details"]["found_sections"] = found_sections
    result["details"]["found_terms"] = found_terms

    if len(found_sections) == len(required_sections) and len(found_terms) >= 2:
        result["pass"] = True
        result["reason"] = "Required RFC sections present and RFC 2119 terms used."
    else:
        result["pass"] = False
        missing = [s for s in required_sections if s not in found_sections]
        if missing:
            result["reason"] = f"Missing sections: {missing}"
        else:
            result["reason"] = "Insufficient RFC 2119 terms (MUST/SHOULD/MAY) usage detected."
    return result


EVALUATORS = {
    "spatial_state_tracking": evaluate_spatial_tracking,
    "technical_integration": evaluate_technical_integration,
    "lipogram_strict_dialogue": evaluate_lipogram_and_dialogue,
    "reversed_causality": evaluate_reversed_causality,
    "rfc_format_emulation": evaluate_rfc_format,
}


def generate_text(model: torch.nn.Module, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 256, temperature: float = 1.0, top_k: Optional[int] = None) -> str:
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    # out_ids is a tensor of shape (batch, seq_len)
    generated = tokenizer.decode(out_ids[0].tolist())
    return generated


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on: cpu or cuda (auto-detect if omitted)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None, help="Tokenizer cache dir for transformers")
    parser.add_argument("--out", type=str, default="eval_results.json", help="Output JSON file with results")
    args = parser.parse_args(argv)

    # device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # create tokenizer
    try:
        tokenizer = create_tokenizer('gpt2', cache_dir=args.cache_dir)
    except Exception as e:
        print("Failed to create tokenizer. Ensure 'transformers' is installed.")
        raise

    # create model
    # use default config; if checkpoint exists we'll try to load weights
    model_cfg = {
        'vocab_size': getattr(tokenizer, 'vocab_size', 50257)
    }
    model = create_model(model_cfg)

    if args.checkpoint:
        loaded = safe_load_checkpoint(model, args.checkpoint, device)
        if not loaded:
            print("Continuing with randomly initialized model (checkpoint load failed or not provided).")

    results = {}

    for entry in PROMPTS:
        print(f"\n=== Running prompt: {entry['id']} ({entry['title']}) ===")
        generated = generate_text(model, tokenizer, entry['prompt'], device, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print("Generated text (truncated to 400 chars):\n", generated[:400].replace('\n', '\\n'))

        evaluator = EVALUATORS.get(entry['id'])
        if evaluator is None:
            eval_res = {"pass": False, "reason": "No evaluator implemented."}
        else:
            eval_res = evaluator(generated)

        results[entry['id']] = {
            "title": entry['title'],
            "prompt": entry['prompt'],
            "generated": generated,
            "evaluation": eval_res,
        }

    # write results to file
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote evaluation results to {args.out}")


if __name__ == '__main__':
    main()
