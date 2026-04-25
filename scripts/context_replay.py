#!/usr/bin/env python3
"""
Small file-based retrieval/update helper for llm_context.

This intentionally uses only the Python standard library so it can run on
plain SSH machines without extra setup.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DIR = ROOT / "llm_context"
CARDS_DIR = CONTEXT_DIR / "context_cards"
MEMORY_LOG = CONTEXT_DIR / "MEMORY_LOG.md"

REQUIRED_KEYS = {
    "id",
    "title",
    "type",
    "status",
    "priority",
    "tags",
    "updated",
    "summary",
}

PRIORITY_BOOST = {
    "critical": 5,
    "high": 3,
    "medium": 1,
    "low": 0,
}


def _today() -> str:
    return _dt.date.today().isoformat()


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_./-]+", text)
        if len(token) > 1
    }


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inside = raw[1:-1].strip()
        if not inside:
            return []
        return [item.strip().strip('"').strip("'") for item in inside.split(",")]
    return raw.strip('"').strip("'")


def parse_card(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    meta: dict[str, object] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            front_matter = parts[1]
            body = parts[2]
            for line in front_matter.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                meta[key.strip()] = _parse_value(value)
    meta["_path"] = path
    meta["_body"] = body
    return meta


def iter_cards() -> list[dict]:
    if not CARDS_DIR.exists():
        return []
    return [parse_card(path) for path in sorted(CARDS_DIR.glob("*.md"))]


def score_card(card: dict, query_terms: set[str]) -> int:
    title = str(card.get("title", ""))
    summary = str(card.get("summary", ""))
    tags = card.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    body = str(card.get("_body", ""))

    tag_terms = _tokenize(" ".join(tags))
    title_terms = _tokenize(title)
    summary_terms = _tokenize(summary)
    body_terms = _tokenize(body)

    score = 0
    score += 6 * len(query_terms & tag_terms)
    score += 4 * len(query_terms & title_terms)
    score += 3 * len(query_terms & summary_terms)
    score += len(query_terms & body_terms)
    score += PRIORITY_BOOST.get(str(card.get("priority", "")).lower(), 0)
    return score


def cmd_retrieve(args: argparse.Namespace) -> int:
    query = " ".join(args.query)
    query_terms = _tokenize(query)
    ranked = sorted(
        ((score_card(card, query_terms), card) for card in iter_cards()),
        key=lambda item: (-item[0], str(item[1].get("id", ""))),
    )
    selected = [(score, card) for score, card in ranked if score > 0][: args.limit]

    print("Mandatory startup files:")
    for rel in [
        "llm_context/README.md",
        "llm_context/SYSTEM_PROMPT.md",
        "llm_context/CURRENT_STATE.md",
    ]:
        print(f"- {rel}")

    print()
    print(f"Query: {query}")
    print("Selected context cards:")
    if not selected:
        print("- No matching cards found. Read llm_context/CODE_MAP.md and inspect source files directly.")
        return 0

    for score, card in selected:
        path = Path(card["_path"]).relative_to(ROOT)
        tags = card.get("tags", [])
        tag_text = ", ".join(tags) if isinstance(tags, list) else str(tags)
        print(f"- {path} | score={score} | priority={card.get('priority')} | tags={tag_text}")
        print(f"  summary: {card.get('summary')}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    tag_filter = set(args.tag or [])
    for card in iter_cards():
        tags = card.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if tag_filter and not (set(tags) & tag_filter):
            continue
        path = Path(card["_path"]).relative_to(ROOT)
        print(f"{card.get('id')} | {card.get('priority')} | {path}")
        print(f"  tags: {', '.join(tags)}")
        print(f"  summary: {card.get('summary')}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    ok = True
    for card in iter_cards():
        missing = REQUIRED_KEYS - set(card.keys())
        path = Path(card["_path"]).relative_to(ROOT)
        if missing:
            ok = False
            print(f"FAIL {path}: missing {', '.join(sorted(missing))}")
        else:
            print(f"OK   {path}")
    return 0 if ok else 1


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "context-card"


def cmd_new(args: argparse.Namespace) -> int:
    card_id = _slug(args.id)
    path = CARDS_DIR / f"{card_id}.md"
    if path.exists() and not args.force:
        print(f"Refusing to overwrite existing card: {path.relative_to(ROOT)}", file=sys.stderr)
        return 2

    tags = ", ".join(args.tags)
    body = f"""---
id: {card_id}
title: {args.title}
type: {args.type}
status: active
priority: {args.priority}
tags: [{tags}]
updated: {_today()}
summary: {args.summary}
---

# {args.title}

## Why This Matters

{args.summary}

## Details

Add concise evidence, file pointers, and operational notes here.

## When To Read

Read this card when working on: {", ".join(args.tags)}.

## Source Pointers

- Add relevant files or commands here.
"""
    path.write_text(body, encoding="utf-8")
    print(f"Created {path.relative_to(ROOT)}")
    return 0


def cmd_append_log(args: argparse.Namespace) -> int:
    entry = f"\n## {_today()}\n\n- Actor: {args.actor}\n- Event: {args.event}\n"
    if args.evidence:
        entry += f"- Evidence: {args.evidence}\n"
    if args.follow_up:
        entry += f"- Follow-up: {args.follow_up}\n"
    with MEMORY_LOG.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    print(f"Appended to {MEMORY_LOG.relative_to(ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve and update llm_context replay-buffer cards.")
    sub = parser.add_subparsers(dest="command", required=True)

    retrieve = sub.add_parser("retrieve", help="Retrieve relevant context cards for a query.")
    retrieve.add_argument("query", nargs="+")
    retrieve.add_argument("--limit", type=int, default=5)
    retrieve.set_defaults(func=cmd_retrieve)

    list_cmd = sub.add_parser("list", help="List context cards.")
    list_cmd.add_argument("--tag", action="append", help="Only list cards with this tag. Can be repeated.")
    list_cmd.set_defaults(func=cmd_list)

    check = sub.add_parser("check", help="Validate required card front matter.")
    check.set_defaults(func=cmd_check)

    new = sub.add_parser("new", help="Create a new context card.")
    new.add_argument("--id", required=True)
    new.add_argument("--title", required=True)
    new.add_argument(
        "--type",
        required=True,
        choices=["overview", "workflow", "finding", "decision", "code-map", "prompt", "task"],
    )
    new.add_argument("--priority", default="medium", choices=["critical", "high", "medium", "low"])
    new.add_argument("--tags", nargs="+", required=True)
    new.add_argument("--summary", required=True)
    new.add_argument("--force", action="store_true")
    new.set_defaults(func=cmd_new)

    append_log = sub.add_parser("append-log", help="Append a durable memory log entry.")
    append_log.add_argument("--actor", required=True)
    append_log.add_argument("--event", required=True)
    append_log.add_argument("--evidence")
    append_log.add_argument("--follow-up")
    append_log.set_defaults(func=cmd_append_log)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
