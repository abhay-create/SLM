---
id: project-overview
title: Project Overview
type: overview
status: active
priority: high
tags: [overview, architecture, curriculum, expansion, slm]
updated: 2026-04-25
summary: The repo trains a small decoder-only language model through staged curriculum learning and function-preserving expansion.
---

# Project Overview

This repository contains an SLM curriculum and expansion project. The model begins as a small decoder-only Transformer trained on TinyStories and is expanded through staged curricula toward more complex narrative domains such as ROCStories, SimpleStories, Children Stories, SimpleWiki, and WritingPrompts.

Core model design:

- decoder-only Transformer
- RMSNorm
- SwiGLU FFN
- learnable positional embeddings or RoPE support
- tied token embedding and LM head
- base shape around 6 layers, `d_model=512`, `d_ff=2048`, context 256, tokenizer vocab 40,000

Expansion strategy:

- depth expansion clones selected layers and adds Gaussian noise
- FFN width expansion zero-pads `w_down` so new neurons start dormant
- context expansion interpolates learned positional embeddings

When to read:

- onboarding to the repo
- deciding which module owns a task
- explaining the project to another model or human

Source pointers:

- `README.md`
- `src/model.py`
- `expand_model.py`
- `configs/expansion_stages.yaml`
