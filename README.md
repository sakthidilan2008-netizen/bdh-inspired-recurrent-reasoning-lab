# BDH Research Laboratory
### A transparent, experimental platform for studying sparse recurrent memory + latent reasoning
**DataForge 2026 — Pathway-sponsored track**

> We implement a transparent experimental reproduction of publicly documented BDH/BDH-CQ principles, systematically evaluate its behavior, and contribute original mechanisms for interpretability, failure prediction, and interference mitigation. We do not claim to reproduce Pathway's proprietary production system.

## What this is

A research instrument, not a demo. It lets you:

- Feed demonstrations into a sparse, recurrent associative memory and watch it adapt.
- Run an ARC-style structured-reasoning benchmark through a latent iterative reasoning engine (no visible chain-of-thought — internal computation only).
- Inspect neuron-level activation, specialization, and memory interference via an interpretability engine.
- Automatically discover — not assume — the complexity thresholds where the architecture fails, and test mitigations against them.
- Compare against lightweight baselines (MLP, RNN, GRU, attention) on identical tasks.

## Why

To empirically investigate: *can a sparse, recurrent, continuously-adapting architecture reason in-context, efficiently and more interpretably than autoregressive text CoT?*

## Layers

- **Memory** — pluggable associative memory (`MemoryMode`: baseline, simple Hebbian, fast-weight, normalized, interference-mitigated)
- **Reasoning** — configurable-depth latent refinement engine with adaptive halting
- **Interpretability** — activation/memory/trajectory telemetry, never verbalized reasoning

## Repository map

See `ARCHITECTURE.md` for the authoritative structure. Key docs:

| Doc | Purpose |
|---|---|
| ARCHITECTURE.md | Source of truth for the whole system |
| MODEL_SPEC.md | Neuron/component math and interfaces |
| MEMORY_SPEC.md | Memory modes, equations, interference metrics |
| REASONING_SPEC.md | Latent reasoning engine |
| BENCHMARK_SPEC.md | Task families, splits, generation rules |
| EXPERIMENT_PROTOCOL.md | The experiment matrix and how failure thresholds are discovered |
| FAILURE_ANALYSIS.md | Failure Atlas methodology |
| INNOVATION.md | What is actually novel here vs. reproduced |
| TESTING_SPEC.md | Acceptance criteria |
| LIMITATIONS.md | What we do NOT claim |
| DEMO_SCRIPT.md | Hackathon demo walkthrough |
| JUDGES_QA.md | Anticipated judge questions |

## Quickstart (for the implementing agent / future contributors)

```
backend/  — Python, PyTorch, FastAPI
frontend/ — React, TypeScript, Tailwind, Recharts
```

Build order and dependency graph: see ARCHITECTURE.md §7.

## Status

This is a specification package. Implementation is performed by a subsequent coding agent using these files as the contract.
