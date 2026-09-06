# EXPERIMENT_PROTOCOL.md — Experiment Matrix & Threshold Discovery

## 1. Principle

Failure thresholds (e.g. "sequence length > 6 fails," "nesting depth ≥ 5 fails") are **hypotheses to test, not facts to hard-code**. Every sweep below must run difficulty as a fine-grained ordered axis (e.g., length 2,3,4,...,20) and the empirical transition point is read off the resulting accuracy curve, then reported with its own confidence interval (across seeds).

## 2. Experiment Matrix (axes)

```
task_family × difficulty × memory_mode × memory_size × plasticity(η) × decay(λ)
  × reasoning_depth × sparsity(top-k) × demonstration_count × distribution(in/OOD)
```

## 3. Default Grid (hackathon-feasible)

To keep compute bounded, `configs/experiments.yaml` defines a **reduced default grid**:
- 3 task families for full sweep depth (one from each of: spatial [C/D/E], structural [H], compositional [J])
- difficulty axis: 8 points per family, log/linear spaced
- memory_mode: `{simple_hebbian, interference_mitigated}` (the causal comparison pair)
- reasoning_depth: `{1, 2, 4, 8}`
- 3 random seeds per configuration (for CI bars)

An **extended grid** (all families, all memory modes, denser difficulty, more seeds) is opt-in via `configs/experiments.yaml: profile: extended` for teams with more compute/time.

## 4. Held-Constant Rule

Any single-axis sweep (e.g., memory_size) must hold all other axes fixed at their default-grid values, so that plots (§5) are causally interpretable one axis at a time. Multi-axis interaction studies are a separate, explicitly labeled experiment set.

## 5. Required Automatic Plots

- accuracy vs difficulty (per family)
- accuracy vs memory size
- accuracy vs recurrent depth
- interference vs sequence length
- sparsity vs task complexity
- latency vs reasoning depth
- memory footprint vs context length
- error rate vs number of demonstrations

All plots generated from stored results files (REPRODUCIBILITY.md), never hand-entered.

## 6. Threshold Discovery Procedure

1. Run the fine-grained difficulty sweep for a family/config.
2. Fit a simple monotonic curve (isotonic regression or logistic fit) to accuracy vs. difficulty.
3. Report the difficulty value at which accuracy crosses a pre-registered threshold (default 50% of in-distribution peak accuracy) — pre-registered *before* looking at the data, to avoid post-hoc cherry-picking.
4. Repeat across seeds; report threshold ± std.
5. Feed the discovered threshold into FAILURE_ANALYSIS.md as the empirical failure boundary — explicitly superseding any earlier hard-coded prototype assumption.

## 7. Baseline Comparison Protocol

Run the identical task/eval protocol against each baseline in BENCHMARK_SPEC.md-compatible harness (MLP, RNN, GRU, attention baseline, fixed-memory baseline, BDH-inspired baseline, improved BDH-inspired model). Match parameter count as closely as feasible per baseline and report the actual parameter count alongside accuracy (never omit it) — see model comparison table in INNOVATION.md.
