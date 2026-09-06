# TRAINING_SPEC.md — Training Strategy & Accuracy Improvement Pipeline

## 1. Base Training Loop

Standard supervised training on generated tasks: cross-entropy over grid cells (+ optional ponder-cost regularizer for adaptive-halting reasoning, REASONING_SPEC.md §2). Optimizer: AdamW. LR schedule: cosine with warmup, config-driven. Gradient clipping mandatory when `reasoning_depth >= 8`.

## 2. Curriculum & Sampling

- **curriculum learning**: start training distribution biased toward low-difficulty samples, anneal toward full difficulty range over training (schedule config-driven).
- **task-balanced sampling**: ensure no task family dominates gradient updates (default: uniform-per-family batch composition).
- **hard-example mining**: periodically re-weight sampling toward validation-identified low-accuracy difficulty buckets.

## 3. Augmentation

- data augmentation via generator re-sampling (cheap — regenerate rather than transform, since generation is a pure function)
- permutation augmentation (color-index permutation on grid tasks that are color-agnostic by construction)
- spatial transformations (rotation/reflection) — only applied where task semantics are invariant to them (must not corrupt tasks whose *answer* is itself a rotation/reflection)
- color remapping — analogous caveat
- noise robustness training (small cell-flip noise injected during training, evaluated in BENCHMARK_SPEC.md robustness metric)

## 4. Normalization & Stability

- state normalization (MEMORY_SPEC.md `normalized_associative`)
- activation normalization (MODEL_SPEC.md numerical stability)
- adaptive plasticity (η scheduled down over a demonstration sequence to reduce late-sequence overwriting)

## 5. Memory-Specific Improvements

- memory gating, interference suppression, selective plasticity — all implemented as `interference_mitigated` MemoryMode (MEMORY_SPEC.md §2.5), evaluated via ablation, not simply enabled.

## 6. Reasoning-Specific Improvements

- adaptive recurrent depth (REASONING_SPEC.md §2)
- confidence-based extra computation (only extra iterations when top-2 logit margin is below threshold)
- self-consistency (sample multiple reasoning trajectories via stochastic components — e.g. dropout-on — and majority-vote decode)
- candidate decoding (beam-style top-k grid hypotheses, re-ranked by a cheap consistency check against demonstrations)

## 7. Loss / Optimization

- loss reweighting per task-family difficulty (to prevent easy tasks dominating loss)
- early stopping on validation task-level accuracy, patience config-driven
- hyperparameter search: small grid/random search over (lr, η, λ, reasoning depth, memory size) — logged as its own experiment set, not hand-tuned to the held-out/OOD splits

## 8. Ablation Requirement

**Every item in §3–7 must have a paired ablation run (on/off, or mode A vs B) recorded via REPRODUCIBILITY.md before being adopted as a default.** No improvement is claimed without an ablation entry in `experiments/`.

## 9. Ensembles

Ensembling across independently-trained seeds is permitted as a comparison point (INNOVATION.md/baseline table) but must be clearly labeled as an ensemble result, distinct from single-model numbers, and justified (i.e., not used to inflate the headline single-model accuracy).
