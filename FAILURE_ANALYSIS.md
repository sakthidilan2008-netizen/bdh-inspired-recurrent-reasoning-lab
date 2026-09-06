# FAILURE_ANALYSIS.md — Failure Atlas Methodology

## 1. Purpose

Turn every experiment into a structured failure record, and build a **pre-decode failure risk estimator** so the system can anticipate, not just report, failure.

## 2. Per-Experiment Record (written for every run)

```
input_complexity → internal_activation_stats → memory_state_summary →
latent_trajectory_summary → output → error_type → failure_threshold_estimate →
mitigation_attempted → improvement_delta
```

`error_type` is classified into a small fixed taxonomy (extend as needed, log new types explicitly rather than silently bucketing as "other"):
- wrong-transformation-type
- correct-type-wrong-cells
- boundary/edge error
- color confusion
- memory-interference-induced (a stored demonstration was overwritten/corrupted)
- non-convergence (reasoning trajectory didn't stabilize within R)
- OOD-distribution-mismatch

## 3. Failure Risk Estimator (pre-decode)

Inputs (all available *before* the decoder runs):
- memory saturation (state norm / effective rank vs. calibrated healthy range)
- interference score (MEMORY_SPEC.md §3)
- latent convergence trend (are `Δ_r` decreasing across iterations so far?)
- activation collapse (sparsity ratio near 100% — dead computation)
- complexity mismatch (current input's difficulty features vs. training distribution range)

```
risk = g(saturation, interference, non_convergence, activation_collapse, complexity_mismatch)
label ∈ {SAFE, WARNING, HIGH_RISK}
```

`g` is a small logistic/gradient-boosted classifier **fit on validation-split outcomes** (features above → actual correct/incorrect), never on test/OOD labels. Calibration curve (predicted risk vs. observed error rate) must be reported — an uncalibrated risk score is not a valid contribution.

## 4. Use of Risk Signal

If `HIGH_RISK`, the system may (config-gated, itself ablated per TRAINING_SPEC.md §8):
- allocate extra reasoning iterations (REASONING_SPEC.md §7)
- switch memory read to an `interference_mitigated` view
- flag the prediction as low-confidence in the UI (never silently "fix" without disclosure)

## 5. Failure Atlas (UI-facing artifact)

For each discovered threshold (EXPERIMENT_PROTOCOL.md §6), the Atlas shows:
- the accuracy-vs-difficulty curve with the fitted threshold marked
- example failing inputs near the threshold
- the classified error type breakdown at/above threshold
- which mitigation (if any) was attempted and the resulting accuracy delta
- the calibration curve of the risk estimator at that threshold region

## 6. Non-Negotiable Rules

- Never hard-code a threshold value in the frontend; always read from the stored sweep result.
- Never report an "improvement" without a paired baseline (no-mitigation) run at the same difficulty and seed set.
- Error taxonomy is fixed at experiment-design time, not adjusted post-hoc to make a family look better.
