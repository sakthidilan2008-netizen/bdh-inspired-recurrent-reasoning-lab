# INNOVATION.md — What Is Actually Novel Here

Per ARCHITECTURE.md §3, most low-level neuron/memory/recurrence mechanics are **reproductions** of publicly documented ideas at experimental scale. The novel contributions of this project are:

## 1. Observable Computational Telemetry (vs. verbalized CoT)
A structured, non-textual instrumentation layer (INTERPRETABILITY_SPEC.md) that makes internal computation auditable without ever generating natural-language "thoughts." This is a genuine design choice distinct from both black-box latent reasoning and textual CoT approaches.

## 2. Interference-Mitigated Memory (`interference_mitigated` MemoryMode)
Orthogonalized writes + concept-aware gating + selective plasticity, evaluated head-to-head against a plain Hebbian baseline (MEMORY_SPEC.md §2.5) with the specific measurable quantities in §3 (overlap, interference score, reconstruction error, etc.).

## 3. Pre-Decode Failure Risk Estimation
Most interpretability/robustness work characterizes failure *after the fact*. This system fits a calibrated estimator that flags risk *before* decoding and can trigger adaptive extra computation (FAILURE_ANALYSIS.md §3–4) — turning failure analysis into an online adaptive mechanism rather than a purely post-hoc report.

## 4. Empirical (not assumed) Failure-Threshold Discovery
Complexity thresholds are discovered via a pre-registered curve-fitting procedure (EXPERIMENT_PROTOCOL.md §6) rather than hard-coded from prototype intuition — a methodological improvement in scientific rigor over the earlier prototype this project supersedes.

## 5. Reasoning-Depth-as-Experiment, not Reasoning-Depth-as-Assumption
Explicit sweep of recurrent depth with accuracy/latency/convergence reported jointly (REASONING_SPEC.md §2, §6), avoiding the common unexamined assumption that "more iterations = better reasoning."

## Minimum-Viable-Novelty Cut (if time-constrained)
If the hackathon team must cut scope, preserve, in priority order:
1. Interpretability telemetry (Layer C) — cheapest to demo, highest visual impact
2. `simple_hebbian` vs `interference_mitigated` comparison (Layer A)
3. Empirical threshold discovery for one task family (nesting or sequence length)
4. Pre-decode risk estimator (can be a simplified logistic model if a full classifier is too time-costly)

Baseline comparisons (MLP/RNN/GRU/attention) and the full 11-task-family benchmark are **valuable but cuttable** — they support the "atlas" framing but are not the core novel claim.

## What This Project Does NOT Claim
See LIMITATIONS.md. In particular: no claim of reproducing Pathway's proprietary BDH-CQ system, no fabricated benchmark numbers, no invented equations presented as official BDH mechanisms.
