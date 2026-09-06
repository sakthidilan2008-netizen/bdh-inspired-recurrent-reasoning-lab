# JUDGES_QA.md — Anticipated Questions

**Q: Is this the actual BDH-CQ architecture?**
A: No — we reproduce publicly documented principles at experimental scale and are explicit about every simplification (LIMITATIONS.md). We do not have access to Pathway's proprietary system.

**Q: How do you know your "failure thresholds" are real and not overfit to your own generator?**
A: Thresholds are pre-registered (the crossing criterion is fixed before viewing the sweep, EXPERIMENT_PROTOCOL.md §6), computed across multiple seeds with reported variance, and validated on a held-out/OOD split never used for tuning.

**Q: Isn't "no chain-of-thought" just hiding the reasoning?**
A: We expose quantitative telemetry (activation, memory, trajectory metrics) sufficient to audit computation — INTERPRETABILITY_SPEC.md. It's a different kind of transparency than verbalized CoT, not an absence of transparency; we say explicitly what it does and doesn't show.

**Q: How is this different from just an RNN with attention-like memory?**
A: The sparse positive-activation constraint, Hebbian/fast-weight memory update, and iterative (non-token-by-token) latent refinement are the specific structural choices we test against RNN/GRU/attention baselines head-to-head (EXPERIMENT_PROTOCOL.md §7) — the baseline comparison table makes the difference falsifiable rather than asserted.

**Q: What's your actual accuracy?**
A: Reported per split (train/val/held-out/OOD) from stored evaluation runs only — see the Benchmark Center, never a hard-coded number. We target 89–95%+ where technically achievable and report honestly where we fall short.

**Q: What would you build next with more time?**
A: The extended experiment grid (all 11 task families × all memory modes), a learned (not logistic) failure-risk estimator, and validation against the official ARC-AGI dataset for external comparability.

**Q: Why should we trust your interference/failure metrics aren't just noise?**
A: Every claimed improvement has a paired ablation (same seeds, same difficulty, mitigation on vs off) — TRAINING_SPEC.md §8 — and calibration curves are reported for the risk estimator (FAILURE_ANALYSIS.md §3).
