# LIMITATIONS.md — What This Project Does Not Claim

- **Not Pathway's proprietary system.** We reproduce publicly documented BDH/BDH-CQ principles at experimental scale. We do not have access to, and do not claim to reproduce, Pathway's internal production architecture, exact hyperparameters, proprietary training data, or unpublished equations.
- **Simplified connectivity.** The neuron connectivity graph (MODEL_SPEC.md §2) is our own tractable stand-in (block-sparse / small-world), not the exact BDH connectivity pattern at production scale.
- **Partial excitatory/inhibitory modeling.** We approximate Dale's-law-style dynamics via signed weight groups; this is not a full biophysical simulation.
- **Memory equation is our simplification.** The general decay/plasticity memory update (MEMORY_SPEC.md §1) is explicitly labeled as our simplification, not an assertion of BDH-CQ's internal formulation.
- **Scale.** All experiments run at hackathon-feasible scale (small grids, small neuron populations, limited training compute). Findings about failure thresholds, interference, and depth-vs-accuracy trade-offs are scoped to this scale and may not extrapolate to production-scale systems.
- **Benchmark is self-defined.** The ARC-style task suite (BENCHMARK_SPEC.md) is our own generator-based benchmark, not the official ARC-AGI dataset, and results are not directly comparable to ARC-AGI leaderboard numbers unless explicitly re-validated against that dataset.
- **No natural-language interpretability.** The interpretability engine intentionally does not produce human-readable rationales; it produces quantitative telemetry only. This is a design choice, not a claim that the model "explains itself" in the colloquial sense.
- **No authentication/production hardening.** The API/frontend are built for local hackathon demonstration, not production deployment.
- **Failure thresholds are empirical findings at our scale**, discovered via the procedure in EXPERIMENT_PROTOCOL.md §6 — they are not claimed as universal properties of BDH-style architectures.
