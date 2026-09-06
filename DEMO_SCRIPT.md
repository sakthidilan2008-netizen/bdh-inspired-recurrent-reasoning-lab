# DEMO_SCRIPT.md — 3–5 Minute Hackathon Demonstration

**Narrative thesis to state up front:** "Don't just show that the architecture works — show how it works, where it breaks, and how we make it more robust."

1. **Show a demonstration-based ARC task** in the Playground (BENCHMARK_SPEC.md family K, demonstration-conditioned).
2. **Show demonstrations entering memory** — Memory Laboratory view, `S_t` updating live as each demonstration streams in.
3. **Show neuron activation/sparsity** — Neuron Observatory heatmap lighting up per demonstration.
4. **Show memory state changing** — diagnostics panel (state norm, effective rank) ticking as writes accumulate.
5. **Show latent computation progressing** — Reasoning Laboratory, step through `H_0..H_R`, convergence curve dropping.
6. **Show prediction** — decoded grid rendered next to the true target (if in a validated split) with confidence/margin shown.
7. **Increase task complexity** — bump nesting depth or sequence length live via the difficulty slider.
8. **Show first signs of interference** — interference score climbing in Memory Lab, risk estimator flipping to WARNING.
9. **Enable the mitigation mechanism** — switch `MemoryMode` to `interference_mitigated` live.
10. **Show improved result** — same task instance, side-by-side before/after accuracy and interference score.
11. **Open the Failure Atlas** — show the empirically discovered threshold curve for this task family.
12. **Explain what failed, why, and how the system adapted** — point at the specific error_type classification and the risk-triggered extra computation, tying back to INNOVATION.md's contributions.

**Close** with the positioning statement (ARCHITECTURE.md §0) and one sentence on what's next (extended grid, more task families) — signals honesty about scope, which plays well with technical judges.
