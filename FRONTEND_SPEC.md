# FRONTEND_SPEC.md — Research Dashboard

Stack: React + TypeScript + Tailwind + Canvas/SVG (grid rendering) + Recharts (or equivalent) for charts. Talks only to API_SPEC.md endpoints — no client-side fabrication of metrics.

## Sections

1. **Overview** — project thesis, positioning statement (ARCHITECTURE.md §0) always visible in footer, links to all sections.
2. **ARC Playground** — pick a task family/difficulty, watch demonstrations feed in, see live prediction. Uses `/playground/episode` + `/playground/step`.
3. **Memory Laboratory** — visualize `S_t` evolution, diagnostics time series, compare `MemoryMode`s side by side on the same episode.
4. **Latent Reasoning Laboratory** — step through `H_0..H_R`, convergence curve, adaptive-halt visualization, trajectory projection plot.
5. **Neuron Observatory** — neuron heatmap, task-neuron matrix, specialization matrix, searchable/sortable by metric.
6. **Failure Atlas** — per-family discovered threshold curves, example failures, mitigation before/after.
7. **Benchmark Center** — full metrics table (exact/cell/task-level/transformation/structural/calibration/robustness/OOD) per run, filterable.
8. **Ablation Laboratory** — pick two runs differing in one config axis, auto-generate the comparison plot (TRAINING_SPEC.md §8 requirement made visible).
9. **Model Comparison** — baseline table (accuracy, params, latency, interpretability metrics) from `/baselines`.
10. **Research Report** — auto-compiled summary of a chosen run/sweep exportable as JSON/CSV (and optionally referencing docx/pdf skill if a shareable write-up is requested downstream — not part of this spec's scope).

## Reproduce-an-Experiment Requirement
Every experiment card/table row must expose a "Reproduce" action that re-POSTs the exact stored config (seed, splits, hyperparameters) to `/runs/train` or `/experiments/sweep`, guaranteeing the UI never shows an unreproducible number.

## Visual Design Notes
- Use the `frontend-design` skill guidance for typography/aesthetic choices when actually implementing — avoid a generic "AI dashboard template" look; this is a research instrument, should read as such (dense information, clear hierarchy, not decorative).
- Grid tasks rendered via Canvas/SVG with a consistent color palette across the whole app (same palette used in ARC playground, neuron heatmaps, memory state).
- Loading/long-running sweeps show live progress via the WebSocket job channel (API_SPEC.md).

## State Management
Server is the source of truth for all metrics; frontend holds only UI/view state (selected run, selected neuron, chart zoom, etc.) — never caches a computed accuracy number outside of what the API returned for a specific `result_ref`.
