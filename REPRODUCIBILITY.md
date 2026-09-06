# REPRODUCIBILITY.md — Experiment Logging & Reproducibility

## Every Stored Experiment Record Must Include

```
experiment_id (uuid)
timestamp (ISO 8601)
git/version identifier (commit hash or spec-version tag)
random_seed
configuration (full resolved config, not just diffs from default)
model_parameters (param count, checkpoint reference)
dataset_generation_parameters (family, difficulty, split seed ranges)
metrics (full MetricBundle per split)
failure_statistics (error-type breakdown, per FAILURE_ANALYSIS.md)
latency (per-sample and per-batch)
memory_usage (peak, avg)
```

Schema: `schemas/experiment_result.schema.json`.

## Storage

- Primary store: append-only JSON Lines file per run under `experiments/{experiment_id}/result.jsonl`, plus a SQLite index (`experiments/index.db`) for fast querying by the API/frontend.
- Exportable as JSON (raw) and CSV (flattened metrics table) via `GET /runs/{run_id}/result?format=csv`.

## Rules

1. No metric may be reported in the frontend or in documentation without a `result_ref` pointing to a stored file.
2. Config is stored fully resolved (post-defaults-merge) so a run can be reproduced without needing the exact `configs/default.yaml` state at the time (which may drift).
3. Git commit hash (or, if git unavailable in the execution environment, a content hash of `backend/`) is mandatory — ties every number to an exact code version.
4. Re-running the same `(config, seed)` pair must reproduce metrics within a documented tolerance (exact for CPU-deterministic ops; small tolerance for GPU nondeterminism, explicitly stated per metric).
