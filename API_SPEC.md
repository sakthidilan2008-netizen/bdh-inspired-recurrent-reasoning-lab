# API_SPEC.md — Backend API (FastAPI)

Consistent with ARCHITECTURE.md §2 pipeline: Frontend → Experiment API → Model Engine → Task Generator → Evaluation Engine → Interpretability Engine → Results Store.

## Conventions
- Base path: `/api/v1`
- All responses JSON; long-running jobs return a `job_id` immediately and are polled or streamed via WebSocket (`/api/v1/ws/jobs/{job_id}`).
- All numeric results trace to a stored results-file path, included in the response (`result_ref`).

## Endpoints

### Tasks
- `POST /tasks/generate` — body: `{family, difficulty, split, seed}` → `Task` (schemas/task.schema.json)
- `GET /tasks/families` → list of task family metadata (difficulty axis definition, description)

### Model / Runs
- `POST /runs/train` — body: config (memory_mode, reasoning_depth, task_families, ...) → `{job_id}`
- `POST /runs/evaluate` — body: `{run_id, split}` → `{job_id}`
- `GET /runs/{run_id}` → run metadata + status
- `GET /runs/{run_id}/result` → `ExperimentResult` (schemas/experiment_result.schema.json)

### Interactive Playground (single-episode, live telemetry)
- `POST /playground/episode` — body: `{family, difficulty, seed, model_config}` → runs one episode synchronously, returns prediction + `TelemetryBundle`
- `POST /playground/step` — for step-through UI: advance one reasoning iteration or one memory write, return incremental telemetry

### Interpretability
- `GET /runs/{run_id}/telemetry` → aggregated `TelemetryBundle` for a run
- `GET /runs/{run_id}/neuron/{neuron_id}` → per-neuron detail (activation frequency, selectivity)

### Failure Atlas
- `GET /failure-atlas/{task_family}` → discovered thresholds, curves, example failures, mitigation deltas
- `POST /failure-atlas/risk-estimate` — body: `{run_id, episode_id}` → `RiskReport` (SAFE/WARNING/HIGH_RISK + feature breakdown)

### Experiments / Sweeps
- `POST /experiments/sweep` — body: sweep spec matching `configs/experiments.yaml` schema → `{job_id}`
- `GET /experiments/{sweep_id}/plots` → declarative plot specs consumed by Recharts on the frontend

### Model Comparison
- `GET /baselines` → list of baseline models with param count, accuracy, latency (from stored runs only)

## Job Lifecycle
```
queued → running → {completed | failed}
```
WebSocket pushes: `{job_id, status, progress_pct, latest_metric}`.

## Error Handling
Standard `{error: {code, message}}` envelope; validation errors (e.g., malformed sweep config) return 422 with field-level detail.

## Auth
Out of scope for hackathon (local-only deployment assumed); note explicitly in LIMITATIONS.md if demoed publicly.
