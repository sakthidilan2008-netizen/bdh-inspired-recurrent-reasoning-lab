# INTERPRETABILITY_SPEC.md — Reasoning Telemetry System

**Core design rule: expose observable computational telemetry, never verbalized/natural-language chain-of-thought.** No component in this spec may generate free-text explanations of "why" the model answered as it did beyond quantitative attributions defined here.

## 1. Neuron-Level Metrics

| Metric | Definition |
|---|---|
| activation frequency | EMA of `1[x_i > 0]` per neuron across samples |
| activation sparsity | fraction of zero activations per forward pass |
| positive activation rate | mean activation magnitude among active neurons |
| task selectivity | mutual information (or simple ANOVA F-stat) between neuron activation and task label |
| concept selectivity | same, against a labeled concept taxonomy defined per task family |
| specialization score | `max_task(mean activation | task) − mean_task(mean activation | task)`, normalized |

## 2. Memory-Level Metrics

Reuses MEMORY_SPEC.md §3 diagnostics (state norm, effective rank, entropy, overlap, interference, reconstruction error, collision, sparsity degradation), plus temporal evolution: each metric logged at every write step to produce a time series per run.

## 3. Latent Reasoning Metrics

Reuses REASONING_SPEC.md §3–4: convergence deltas, per-iteration activation change, halting step distribution. Additionally:
- **trajectory clustering**: k-means or UMAP+HDBSCAN over PCA-reduced trajectories across many samples, to check whether reasoning states cluster by task family or difficulty (a genuine interpretability question, results reported honestly whether or not clean clusters emerge).

## 4. Outputs / Visualizations

| Output | Source |
|---|---|
| neuron heatmap | activation frequency / specialization per neuron, sortable by task |
| state heatmap | `S_t` matrix visualization (down-sampled if large) |
| latent trajectory plot | 2D/3D projection (PCA/UMAP) of `H_0..H_R` |
| task-neuron matrix | neurons × tasks selectivity matrix |
| specialization matrix | neurons × concepts |
| interference graph | nodes = stored patterns, edges weighted by overlap/interference |
| convergence curve | Δ_r vs. r |
| confidence + competing scores | decoder logits margin between top-2 predictions |

## 5. Module Interface

```python
class InterpretabilityEngine(Protocol):
    def collect(self, trace: ReasoningTrace, memory_diag: MemoryDiagnostics,
                activation_records: list[ActivationRecord]) -> TelemetryBundle: ...

@dataclass
class TelemetryBundle:
    neuron_metrics: NeuronMetrics
    memory_metrics: MemoryDiagnostics
    reasoning_metrics: ReasoningMetrics
    figures: dict[str, FigureSpec]   # declarative spec, rendered by frontend
```

## 6. Performance Note

Full telemetry collection adds overhead; provide a `telemetry_level: {off, light, full}` config. `light` = scalar metrics only (no full tensors retained); `full` = trajectory + activation tensors retained for the sampled deep-dive runs referenced in REASONING_SPEC.md §4.

## 7. Non-Goals

- No natural-language rationale generation.
- No exposure of raw internal float tensors to the frontend without going through the declared metric functions above (prevents "reasoning leakage" via raw dumps that could be mistaken for a CoT trace).
