# REASONING_SPEC.md — Latent Iterative Reasoning Engine

**[OURS]** — inspired by the general publicly-discussed idea that BDH-style architectures can perform internal iterative computation rather than token-by-token autoregression, but the specific recursion below is our design.

## 1. Recursion

```
H_0 = Encoder(query, Read(S))
H_{r+1} = Norm(H_r + ReasoningBlock(H_r, S))
prediction = Decode(H_R)
```

`ReasoningBlock` is a small residual MLP/gated block operating on `H_r ∈ R^{B×d_h}`, optionally re-reading memory `S` at each iteration (config: `reread_memory: bool`).

## 2. Depth Control

- **fixed-depth**: `R` is a config constant; used as the default for controlled ablations.
- **adaptive halting**: a per-sample halting probability `p_r = σ(W_h H_r)` (ACT-style); accumulate until threshold or `R_max`; report both mean halting step and its variance. Ponder cost added to loss as a config-weighted regularizer.

The framework must **not assume deeper is better** — REASONING_SPEC experiments (EXPERIMENT_PROTOCOL.md) sweep `R ∈ {1,2,4,8,16}` and report accuracy vs. R explicitly; the "useful depth" is an empirical finding, reported in results, not asserted here.

## 3. Stability Features

- Pre-norm residual connections at every iteration.
- Recurrent dropout (applied to `H_r` update, not to memory) — config-toggleable, default off (many recurrent-depth studies show dropout can hurt convergence measurement; ablation required before enabling by default).
- Gradient clipping (global norm) — required for depths ≥ 8 to avoid instability, per TRAINING_SPEC.md.
- Convergence measurement: `Δ_r = ‖H_{r+1} − H_r‖ / ‖H_r‖`, logged every iteration; used both as an interpretability signal and as an internal halting criterion candidate.

## 4. Latent Trajectory Recording

For every reasoning pass (train and eval), record:
```
trajectory = [H_0, H_1, ..., H_R]
convergence_deltas = [Δ_1, ..., Δ_R]
halting_step (if adaptive)
```
Stored at reduced precision (float16) and optionally PCA-compressed (config `trajectory_compress_dim`) to bound storage cost across large sweeps — full precision only retained for a sampled subset of runs used in the interpretability deep-dive.

## 5. Module Interface

```python
class ReasoningEngine(Protocol):
    def forward(self, query: Tensor, memory: MemoryState, cfg: ReasoningConfig) -> ReasoningTrace: ...

@dataclass
class ReasoningTrace:
    H: list[Tensor]              # H_0..H_R
    convergence_deltas: list[float]
    halting_step: int | None
    final_prediction: Tensor
```

## 6. Complexity

O(R · cost(ReasoningBlock)); ReasoningBlock cost is O(d_h²) for the MLP variant. Reported per-sample latency vs. R is part of the standard experiment output (see EXPERIMENT_PROTOCOL.md, BENCHMARK_SPEC.md metrics table).

## 7. Interaction with Failure Risk Estimator

If FAILURE_ANALYSIS.md's pre-decode risk estimator reports `WARNING` or `HIGH RISK`, the engine may (config-gated) allocate additional iterations up to `R_max_extended` or switch to an `interference_mitigated` memory read — this is the "adaptive reasoning system" referenced in INNOVATION.md. This must be evaluated as its own ablation (risk-triggered extra compute vs. fixed compute) rather than assumed beneficial.
