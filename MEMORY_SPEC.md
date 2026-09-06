# MEMORY_SPEC.md — Recurrent Associative Memory

## 1. General Form

```
S_t = decay(λ) ⊙ S_{t-1} + plasticity(η) · update(x_t, y_t)
```

This is our **explicit simplification**, not an assertion of the exact BDH-CQ internal equation (which is not publicly specified in full). λ, η may be scalar, per-neuron, or learned; config-selectable.

`S_t ∈ R^{n×n}` (or `R^{n×d}` for a rectangular associative bank) is the memory tensor; `Read`/`Write` operate on it.

## 2. MemoryMode Implementations

Each mode must be independently unit-testable (shape, determinism, no-NaN) per TESTING_SPEC.md.

### 2.1 `baseline`
No memory — reasoning engine reads a zero tensor. Used as the floor/control condition, not as a claim about BDH.

### 2.2 `simple_hebbian`
```
update(x_t, y_t) = x_t ⊗ y_t        (outer product)
S_t = λ S_{t-1} + η (x_t ⊗ y_t)
Read(S, q) = S · q
```
Publicly-referenced Hebbian associative principle; simplest working implementation.

### 2.3 `fast_weight_associative`
Adds a normalization term to prevent unbounded growth:
```
S_t = λ S_{t-1} + η (x_t ⊗ y_t) / (‖x_t‖‖y_t‖ + ε)
```

### 2.4 `normalized_associative`
Row-normalizes `S_t` after each write (`S_t ← S_t / max(1, ‖S_t‖_row)`), directly targeting activation-saturation failure mode.

### 2.5 `interference_mitigated` **[original contribution, see INNOVATION.md]**
Composes:
- **orthogonalization**: project new write onto the complement of the top-k principal directions of `S_{t-1}` before adding, reducing overlap with existing stored patterns
- **concept-aware gating**: a learned gate `g(x_t)` down-weights writes highly similar (cosine) to recent writes, computed via a small ring-buffer of recent keys
- **selective plasticity**: `η` becomes `η · (1 - similarity_to_recent)`

```
ŵ_t = (x_t ⊗ y_t) − Proj_{top-k(S_{t-1})}(x_t ⊗ y_t)
S_t = λ S_{t-1} + η · g(x_t) · ŵ_t
```

## 3. Diagnostics (shared across all modes)

| Metric | Definition |
|---|---|
| state norm | `‖S_t‖_F` |
| effective rank | number of singular values of `S_t` above a threshold fraction of the max |
| entropy | Shannon entropy of normalized singular value spectrum |
| overlap | mean pairwise cosine similarity of a sample of stored key vectors |
| interference score | drop in reconstruction accuracy for an earlier-written key after N subsequent writes |
| reconstruction error | `‖Read(S_t, key_i) − value_i‖` for previously written `(key_i, value_i)` |
| activation collision | fraction of neuron indices simultaneously active across two different stored patterns |
| sparsity degradation | change in downstream neuron-layer sparsity ratio as memory fills |

## 4. Module Interface

```python
class MemoryModule(Protocol):
    mode: MemoryMode
    def reset(self, batch_size: int) -> MemoryState: ...
    def write(self, state: MemoryState, x: Tensor, y: Tensor) -> MemoryState: ...
    def read(self, state: MemoryState, query: Tensor) -> Tensor: ...
    def diagnostics(self, state: MemoryState) -> MemoryDiagnostics: ...

@dataclass
class MemoryDiagnostics:
    state_norm: float
    effective_rank: float
    entropy: float
    overlap: float
    interference_score: float
    reconstruction_error: float
    activation_collision: float
    sparsity_degradation: float
```

## 5. Numerical Stability

- λ constrained to `[0, 1)` via sigmoid reparameterization.
- η constrained non-negative via softplus.
- All writes accumulated in float32 even under mixed precision elsewhere.
- Row-norm clipping in `normalized_associative` and `interference_mitigated` prevents divergence over long demonstration sequences (validated by determinism/no-NaN tests, TESTING_SPEC.md).

## 6. Comparison Protocol

Every experiment that varies memory mode must otherwise hold task, model size, and training budget constant (see EXPERIMENT_PROTOCOL.md §3) so that `BASELINE vs MITIGATED` comparisons in INNOVATION.md are causally interpretable.
