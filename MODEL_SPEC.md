# MODEL_SPEC.md — Neuron-Level Components

Consistent with ARCHITECTURE.md §3–4. Every component below is labeled by provenance: **[PUBLIC]** = publicly documented BDH concept we reproduce; **[OURS]** = our design choice made where public material is silent or where we simplify for tractable scale.

## 1. Sparse Positive Neuron Layer **[PUBLIC, simplified]**

```
x_t = σ_sparse(W_in e_t + b_in)
```
- `e_t ∈ R^d_in`: input embedding at step t
- `W_in ∈ R^{n×d_in}`, `b_in ∈ R^n`, n = neuron population size
- `σ_sparse`: one of `ReLU`, `ReLU²` (published BDH material describes non-negative sparse activations; we support both ReLU and ReLU² and let experiments select via config, rather than asserting one is "the" official function)
- Optional top-k masking for hard sparsity control (`k` a config parameter): keeps only the top-k activations per sample, zeroing the rest.

**Trainable params**: `W_in, b_in`. **Init**: Kaiming-uniform for `W_in`, zeros for `b_in`.
**Numerical stability**: activations clipped at a configurable ceiling; ReLU² pre-scaled by `1/sqrt(n)` to avoid blow-up.
**Complexity**: O(n·d_in) per step.
**Interpretability hook**: raw `x_t` and post-mask `x_t` both logged (pre/post sparsity).

## 2. Local Neuron Connectivity Graph **[PUBLIC concept, OURS at implementation scale]**

A fixed or learned sparse adjacency structure `A ∈ {0,1}^{n×n}` restricting which neurons can influence each other in the recurrent update, rather than full dense n×n interaction. We implement:
- `block_sparse`: neurons grouped into blocks, dense within block, sparse across (config: `n_blocks`, `block_size`)
- `small_world`: Watts-Strogatz-style graph as a topology option for ablation

This is explicitly our simplified stand-in for the graph/network connectivity described publicly; the exact BDH connectivity pattern at production scale is not reproduced.

## 3. Excitatory/Inhibitory Grouping **[PARTIAL]**

Neurons partitioned into excitatory (E) and inhibitory (I) subsets at a fixed ratio (config, default 80/20). Outgoing weights from I-neurons are constrained non-positive via `-softplus(w)` reparameterization. This is a partial nod to Dale's-law-style dynamics referenced in public discussion, not a full biological simulation — labeled as such in LIMITATIONS.md.

## 4. Hebbian / Fast-Weight Memory **[PUBLIC concept]**

See MEMORY_SPEC.md for full treatment. The neuron layer exposes `(x_t, y_t)` pairs (pre/post activation) as the write signal into memory.

## 5. Recurrent State Update **[PUBLIC]**

```
u_t = U · x_t + V · Read(S_{t-1})
h_t = LayerNorm(h_{t-1} + u_t)
```
`U, V` learned projections. Residual + norm for stability across long demonstration sequences.

## 6. Parameterized Neuron Projections **[PUBLIC]**

Low-rank in/out projections `W_in = A_in B_in` (rank r ≪ n) to keep parameter count tractable at hackathon scale, config-selectable rank.

## 7. Latent Iterative Computation **[OURS, see REASONING_SPEC.md]**

## 8. Decoder **[OURS]**

Task-appropriate:
- Grid tasks (ARC-style): a small deconvolution/MLP head producing a `H×W×C` categorical grid, cross-entropy per cell.
- Sequence tasks: pointer/softmax head over a fixed vocabulary of grid symbols.

**Trainable params**: decoder head weights only (encoder/reasoning trunk shared).

## 9. Activation Instrumentation **[OURS, original]**

Every forward pass records, without altering the computation graph (detached, no-grad copies):
- pre/post activation tensors per layer
- sparsity ratio (`fraction of x_t == 0`)
- per-neuron running activation frequency (EMA)
- graph-local activation clusters

This feeds directly into INTERPRETABILITY_SPEC.md and must be a **hook**, not a rewrite of the forward pass, so it can be toggled off for speed.

## Module Interface (Python-level contract)

```python
class NeuronLayer(nn.Module):
    def forward(self, e_t: Tensor, memory_read: Tensor) -> tuple[Tensor, ActivationRecord]: ...

@dataclass
class ActivationRecord:
    pre: Tensor
    post: Tensor
    sparsity: float
    neuron_freq_ema: Tensor
```

## Shapes Summary

| Tensor | Shape |
|---|---|
| `e_t` | `[B, d_in]` |
| `x_t` | `[B, n]` |
| `h_t` | `[B, d_h]` |
| memory read | `[B, d_h]` |
| decoder output (grid task) | `[B, H, W, C]` |
