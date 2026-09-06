# ARCHITECTURE.md — Authoritative System Specification
## BDH / BDH-CQ Research Laboratory — DataForge 2026

> **Positioning statement (must appear verbatim or paraphrased close to this in every doc and in the UI footer):**
> "We implement a transparent experimental reproduction of publicly documented BDH/BDH-CQ principles, systematically evaluate its behavior, and contribute original mechanisms for interpretability, failure prediction, and interference mitigation. We do not claim to reproduce Pathway's proprietary production system."

This file is the single source of truth. Every other spec file must be consistent with it; where a conflict exists, this file wins and the other file must be corrected.

---

## 1. System Thesis

Can a sparse, recurrent, continuously-adapting neural architecture perform structured in-context reasoning efficiently while remaining more interpretable than conventional autoregressive reasoning systems (which rely on verbalized chain-of-thought)?

The system is organized into three conceptual layers, each independently testable:

- **Layer A — Memory**: a recurrent/associative store that ingests demonstrations sequentially and is queried at inference time.
- **Layer B — Reasoning**: an iterative latent-refinement engine that computes internally (no natural-language CoT) before decoding.
- **Layer C — Interpretability**: instrumentation that turns every forward pass into recorded telemetry (activations, memory state, trajectories) usable for analysis and for the Failure Atlas.

The project is a **research instrument**, not a demo: every experiment must produce a structured, reproducible record (see REPRODUCIBILITY.md).

---

## 2. High-Level Data / Control Flow

```
Frontend (React/TS)
      │  REST/WebSocket
      ▼
Experiment API (FastAPI)
      │
      ├─► Task Generator ──► Task/Dataset objects (train/val/held-out/OOD)
      │
      ├─► Model Engine
      │        ├─ Memory Module (pluggable MemoryMode)
      │        ├─ Reasoning Engine (latent iterative refinement)
      │        ├─ Decoder
      │        └─ Instrumentation Hooks ──► Interpretability Engine
      │
      ├─► Evaluation Engine (metrics, no label leakage)
      │
      ├─► Failure Risk Estimator (pre-decode) + Failure Atlas (post-hoc)
      │
      └─► Results Store (JSON/CSV files + SQLite index)
```

Every arrow is a **module contract** (see §7). No module may reach across more than one layer without going through its declared interface — this is what keeps the system swappable (e.g., baseline RNN vs BDH-inspired model) and testable in isolation.

---

## 3. Component Inventory (BDH-style, public-concepts only)

| # | Component | Status |
|---|---|---|
| 1 | Sparse positive neuron activations (ReLU/ReLU²-style, non-negative) | Reproduced from public description |
| 2 | Locally-interacting neuron computation (sparse connectivity graph) | Reproduced, simplified scale |
| 3 | Graph/network connectivity structure | Reproduced, simplified (fixed small-world / block-sparse graph, not the full biological-inspired graph) |
| 4 | Excitatory/inhibitory dynamics | Partial — modeled via signed synapse groups, not a full Dale's-law simulation |
| 5 | Hebbian / fast-weight working memory | Reproduced (outer-product update rule), configurable |
| 6 | Recurrent state updates | Reproduced |
| 7 | Parameterized neuron projections (low-rank in/out projections) | Reproduced |
| 8 | Latent iterative computation ("thinking in latent space") | Our design, inspired by publicly described recurrent depth concept |
| 9 | Decoder | Our design (task-appropriate: grid/sequence decoder) |
| 10 | Activation instrumentation | Our original contribution |

Anything not in this table (e.g., exact BDH-CQ query-routing internals, proprietary scaling recipes) is explicitly **out of scope** and must not be fabricated — see LIMITATIONS.md.

---

## 4. Critical Equations (see MODEL_SPEC.md / MEMORY_SPEC.md / REASONING_SPEC.md for full derivations)

**Neuron activation (sparse, non-negative):**
```
x_t = σ_sparse(W_in · e_t + b_in),   σ_sparse ∈ {ReLU, top-k ReLU, ReLU²}
```

**Generalized memory update (MemoryMode-dependent — see MEMORY_SPEC.md):**
```
S_t = decay(λ) ⊙ S_{t-1} + plasticity(η) · update(x_t, y_t)
```
`update()` varies by mode (outer product for Hebbian, normalized/orthogonalized variants for mitigation modes). λ and η are learned or scheduled, never hard-coded as "the" BDH equation — they are our explicit simplification, labeled as such.

**Latent reasoning recursion:**
```
H_0 = Encoder(query, Read(S))
H_{r+1} = ReasoningBlock(H_r, S) = Norm(H_r + f(H_r, S))
prediction = Decode(H_R)   [R fixed or adaptive-halt]
```

**Failure risk score (pre-decode):**
```
risk = g(saturation(S), interference(S), Δ‖H_r‖ non-convergence, activation_collapse)
```
`g` is a calibrated logistic combiner trained/fitted on labeled failure outcomes from validation runs — not a guess at inference time on test labels.

---

## 5. Module Contracts (interfaces only — implementations live in code, not here)

```python
class MemoryModule(Protocol):
    def reset(self, batch_size: int) -> MemoryState: ...
    def write(self, state: MemoryState, x: Tensor, y: Tensor | None) -> MemoryState: ...
    def read(self, state: MemoryState, query: Tensor) -> Tensor: ...
    def diagnostics(self, state: MemoryState) -> MemoryDiagnostics: ...

class ReasoningEngine(Protocol):
    def forward(self, query: Tensor, memory: MemoryState, max_iters: int) -> ReasoningTrace: ...
    # ReasoningTrace includes per-iteration H_r, convergence delta, halting step

class TaskGenerator(Protocol):
    def sample(self, split: Split, difficulty: DifficultyConfig, seed: int) -> Task: ...

class Evaluator(Protocol):
    def score(self, prediction, target) -> MetricBundle: ...
    # must never receive test-split labels during model selection

class InterpretabilityEngine(Protocol):
    def collect(self, trace: ReasoningTrace, memory: MemoryState) -> TelemetryBundle: ...

class FailureRiskEstimator(Protocol):
    def estimate(self, memory: MemoryState, partial_trace: ReasoningTrace) -> RiskReport: ...
```

Full type definitions belong in `schemas/` and in code docstrings — this file only fixes the contract shape.

---

## 6. Exact Folder / File Architecture

```
bdh-lab/
├── README.md
├── ARCHITECTURE.md                # this file, authoritative
├── MODEL_SPEC.md
├── MEMORY_SPEC.md
├── REASONING_SPEC.md
├── INTERPRETABILITY_SPEC.md
├── BENCHMARK_SPEC.md
├── TRAINING_SPEC.md
├── EXPERIMENT_PROTOCOL.md
├── FAILURE_ANALYSIS.md
├── INNOVATION.md
├── API_SPEC.md
├── FRONTEND_SPEC.md
├── TESTING_SPEC.md
├── REPRODUCIBILITY.md
├── LIMITATIONS.md
├── DEMO_SCRIPT.md
├── JUDGES_QA.md
├── configs/
│   ├── default.yaml
│   └── experiments.yaml
├── schemas/
│   ├── experiment_result.schema.json
│   └── task.schema.json
├── backend/                       # (to be built by implementing agent)
│   ├── api/                       # FastAPI routers matching API_SPEC.md
│   ├── engine/
│   │   ├── memory/                # one file per MemoryMode
│   │   ├── reasoning/
│   │   ├── decoder/
│   │   └── baselines/             # MLP, RNN, GRU, attention baseline
│   ├── tasks/                     # ARC-style generators, one per task family
│   ├── eval/
│   ├── interpretability/
│   ├── failure/
│   └── storage/                   # results DB + file store
├── frontend/                      # React/TS app matching FRONTEND_SPEC.md
├── experiments/                   # generated run artifacts (gitignored)
└── tests/                         # matching TESTING_SPEC.md
```

---

## 7. Implementation Dependency Graph (build order for the next agent)

1. `schemas/` (data contracts first — everything else depends on these)
2. `backend/tasks/` (task generator; needed before any model work, to enable TDD)
3. `backend/engine/memory/` (baseline + Hebbian first; mitigated modes after)
4. `backend/engine/reasoning/`
5. `backend/engine/decoder/`
6. `backend/engine/baselines/` (can be parallelized with 3–5)
7. `backend/eval/`
8. `backend/interpretability/`
9. `backend/failure/`
10. `backend/api/`
11. `backend/storage/`
12. `frontend/` (last; depends on a stable API contract)

---

## 8. Risks

- **Scope risk**: the full spec is large; a hackathon team must ruthlessly prioritize Layers A+B+C over exhaustive baseline coverage. See INNOVATION.md for the minimum-viable-novelty cut.
- **Fabrication risk**: pressure to report "impressive" numbers. Mitigated by TESTING_SPEC.md acceptance criteria and REPRODUCIBILITY.md logging — every number must trace to a stored run.
- **Overclaiming risk**: must not imply proprietary BDH-CQ reproduction. Mitigated by the positioning statement appearing in README, LIMITATIONS.md, and the frontend footer.
- **Compute risk**: recurrent-depth sweep + memory-size sweep + task-family sweep is combinatorially large. EXPERIMENT_PROTOCOL.md defines a reduced default grid with an "extended" opt-in grid.

---

## 9. Instructions for the Implementing Agent (e.g., Antigravity)

1. Read this file fully before touching code.
2. Implement `schemas/` types first; treat them as contracts, not suggestions.
3. Build task generators before any model code — they're the test fixtures.
4. Implement `MemoryMode.baseline` and `MemoryMode.simple_hebbian` first; get the full pipeline (task → memory → reasoning → decode → eval) working end-to-end on the easiest task family before adding the other memory modes or baselines.
5. Add instrumentation hooks from day one (do not retrofit) — Interpretability and Failure Atlas both depend on hooks existing in the forward pass.
6. Do not hard-code failure thresholds (e.g. "length > 6 fails"); the sweep in EXPERIMENT_PROTOCOL.md must discover them empirically and the Failure Atlas must render the discovered curve.
7. Every reported accuracy number must come from `backend/eval/` reading a stored results file — never inline/hard-coded in the frontend.
8. Follow TESTING_SPEC.md acceptance criteria before marking any module "done."
9. Keep the positioning statement (§0) visible in the UI and in README.md.
