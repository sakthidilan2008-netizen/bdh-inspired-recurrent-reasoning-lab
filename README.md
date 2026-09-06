# BDH-Inspired Recurrent Reasoning Lab

An interactive research and learning artifact for exploring **sparse recurrent computation, associative memory, latent iterative reasoning, interpretability telemetry, and failure analysis** through ARC-style structured reasoning tasks.

> **Important:** This project is an educational/research-scale implementation inspired by publicly documented ideas from the Dragon Hatchling (BDH) and BDH-CQ work. It is **not a reproduction of Pathway's proprietary production system**, nor does it claim to reproduce Pathway's internal benchmark results.

---

## 1. Project Overview

Many modern language models rely heavily on attention-based sequence processing, while recent research explores recurrent and state-based alternatives.

This project builds a **controlled experimental environment** for studying ideas including:

- sparse neural activity,
- recurrent computation,
- associative/fast-weight memory,
- local interactions,
- test-time memory updates,
- iterative latent computation,
- and computation that does not require exposing a textual chain-of-thought.

The system allows a learner or researcher to:

1. Provide ARC-style demonstrations.
2. Provide a query grid.
3. Select a memory mechanism.
4. Change recurrent reasoning depth.
5. Run the reasoning engine.
6. Inspect the predicted transformation.
7. Examine memory and latent-state telemetry.
8. Evaluate the prediction against generated ground truth.
9. Explore benchmark and failure-analysis behaviour.

The goal is not simply to build a demo, but to make the computational process **experimentally inspectable**.

---

## 2. Research Question

The central research question is:

> **Can a sparse recurrent architecture with associative memory and latent iterative computation learn and apply structured transformations from demonstrations while remaining computationally inspectable and avoiding the need for verbalized chain-of-thought?**

A secondary question is:

> **Where does such an architecture fail as task complexity increases, and can those failure modes be detected or mitigated?**

---

## 3. What This Project Is — and Is Not

### This project IS

- An educational research instrument.
- An experimental implementation inspired by publicly documented BDH/BDH-CQ concepts.
- A controlled environment for studying recurrent memory and latent reasoning.
- An interactive ARC-style reasoning playground.
- A platform for empirical failure analysis.
- A reproducible benchmark environment using generated structured tasks.

### This project IS NOT

- Pathway's production BDH implementation.
- Pathway's proprietary continual-learning system.
- A reproduction of the internal BDH-CQ implementation.
- A claim of reproducing Pathway's internal benchmark scores.
- A claim of achieving the published BDH-CQ ARC-AGI performance.
- A clinical or production AI system.

Where this project introduces mechanisms not explicitly described in the public BDH/BDH-CQ materials, they are identified as **project-level experimental mechanisms**.

---

## 4. Intended Learner

This artifact is designed for learners and researchers with a basic understanding of:

- Python
- NumPy/PyTorch
- neural networks
- matrix operations
- recurrent neural networks
- machine learning evaluation
- basic algorithmic reasoning

Prior knowledge of BDH or ARC is helpful but not required.

---

## 5. Conceptual Background

### 5.1 Dragon Hatchling (BDH)

The Dragon Hatchling research explores an alternative approach to language-model architecture involving ideas such as:

- sparse positive neural activations,
- local neuron interactions,
- recurrent computation,
- associative/fast-weight memory,
- and biologically inspired computation.

The public BDH repository exposes selected research components and experimental code.

This project uses publicly documented concepts as inspiration while implementing a smaller experimental architecture.

**Primary reference:**
The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain — arXiv:2509.26507
https://arxiv.org/abs/2509.26507

### 5.2 BDH-CQ

BDH-CQ explores **in-context learning with recurrent latent reasoning**.

Instead of requiring a model to generate a long natural-language chain of thought, the system performs repeated computation over a recurrent latent state.

Conceptually:

\[
H_{r+1}=F_\theta(H_r,S_K)
\]

where:

- \(H_r\) is the latent workspace at reasoning step \(r\),
- \(S_K\) is the memory state after processing demonstrations,
- \(F_\theta\) is the recurrent computation.

This project uses a simplified version of this idea for structured ARC-style tasks.

**Primary reference:**
BDH-CQ: In-Context Learning with Recurrent Latent Reasoning — arXiv:2608.09888
https://arxiv.org/abs/2608.09888

---

## 6. System Architecture

The experimental pipeline is:

```text
                 ┌─────────────────────┐
                 │   React Playground  │
                 │                     │
                 │ Grid / Parameters  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      FastAPI        │
                 │     Backend API     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Task Generator    │
                 │                     │
                 │ ARC-style Tasks     │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌──────────────────────────────┐
              │        Model Engine          │
              │                              │
              │  Encoder                     │
              │      ↓                       │
              │  Sparse Neuron Layer         │
              │      ↓                       │
              │  Associative Memory          │
              │      ↓                       │
              │  Latent Reasoning            │
              │      ↓                       │
              │  Decoder / Hypothesis Engine │
              └──────────────┬───────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │       Evaluation        │
                │                         │
                │ Exact Match             │
                │ Cell Accuracy           │
                │ Telemetry               │
                └────────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      ┌────────────────┐             ┌────────────────┐
      │ Interpretability│             │ Failure Analysis│
      │    Telemetry    │             │     / Risk      │
      └────────────────┘             └────────────────┘
```

---

## 7. Core Computational Components

### 7.1 Demonstration-Conditioned Memory

Demonstrations are processed sequentially and used to update an associative memory state.

A simplified memory update is:

\[
S_t=(1-\gamma)S_{t-1}+\eta(A_tB_t^T)
\]

where:

- \(S_t\) = memory state at step \(t\)
- \(\gamma\) = decay factor
- \(\eta\) = plasticity/update rate
- \(A_t,B_t\) = encoded demonstration representations

The purpose of this component is to investigate how information from earlier demonstrations can influence later computation without requiring a conventional growing key-value cache.

### 7.2 Recurrent Latent Reasoning

After demonstrations have been processed, the query is encoded and passed through a recurrent latent workspace.

The simplified computation is:

\[
H_{r+1}=F_\theta(H_r,S_K)
\]

The playground allows the reasoning depth to be changed experimentally.

For example:

```text
Depth = 1
Depth = 4
Depth = 8
Depth = 16
```

This allows experiments investigating whether additional latent computation improves structured reasoning.

---

## 8. Sparse Neural Computation

The model includes a sparse neuron layer using positive nonlinear activations.

The experimental neuron layer provides telemetry related to:

- activation sparsity,
- neuron activity,
- layer-level activation behaviour,
- and specialization-related statistics.

The goal is to make the behaviour of computational units more inspectable than a conventional opaque end-to-end prediction.

---

## 9. Interpretability Without Verbalized Chain-of-Thought

A key design choice of this project is:

> **Do not require the system to expose a natural-language chain of thought.**

Instead, the project exposes computational telemetry such as:

- memory norm,
- effective rank,
- entropy,
- interference-related metrics,
- reconstruction error,
- activation sparsity,
- recurrent reasoning steps,
- halting step,
- latent trajectory information.

These are **diagnostic measurements**, not human-readable reasoning traces.

They should not be interpreted as a literal transcript of the model's internal reasoning.

---

## 10. Transformation Hypothesis Engine

The project includes a transformation hypothesis engine for ARC-style tasks.

Instead of directly accessing the target answer during prediction, the engine:

1. Generates candidate transformations.
2. Applies each candidate to the demonstrations.
3. Compares the generated outputs with the demonstration targets.
4. Measures consistency.
5. Rejects inconsistent transformations.
6. Ranks remaining candidates.
7. Applies the selected transformation to the query.

Conceptually:

```text
Demonstrations
      │
      ▼
Candidate Transformations
      │
      ▼
Apply to Demonstrations
      │
      ▼
Consistency Verification
      │
      ├── inconsistent ──► reject
      │
      ▼
Rank Consistent Candidates
      │
      ▼
Apply to Query
      │
      ▼
Predicted Output
```

The current experimental implementation includes:

- translation,
- color transformation,
- boundary propagation.

This component is a **project-level mechanism** for structured transformation discovery and should not be interpreted as an official component of Pathway's BDH-CQ architecture.

---

## 11. ARC-Style Task Families

The controlled playground currently focuses on:

### Boundary Propagation

A colored cell located on a boundary propagates its color into the corresponding row or column according to the generated task rule.

### Color Transformation

A transformation maps one grid color to another according to the demonstrations.

### Translation

Objects or patterns are shifted according to the transformation demonstrated by the examples.

The benchmark infrastructure is designed to support additional task families as the project develops.

---

## 12. Interactive Playground

The frontend provides an interactive research playground.

Users can modify:

- task family,
- memory mode,
- reasoning depth,
- number of demonstrations,
- grid size,
- random seed,
- adaptive halting,
- memory rereading behaviour.

The interface displays:

### Input

- demonstration grids,
- demonstration outputs,
- query grid.

### Prediction

- predicted query output,
- selected transformation,
- transformation consistency information.

### Evaluation

- exact-match accuracy,
- cell-level accuracy,
- correct cells,
- total cells.

### Telemetry

- memory norm,
- effective rank,
- entropy,
- interference-related diagnostics,
- activation sparsity,
- reasoning steps,
- halting step.

This makes the experiment directly observable rather than presenting only a final prediction.

---

## 13. Experimental Controls

The playground is parameterized so that learners can change experimental assumptions.

Example:

```text
Task:
Boundary Propagation

Memory:
Simple Hebbian

Reasoning Depth:
8

Demonstrations:
3

Grid:
5 × 5

Seed:
42
```

Changing these parameters produces a different experimental episode.

This provides the required interactive element: the learner changes an input or computational parameter and observes the resulting output and telemetry.

---

## 14. Controlled Evaluation

A controlled benchmark was implemented to establish a reproducible baseline.

| Parameter | Configuration |
|---|---:|
| Task families | 3 |
| Memory modes | 2 |
| Reasoning depths | 3 |
| Random seeds | 20 |
| Grid size | 5 × 5 |
| Demonstrations | 3 |
| Total episodes | 360 |

### Task families

- Boundary Propagation
- Color Transformation
- Translation

### Memory configurations

- Baseline
- Simple Hebbian

### Reasoning depths

- 1
- 4
- 8

---

## 15. Verified Benchmark Result

The corrected controlled evaluation produced:

```text
Episodes:       360
Exact matches:  360
Exact rate:     100%
Mean accuracy:  100%
```

All 360 episodes produced exact grid matches under the controlled benchmark configuration.

The complete machine-readable run is stored in:

```text
results/controlled_evaluation.csv
```

---

## 16. How to Interpret the 100% Result

The 100% result must be interpreted carefully.

It means:

> **The current implementation solved all episodes in the defined controlled benchmark.**

It does **not** mean:

- 100% performance on ARC-AGI.
- 100% performance on BDH-CQ.
- 100% performance on Pathway's internal benchmarks.
- human-level general reasoning.
- generalization to arbitrary unseen reasoning problems.

The benchmark currently contains a relatively small number of structured task families and is therefore not sufficiently difficult to establish broad superiority.

In particular, the current benchmark is **saturated**: the tested configurations all achieve the same perfect result.

Therefore:

> **The current experiments do not provide evidence that increasing reasoning depth or changing memory mode improves accuracy on this benchmark.**

This is an important experimental observation and motivates the next stage of benchmark development.

---

## 17. Failure Analysis

One of the project's research directions is to discover failure boundaries empirically rather than hard-code them.

For example, instead of assuming:

```text
Sequence length > 6 → failure
```

the experiment should sweep increasing complexity and observe where performance begins to degrade.

The exact threshold is expected to depend on:

- task family,
- memory size,
- reasoning depth,
- number of demonstrations,
- grid size,
- random seed,
- and model configuration.

Therefore, thresholds should be reported as **experiment-specific observations**, not universal architectural limits.

---

## 18. Debugging as a Scientific Experiment

During development, the initial Boundary Propagation benchmark showed reduced accuracy.

The investigation revealed an ambiguity in how corner cells were interpreted. The task generator and transformation implementation could disagree about whether certain corner cells should trigger vertical or horizontal propagation.

The benchmark was corrected so that the generator and transformation definition share the same boundary semantics.

After correction:

```text
Boundary Propagation
20 seeds
20/20 exact matches
100% exact accuracy
```

This debugging process illustrates why benchmark generation and evaluation logic must be independently inspected before attributing observed failures to the model architecture.

---

## 19. Reproducibility

The controlled evaluation uses explicit random seeds.

Current configuration:

```text
Seeds:
1–20

Grid:
5 × 5

Demonstrations:
3

Reasoning depths:
1, 4, 8

Memory:
baseline, simple_hebbian
```

The resulting measurements are written to:

```text
results/controlled_evaluation.csv
```

This allows the benchmark to be rerun and inspected.

---

## 20. Installation

### Requirements

Recommended environment:

- Python 3.10+
- Node.js
- npm
- Git
- Windows, Linux, or macOS

### Backend setup

Create a Python virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 21. Run the Backend

From the repository root:

```bash
python -m uvicorn backend.api.main:app --reload
```

The API will be available at the local address displayed by Uvicorn.

---

## 22. Run the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local address displayed by Vite.

---

## 23. Run the Tests

From the repository root:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

The test suite covers core functionality including:

- engine behaviour,
- numerical consistency,
- tensor dimensions and shapes,
- deterministic execution,
- evaluation behaviour,
- model components,
- reasoning components,
- and related integration checks.

---

## 24. Run the Controlled Evaluation

From the repository root:

```powershell
.venv\Scripts\python.exe -m backend.experiments.controlled_evaluation
```

The experiment writes its results to:

```text
results/controlled_evaluation.csv
```

---

## 25. Repository Structure

```text
DATAKGP/
│
├── backend/
│   ├── api/
│   │   └── ...
│   │
│   ├── engine/
│   │   ├── base.py
│   │   ├── model.py
│   │   ├── neurons.py
│   │   │
│   │   ├── baselines/
│   │   │   └── models.py
│   │   │
│   │   ├── decoder/
│   │   │   └── hypothesis.py
│   │   │
│   │   ├── memory/
│   │   │   ├── implementations.py
│   │   │   └── mitigated.py
│   │   │
│   │   └── reasoning/
│   │       └── core.py
│   │
│   ├── eval/
│   │   └── metrics.py
│   │
│   ├── experiments/
│   │   ├── run.py
│   │   └── controlled_evaluation.py
│   │
│   ├── failure/
│   │   └── engine.py
│   │
│   ├── interpretability/
│   │   └── engine.py
│   │
│   ├── tasks/
│   │   ├── base.py
│   │   └── generators.py
│   │
│   └── schemas.py
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   │       └── ARCPlayground.tsx
│   ├── package.json
│   └── ...
│
├── configs/
│   ├── default.yaml
│   └── experiments.yaml
│
├── results/
│   ├── controlled_evaluation.csv
│   ├── ablation_memory_reasoning.csv
│   └── experiments/
│
├── schemas/
│   ├── task.schema.json
│   └── experiment_result.schema.json
│
├── tests/
│   ├── unit/
│   └── ...
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 26. Configuration

Default experimental parameters are maintained under:

```text
configs/default.yaml
```

Experiment configurations are maintained under:

```text
configs/experiments.yaml
```

Important parameters include:

- random seed,
- neuron count,
- input dimension,
- model dimension,
- memory size,
- memory decay,
- plasticity,
- sparsity configuration,
- reasoning depth,
- adaptive halting,
- and related experimental settings.

---

## 27. Metrics

### Exact Match

Whether the complete predicted grid is identical to the ground-truth grid.

\[
ExactMatch =
\begin{cases}
1 & \text{if prediction = target}\\
0 & \text{otherwise}
\end{cases}
\]

### Cell Accuracy

The fraction of cells predicted correctly.

\[
Accuracy =
\frac{\text{Correct Cells}}
{\text{Total Cells}}
\]

### Telemetry

Additional diagnostics include measurements such as:

- memory norm,
- effective rank,
- entropy,
- interference score,
- reconstruction error,
- mean activation sparsity,
- reasoning steps,
- halting step.

---

## 28. Original Project Contributions

The project combines publicly documented research ideas with project-level experimental mechanisms.

### 1. Interactive Research Playground

A browser-based interface allowing users to manipulate reasoning parameters and inspect results.

### 2. Latent Telemetry

The project exposes internal computational diagnostics instead of generating a natural-language chain-of-thought.

### 3. Empirical Failure-Boundary Discovery

Complexity boundaries are intended to be discovered experimentally through controlled sweeps rather than assumed in advance.

### 4. Transformation Consistency Verification

Candidate transformations are tested against demonstrations before being applied to the query.

### 5. Interference-Oriented Memory Diagnostics

Memory behaviour is monitored using metrics such as overlap, rank, entropy, reconstruction behaviour, and interference-related measurements.

### 6. Reproducible Controlled Evaluation

The system includes seeded experiments and machine-readable experiment outputs.

---

## 29. Comparison With Conventional Approaches

The project is intended to make architectural differences experimentally visible.

| Approach | Main computation | Persistent state | Reasoning style |
|---|---|---|---|
| MLP | Feed-forward | No | Single-pass |
| RNN | Recurrent hidden state | Yes | Sequential |
| GRU | Gated recurrence | Yes | Sequential |
| Attention model | Attention over context | Context-dependent | Token/context based |
| This project | Sparse + recurrent + associative memory | Recurrent memory | Latent iterative |

This is a conceptual comparison, not evidence that this project outperforms these architectures.

---

## 30. Limitations

### Scale

The implementation is research-scale and substantially smaller than production-scale language models.

### Benchmark Scope

The controlled evaluation currently covers a limited number of generated task families.

### Benchmark Saturation

The current 360-episode benchmark is solved perfectly, so it cannot distinguish the tested configurations effectively.

### Generalization

Performance on generated tasks does not establish generalization to arbitrary ARC-AGI tasks.

### Proprietary Components

The project does not have access to Pathway's proprietary internal implementation or training infrastructure.

### Interpretability

Telemetry provides quantitative diagnostics but does not constitute a complete semantic explanation of the model's computation.

### Failure Prediction

Failure-risk mechanisms should be validated on sufficiently difficult and independently generated tasks before making strong predictive claims.

---

## 31. Future Work

The next experimental stages include:

1. Expand the ARC task families.
2. Introduce more difficult multi-step transformations.
3. Add held-out and out-of-distribution tasks.
4. Perform systematic reasoning-depth sweeps.
5. Compare additional memory mechanisms.
6. Compare against lightweight neural baselines.
7. Improve calibration of failure-risk estimation.
8. Investigate memory interference under increasing demonstration counts.
9. Add larger grids.
10. Evaluate on external/public ARC datasets.
11. Perform multi-seed statistical analysis.
12. Investigate compute and memory scaling.
13. Improve the failure atlas using measurements generated directly from experiments.

The long-term goal is to determine whether recurrent latent computation and associative memory provide useful advantages for structured reasoning under controlled experimental conditions.

---

## 32. Scientific Integrity

This project follows several reporting principles:

- Experimental results are reported with their benchmark scope.
- Internal/proprietary Pathway results are not represented as reproduced results.
- Generated benchmark results are not presented as external benchmark results.
- Failure thresholds are not presented as universal facts.
- Diagnostic telemetry is not described as literal chain-of-thought.
- Perfect performance on a limited benchmark is not treated as proof of general intelligence.
- Future experiments are clearly separated from verified current results.

---

## 33. Key Public References

1. **The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain**
   arXiv:2509.26507
   https://arxiv.org/abs/2509.26507

2. **BDH-CQ: In-Context Learning with Recurrent Latent Reasoning**
   arXiv:2608.09888
   https://arxiv.org/abs/2608.09888

3. **Titans: Learning to Memorize at Test Time**
   arXiv:2501.00663
   https://arxiv.org/abs/2501.00663

4. **Hungry Hungry Hippos: Towards Language Modeling with State Space Models**
   arXiv:2212.14052
   https://arxiv.org/abs/2212.14052

### Public implementation reference

**Pathway BDH repository:**
https://github.com/pathwaycom/bdh

**Kharagpur Data Science Hackathon 2026 / Pathway:**
https://pathway.com/events/kharagpur-data-science-hackathon-2026

---

## 34. Source and License Record

The project should maintain a clear record of:

- original source code,
- third-party libraries,
- datasets,
- pretrained weights, if any,
- graphics,
- fonts,
- icons,
- reused UI components,
- and external research material.

### Project code

The project source code is released under the license specified in `LICENSE`.

### Third-party software

Third-party dependencies remain subject to their respective licenses.

### Research papers

Research papers are cited for scientific reference and are not redistributed as project-owned content unless their respective licenses permit redistribution.

---

## 35. AI Assistance Disclosure

AI tools were used during development for activities including:

- code generation assistance,
- debugging assistance,
- documentation drafting,
- architecture discussion,
- experiment design discussion,
- and explanation of technical concepts.

AI-assisted material was reviewed and integrated by the project team.

Experimental results reported in this README correspond to executions of the project code and the stored benchmark output.

---

## 36. Hackathon Scope

This project was developed as an experimental learning artifact for the **Kharagpur Data Science Hackathon 2026 / DataForge context**.

The artifact is designed to be:

- explorable,
- computationally interactive,
- reproducible,
- supported by source code,
- and accompanied by technical documentation.

The browser playground provides the interactive component, while the repository, benchmark results, documentation, and experiment scripts provide the supporting research material.

---

## 37. Recommended Demo Flow

For a short presentation:

### Step 1 — Introduce the problem

Explain the motivation for studying recurrent latent reasoning and associative memory.

### Step 2 — Show a demonstration

Provide ARC-style examples.

### Step 3 — Change reasoning depth

Run:

```text
Depth = 1
```

Then:

```text
Depth = 8
```

Compare the prediction and telemetry.

### Step 4 — Change memory mode

Compare:

```text
Baseline
```

against:

```text
Simple Hebbian
```

### Step 5 — Inspect telemetry

Show:

- memory state,
- sparsity,
- effective rank,
- entropy,
- reasoning steps.

### Step 6 — Explain the hypothesis engine

Show how candidate transformations are checked against demonstrations.

### Step 7 — Show benchmark results

Explain:

```text
360 controlled episodes
360 exact matches
100% exact-match rate
```

Then clarify that this is a controlled generated benchmark, not an ARC-AGI result.

### Step 8 — Explain the research direction

The next goal is to make the benchmark difficult enough to expose meaningful failure boundaries and determine whether memory or deeper latent reasoning provides measurable benefits.

---

## 38. Project Status

### Current status

**Functional experimental prototype**

Implemented:

- Interactive frontend
- FastAPI backend
- ARC-style task generation
- Recurrent latent reasoning
- Sparse neural computation
- Associative memory
- Transformation hypothesis engine
- Evaluation metrics
- Interpretability telemetry
- Controlled experiment runner
- Reproducible seeded evaluation
- Automated tests

### Current verified result

```text
Controlled benchmark:
360 / 360 exact matches
100% exact-match rate
```

### Current research limitation

The present benchmark is too easy to establish meaningful architectural superiority because all tested configurations reach the same result.

Therefore, the next research stage is **difficulty expansion and failure-boundary discovery**, rather than claiming that one configuration is superior.

---

## 39. Final Takeaway

This project does not attempt to claim:

> **"We reproduced BDH-CQ."**

Instead, it asks:

> **What can we learn by implementing publicly described principles of sparse recurrent computation, associative memory, and latent reasoning in a transparent, controllable environment?**

The resulting system provides an interactive laboratory where researchers can manipulate:

```text
Memory
   +
Sparse Computation
   +
Latent Recurrence
   +
Structured Transformations
   +
Interpretability Telemetry
   +
Failure Analysis
```

and observe the resulting behaviour.

The current benchmark establishes that the implementation can solve its controlled task suite reliably. The more important next step is to increase task difficulty, introduce held-out/generalization tests, and determine where the architecture succeeds, where it fails, and why.

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

---

## Project Links

Before final submission, add the public deployment and repository URLs:

- **Live Demo:** `YOUR_PUBLIC_DEMO_URL`
- **Source Code:** `YOUR_PUBLIC_GITHUB_URL`
- **Technical Blog / Report:** `YOUR_PUBLIC_REPORT_URL`
