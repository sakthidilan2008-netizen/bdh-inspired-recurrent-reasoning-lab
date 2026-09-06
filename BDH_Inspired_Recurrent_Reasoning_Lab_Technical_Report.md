# BDH-Inspired Recurrent Reasoning Lab
## Technical Report and Explorable Learning Artifact

### Abstract

This project presents an interactive, research-scale laboratory for studying sparse recurrent computation, associative memory, latent iterative reasoning, interpretability telemetry, and structured transformation reasoning.

The system is inspired by publicly documented ideas from the Dragon Hatchling (BDH) and BDH-CQ research, but it is explicitly **not a reproduction of Pathway's proprietary production system or internal benchmark results**.

The artifact combines a React-based interactive playground with a FastAPI backend, generated ARC-style tasks, recurrent latent computation, associative memory, a transformation hypothesis engine, quantitative telemetry, and reproducible controlled evaluation.

A corrected controlled benchmark containing 360 episodes achieved 360/360 exact matches. This result is deliberately scoped to the project's generated benchmark and is not presented as ARC-AGI, BDH-CQ, or Pathway performance.

---

## 1. Motivation

Many modern language models rely heavily on attention-based sequence processing. Recent research also explores recurrent, state-based, and memory-oriented approaches that can process information through persistent internal state.

BDH introduces a biologically inspired architecture involving locally interacting neurons, sparse positive activations, recurrent computation, and synaptic-plasticity-based working memory. BDH-CQ extends the direction toward in-context learning with recurrent latent reasoning, where inference-time inputs update recurrent memory and a query is solved through iterative latent computation without requiring verbalized intermediate reasoning.

These ideas motivate a practical question:

> Can a smaller, transparent implementation provide an experimental environment in which recurrent memory and latent reasoning can be manipulated, observed, and stress-tested?

This project addresses that question through an interactive research instrument.

---

## 2. Research Questions

### Primary question

Can a sparse recurrent architecture with associative memory and latent iterative computation learn and apply structured transformations from demonstrations while remaining computationally inspectable and avoiding the need for verbalized chain-of-thought?

### Secondary question

Where does such an architecture fail as task complexity increases, and can those failure modes be detected or mitigated?

---

## 3. Relationship to Public Research

### 3.1 Dragon Hatchling (BDH)

The BDH paper describes a scale-free, biologically inspired network of locally interacting neuron particles. It discusses sparse positive activations, recurrent computation, synaptic plasticity, and interpretability.

Source: Kosowski et al., *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*, arXiv:2509.26507.

https://arxiv.org/abs/2509.26507

### 3.2 BDH-CQ

BDH-CQ investigates in-context learning with recurrent latent reasoning. Inputs presented during inference update recurrent memory, after which a query is solved through iterative computation in a high-dimensional latent space without verbalizing intermediate reasoning.

The published paper reports evaluation on ARC-AGI-1 and controlled ARC-like interventions. Its reported results are **not** the results of this project.

Source: Engdahl et al., *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*, arXiv:2608.09888.

https://arxiv.org/abs/2608.09888

### 3.3 Related research

The project is also conceptually related to:

- Fu et al., *Hungry Hungry Hippos: Towards Language Modeling with State Space Models*, arXiv:2212.14052.
- Behrouz et al., *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663.

These works provide broader context for recurrent/state-space computation and test-time memory.

---

## 4. Project Architecture

The system follows this experimental pipeline:

```text
React Playground
       |
       v
FastAPI Backend
       |
       v
Task Generator
       |
       v
Model Engine
  |    |    |
  |    |    +--> Sparse Neuron Layer
  |    +-------> Associative Memory
  +------------> Latent Reasoning
       |
       v
Transformation / Hypothesis Engine
       |
       v
Evaluation + Telemetry
       |
       +--> Interpretability Diagnostics
       |
       +--> Failure Analysis
```

The frontend controls the experiment. The backend is responsible for task generation, model execution, evaluation, and telemetry.

---

## 5. Demonstration-Conditioned Memory

Demonstrations are processed sequentially and used to update an associative memory state.

A simplified update is:

\[
S_t=(1-\gamma)S_{t-1}+\eta(A_tB_t^T)
\]

where \(S_t\) represents the memory state, \(\gamma\) controls decay, and \(\eta\) controls plasticity.

The purpose is to study how information from previous demonstrations can influence later computation through a persistent state.

This is an experimental implementation inspired by the broader memory concepts discussed in the cited research; it should not be interpreted as an exact reproduction of Pathway's internal memory implementation.

---

## 6. Recurrent Latent Reasoning

After demonstrations have been processed, the query is encoded and passed through a recurrent latent workspace.

A simplified abstraction is:

\[
H_{r+1}=F_\theta(H_r,S_K)
\]

where \(H_r\) is the latent workspace at reasoning step \(r\), and \(S_K\) is the memory state after demonstration processing.

The interactive playground allows reasoning depth to be changed, making it possible to experimentally compare shallow and deeper recurrent computation.

---

## 7. Sparse Computation and Telemetry

The experimental model includes a sparse neuron layer using positive nonlinear activations.

The system records diagnostics such as:

- activation sparsity,
- memory norm,
- effective rank,
- entropy,
- interference-related metrics,
- reconstruction error,
- reasoning steps,
- halting step.

These measurements are intended as **diagnostic telemetry**.

They are not a human-readable chain-of-thought and should not be interpreted as a literal transcript of internal reasoning.

---

## 8. Transformation Hypothesis Engine

The project includes a structured transformation hypothesis engine.

Instead of directly accessing the query target during prediction, candidate transformations are tested against the demonstrations.

The process is:

```text
Demonstrations
      |
      v
Candidate transformations
      |
      v
Apply candidates to demonstrations
      |
      v
Consistency verification
      |
      +---- inconsistent --> reject
      |
      v
Rank candidates
      |
      v
Apply selected transformation to query
      |
      v
Prediction
```

The current experimental implementation includes:

- translation,
- color transformation,
- boundary propagation.

This hypothesis engine is a project-level experimental mechanism, not an official component of BDH-CQ.

---

## 9. Interactive Task Families

The current controlled playground contains three task families.

### Boundary Propagation

A generated colored boundary cell propagates according to the task's boundary rule.

### Color Transformation

Demonstrations define a mapping between grid colors.

### Translation

Objects or patterns are shifted according to the transformation demonstrated by the examples.

The infrastructure is designed to allow additional task families to be added later.

---

## 10. Interactive Learning Artifact

The browser playground allows the learner to change:

- task family,
- memory mode,
- reasoning depth,
- number of demonstrations,
- grid size,
- random seed,
- adaptive halting,
- memory rereading behaviour.

The learner can therefore modify a computational assumption and immediately observe:

- a new prediction,
- evaluation metrics,
- transformation selection,
- and internal telemetry.

This is the central interactive component of the submission.

---

## 11. Controlled Evaluation

The project includes a seeded controlled evaluation.

| Parameter | Configuration |
|---|---:|
| Task families | 3 |
| Memory modes | 2 |
| Reasoning depths | 3 |
| Seeds | 20 |
| Grid size | 5 × 5 |
| Demonstrations | 3 |
| Total episodes | 360 |

The configurations are:

- Boundary Propagation, Color Transformation, Translation
- Baseline and Simple Hebbian memory
- Reasoning depths 1, 4, and 8
- Seeds 1 through 20

The machine-readable results are stored in:

`results/controlled_evaluation.csv`

---

## 12. Verified Result

The corrected evaluation produced:

| Metric | Result |
|---|---:|
| Episodes | 360 |
| Exact matches | 360 |
| Exact-match rate | 100% |
| Mean accuracy | 100% |

All 360 generated episodes produced exact grid matches.

### Scientific interpretation

This result establishes that the implementation reliably solves the **defined controlled benchmark**.

It does not establish:

- 100% ARC-AGI performance,
- 100% BDH-CQ performance,
- 100% Pathway benchmark performance,
- general human-level reasoning,
- or broad generalization.

The benchmark is currently saturated: all tested configurations reached the same result. Therefore, these experiments do **not** show that deeper reasoning or a particular memory mode is superior.

This saturation motivates the next stage of experimentation.

---

## 13. Debugging and Benchmark Validation

During development, the Boundary Propagation benchmark initially produced reduced accuracy.

Investigation showed that the generator and transformation implementation disagreed about corner semantics. Certain corner cells could be interpreted simultaneously as vertical and horizontal boundary seeds.

The benchmark was corrected so that the generator and transformation implementation share the same boundary semantics.

The corrected 20-seed Boundary Propagation evaluation achieved:

- 20/20 exact matches
- 100% exact accuracy

This debugging episode is important scientifically because it demonstrates that an apparent model failure can originate in the benchmark specification itself. Generator semantics and evaluator semantics therefore need to be validated before attributing failures to the architecture.

---

## 14. Original Project Contributions

The project combines public research inspiration with project-level experimental mechanisms.

### Interactive research playground

Users can directly manipulate model and task parameters.

### Latent telemetry

The system provides quantitative diagnostic measurements rather than exposing a natural-language chain-of-thought.

### Transformation consistency verification

Candidate transformations are checked against demonstrations before being applied to the query.

### Empirical failure-boundary discovery

The intended failure-analysis workflow sweeps task complexity rather than assuming universal failure thresholds.

### Interference-oriented diagnostics

Memory behaviour can be examined using state statistics and interference-related measurements.

### Reproducible evaluation

Experiments use explicit seeds and produce machine-readable result files.

---

## 15. Limitations

### Limited benchmark difficulty

The current benchmark is intentionally controlled and relatively small.

### Benchmark saturation

The current 360 episodes do not distinguish the tested memory and reasoning-depth configurations.

### Generalization

Generated ARC-style tasks do not establish performance on arbitrary external ARC-AGI tasks.

### Scale

The implementation is research-scale and much smaller than production language models.

### Proprietary implementation unavailable

The project does not have access to Pathway's proprietary internal implementation or infrastructure.

### Interpretability scope

Telemetry provides quantitative diagnostics, not complete semantic explanations of internal computation.

### Failure prediction

Failure-risk mechanisms require validation on substantially more difficult and independently generated tasks before strong claims can be made.

---

## 16. Future Work

The next experimental stages are:

1. Expand task families.
2. Introduce harder multi-step transformations.
3. Add held-out and out-of-distribution tasks.
4. Sweep reasoning depth systematically.
5. Compare additional memory mechanisms.
6. Compare lightweight neural baselines.
7. Increase grid size and demonstration complexity.
8. Study memory interference as demonstrations increase.
9. Evaluate on external/public ARC datasets.
10. Improve failure-risk calibration.
11. Investigate compute and memory scaling.
12. Build a failure atlas directly from experimental measurements.

The objective is to determine where recurrent latent computation and associative memory provide measurable advantages, where they fail, and whether their internal diagnostics can help explain those failures.

---

## 17. Reproducibility

The controlled evaluation uses explicit seeds:

```text
Seeds: 1–20
Grid: 5 × 5
Demonstrations: 3
Reasoning depths: 1, 4, 8
Memory: baseline, simple_hebbian
```

Run the benchmark from the repository root:

```powershell
.venv\Scripts\python.exe -m backend.experiments.controlled_evaluation
```

Run the test suite:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

Run the backend:

```bash
python -m uvicorn backend.api.main:app --reload
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

---

## 18. Scientific Integrity

The project follows these reporting principles:

- Proprietary Pathway results are not represented as reproduced results.
- Generated benchmark results are explicitly labeled as generated benchmark results.
- The 100% result is not presented as ARC-AGI performance.
- Diagnostic telemetry is not described as literal chain-of-thought.
- Failure thresholds are not presented as universal architectural limits.
- Future work is separated from verified current results.
- The project distinguishes public research concepts from project-level experimental mechanisms.

---

## 19. Conclusion

This project does not claim to reproduce BDH-CQ.

Instead, it builds an interactive experimental laboratory around publicly described principles of:

```text
Sparse Computation
        +
Associative Memory
        +
Latent Recurrence
        +
Structured Transformations
        +
Interpretability Telemetry
        +
Failure Analysis
```

The current implementation solves all 360 episodes in its controlled benchmark. The more important research opportunity is now to make the benchmark harder, introduce held-out generalization tests, and determine where the architecture succeeds, where it fails, and whether its internal telemetry can help identify those failure modes.

The artifact therefore serves both as an educational demonstration and as a foundation for further controlled research.

---

## References

1. Kosowski, A., Uznański, P., Chorowski, J., Stamirowska, Z., & Bartoszkiewicz, M. (2025). **The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.** arXiv:2509.26507.  
   https://arxiv.org/abs/2509.26507

2. Engdahl, B., Kosowski, A., Chorowski, J., Stamirowska, Z., Uznański, P., Jiang, J., Phadke, R., Kinas, R., & Zhong, R. (2026). **BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.** arXiv:2608.09888.  
   https://arxiv.org/abs/2608.09888

3. Behrouz, A., Zhong, P., & Mirrokni, V. (2024/2025). **Titans: Learning to Memorize at Test Time.** arXiv:2501.00663.  
   https://arxiv.org/abs/2501.00663

4. Fu, D. Y., Dao, T., Saab, K. K., Thomas, A. W., Rudra, A., & Ré, C. (2022). **Hungry Hungry Hippos: Towards Language Modeling with State Space Models.** arXiv:2212.14052.  
   https://arxiv.org/abs/2212.14052

5. **Pathway — Kharagpur Data Science Hackathon 2026.**  
   https://pathway.com/events/kharagpur-data-science-hackathon-2026

6. **Pathway BDH public repository.**  
   https://github.com/pathwaycom/bdh

---

## Disclosure

AI tools were used during development for code-generation assistance, debugging, documentation drafting, architecture discussion, experiment-design discussion, and technical explanation.

AI-assisted material was reviewed and integrated by the project team. Experimental results reported here correspond to executions of the project code and stored benchmark outputs.
