# ARCHITECTURE_DECISIONS.md

This document tracks architectural conflicts between the provided specifications and the overarching project requirements, detailing how they are resolved in the implementation. `ARCHITECTURE.md` is treated as the primary source of truth, superseded only by explicit mandates in the hackathon project brief.

## Decision 1: Decoder Architecture vs. Transformation Hypothesis Engine

**1. The Conflict**
`MODEL_SPEC.md` (§8 Decoder) specifies a simple direct decoder for grid tasks: "a small deconvolution/MLP head producing a H×W×C categorical grid, cross-entropy per cell." However, the main project brief explicitly mandates an original addition: the **Transformation Hypothesis Engine** and **Demonstration Consistency Verifier**. The brief states: "Do not simply decode H_R directly into the answer... implement a structured hypothesis layer."

**2. The Technically Correct Interpretation**
The Transformation Hypothesis Engine supersedes the simple MLP decoder for structured ARC-style tasks. 

**3. Why that interpretation was chosen**
The project brief explicitly emphasizes this pipeline as "OUR proposed accuracy/verification layer" and a critical original addition for achieving the 89-95% accuracy target on controlled benchmarks. The simple decoder would fail to achieve this and misses the project's verification requirements.

**4. How the implementation resolves it**
The `backend/engine/decoder/` module will be implemented as a modular hypothesis generation and verification engine. It will take the final latent state `H_R`, generate candidate transformations (translation, reflection, recoloring, etc.), apply them to the demonstration inputs, compare against demonstration outputs to reject inconsistent candidates, and only then apply the top-ranked valid transformation to the query.

---

## Decision 2: Input Representation (Pixels vs. Objects)

**1. The Conflict**
The specifications (`MODEL_SPEC.md`) generally imply a standard dense neural input embedding `e_t ∈ R^d_in`. However, the project brief (Section 17) requires an **Object-Centric ARC Representation**, mandating the extraction of connected components, bounding boxes, shapes, and relative positions, maintaining both pixel and object representations.

**2. The Technically Correct Interpretation**
The input processing pipeline must parse raw grids into object-centric graphs/features before or during the encoding step into `e_t`.

**3. Why that interpretation was chosen**
ARC tasks are notoriously difficult for pure pixel-level neural networks without immense scale. Explicitly extracting object features (as an admitted non-BDH mechanism) allows the sparse recurrent memory and reasoning engine to operate on a tractable, structured representation, enabling the educational and interpretability goals of the project at a hackathon scale.

**4. How the implementation resolves it**
The Task Generator (`backend/tasks/`) will parse ARC grids into both raw pixel arrays and structured object lists. The Encoder (`H_0 = Encoder(query, Read(S))`) will embed these object features (color, bounding box, centroid) alongside pixel data to form the initial state, allowing the sparse neuron layer to route and reason over structural properties.
