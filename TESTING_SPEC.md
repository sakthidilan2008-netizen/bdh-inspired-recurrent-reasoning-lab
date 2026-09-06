# TESTING_SPEC.md — Testing Strategy & Acceptance Criteria

## Test Categories

- **Unit tests**: each `MemoryMode`, `NeuronLayer`, `ReasoningEngine` config, each task generator — isolated, fast.
- **Integration tests**: full pipeline (task → memory → reasoning → decode → eval) on a tiny synthetic config.
- **Numerical tests**: no NaN/Inf across a stress-test batch (large batch, max reasoning depth, max memory writes).
- **Determinism tests**: same seed + same config → bit-identical (or tolerance-bounded for GPU nondeterminism) result twice in a row.
- **Shape tests**: every module's output shape matches its declared contract (ARCHITECTURE.md §5) across a matrix of batch sizes/grid sizes.
- **Gradient tests**: `torch.autograd.gradcheck`-style verification on small configs for custom ops (e.g., memory write/read, orthogonalization projection).
- **Benchmark tests**: task generators produce valid, solvable-by-construction instances (a reference solver, where feasible, must solve 100% of instances it's given the ground-truth rule for).
- **OOD tests**: OOD split generation actually falls outside the declared training difficulty range (assert on the difficulty parameter, not just "different seed").
- **Regression tests**: pin known-good metric values (with tolerance) for a fixed tiny config + seed, to catch silent breakage.

## Explicit Acceptance Criteria (must all pass before a module is "done")

- [ ] Same random seed → reproducible result (determinism test)
- [ ] Memory dimensions remain stable across arbitrary-length demonstration sequences
- [ ] No NaN/Inf under stress-test batch
- [ ] Training loss decreases over N steps on a toy overfit-a-single-batch test
- [ ] Evaluation code path provably does not access labels (static check: label tensor not referenced in `Evaluator.score`'s prediction-generation call path; enforced by a dedicated test that monkeypatches labels to garbage post-prediction and confirms prediction is unaffected)
- [ ] Decoder output dimensions match target dimensions for every task family
- [ ] All reported metrics trace to `Evaluator.score` output, not hand-computed elsewhere

## CI Requirement
All categories run on every PR against `backend/`; sweep/experiment jobs are NOT part of CI (too slow) but their *code paths* are exercised via the integration test on a tiny config.

## Test Data
Use the smallest valid instance of each task family (e.g., 3×3 grid, depth-1 nesting) for unit/integration/shape tests to keep CI fast; reserve full-size instances for the numerical/stress tests only.
