import torch
import pytest
from backend.tasks.generators import ColorTransformationGenerator, TranslationGenerator
from backend.schemas import DifficultyConfig, Split
from backend.engine.decoder.hypothesis import TransformationHypothesisEngine
from backend.failure.engine import FailureRiskEstimator
from backend.engine.memory.mitigated import InterferenceMitigatedMemory

def test_task_generators():
    difficulty = DifficultyConfig(grid_size=[5, 5], n_demonstrations=3)
    gen = ColorTransformationGenerator()
    task = gen.sample(Split.train, difficulty, seed=42)
    
    assert task.family == "ColorTransformation"
    assert len(task.demonstrations) == 3
    assert len(task.query.input_grid) == 5
    assert len(task.target) == 5
    
    gen_trans = TranslationGenerator()
    task_trans = gen_trans.sample(Split.train, difficulty, seed=42)
    assert task_trans.family == "Translation"

def test_hypothesis_engine():
    engine = TransformationHypothesisEngine(d_model=64)
    H_R = torch.randn(1, 64)
    
    difficulty = DifficultyConfig(grid_size=[5, 5], n_demonstrations=2)
    gen = ColorTransformationGenerator()
    task = gen.sample(Split.train, difficulty, seed=42)
    
    pred, trace = engine(H_R, task.query, task.demonstrations)
    
    assert pred.shape == (5, 5)
    assert len(trace) > 0
    assert "score" in trace[0]
    
def test_failure_estimator():
    estimator = FailureRiskEstimator()
    report = estimator.estimate(0.9, 0.6, 0.1, 0.05, 0.2)
    assert report["label"] in ["WARNING", "HIGH_RISK", "SAFE"]
    assert "score" in report
    
def test_mitigated_memory():
    mem = InterferenceMitigatedMemory(d_model=64, decay=0.9, plasticity=1.0)
    state = mem.reset(2)
    x = torch.randn(2, 64)
    y = torch.randn(2, 64)
    new_state = mem.write(state, x, y)
    
    assert new_state.S.shape == (2, 64, 64)
    # Check norms are clamped to 1.0 (since they were likely > 1)
    norms = torch.linalg.vector_norm(new_state.S, ord=2, dim=2)
    assert (norms <= 1.001).all()
