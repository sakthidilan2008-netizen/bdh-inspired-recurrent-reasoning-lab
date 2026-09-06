import torch
import pytest
from backend.engine.memory.implementations import BaselineMemory, SimpleHebbianMemory
from backend.engine.base import MemoryState
from backend.schemas import MemoryMode, ReasoningConfig
from backend.engine.neurons import NeuronLayer
from backend.engine.reasoning.core import LatentReasoningEngine

def test_baseline_memory():
    mem = BaselineMemory(d_model=64)
    state = mem.reset(batch_size=2)
    assert state.S.shape == (2, 64, 64)
    assert (state.S == 0).all()
    
    x = torch.randn(2, 64)
    y = torch.randn(2, 64)
    new_state = mem.write(state, x, y)
    assert (new_state.S == 0).all()
    
    q = torch.randn(2, 64)
    read_out = mem.read(new_state, q)
    assert (read_out == 0).all()
    
def test_simple_hebbian_memory():
    mem = SimpleHebbianMemory(d_model=64, decay=1.0, plasticity=1.0)
    state = mem.reset(batch_size=2)
    
    x = torch.randn(2, 64)
    y = torch.randn(2, 64)
    
    new_state = mem.write(state, x, y)
    assert new_state.S.shape == (2, 64, 64)
    
    q = torch.randn(2, 64)
    read_out = mem.read(new_state, q)
    assert read_out.shape == (2, 64)
    
def test_neuron_layer():
    layer = NeuronLayer(d_in=128, n_neurons=256, activation_type='relu2', top_k=32, topology='block_sparse', n_blocks=4)
    e_t = torch.randn(2, 128)
    
    post_act, record = layer(e_t)
    assert post_act.shape == (2, 256)
    assert record.pre.shape == (2, 256)
    assert record.post.shape == (2, 256)
    
    # check sparsity
    assert (post_act > 0).sum(dim=1).max() <= 32
    
def test_reasoning_engine():
    mem = BaselineMemory(d_model=64)
    reasoner = LatentReasoningEngine(d_model=64, d_hidden=128, memory_module=mem)
    
    H_0 = torch.randn(2, 64)
    mem_state = mem.reset(2)
    cfg = ReasoningConfig(depth=3, adaptive_halt=False, reread_memory=True)
    
    trace = reasoner(H_0, mem_state, cfg)
    
    assert len(trace.H) == 4 # H_0 + 3 steps
    assert len(trace.convergence_deltas) == 3
    assert trace.final_prediction.shape == (2, 64)
