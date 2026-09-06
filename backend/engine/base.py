from dataclasses import dataclass
from typing import Protocol, List, Optional
import torch
import torch.nn as nn
from backend.schemas import MemoryMode, ReasoningConfig

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

@dataclass
class MemoryState:
    S: torch.Tensor
    # Add other states like gating history or keys if needed
    
class MemoryModule(Protocol):
    mode: MemoryMode
    
    def reset(self, batch_size: int) -> MemoryState:
        ...
        
    def write(self, state: MemoryState, x: torch.Tensor, y: torch.Tensor) -> MemoryState:
        ...
        
    def read(self, state: MemoryState, query: torch.Tensor) -> torch.Tensor:
        ...
        
    def diagnostics(self, state: MemoryState) -> MemoryDiagnostics:
        ...

@dataclass
class ReasoningTrace:
    H: List[torch.Tensor]
    convergence_deltas: List[float]
    halting_step: Optional[int]
    final_prediction: torch.Tensor

class ReasoningEngine(Protocol):
    def forward(self, query: torch.Tensor, memory: MemoryState, cfg: ReasoningConfig) -> ReasoningTrace:
        ...

@dataclass
class ActivationRecord:
    pre: torch.Tensor
    post: torch.Tensor
    sparsity: float
    neuron_freq_ema: torch.Tensor

class NeuronLayer(Protocol):
    def forward(self, e_t: torch.Tensor, memory_read: torch.Tensor) -> tuple[torch.Tensor, ActivationRecord]:
        ...
