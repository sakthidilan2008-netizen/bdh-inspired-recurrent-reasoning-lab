import torch
import torch.nn as nn
from typing import Optional
from backend.engine.base import MemoryModule, MemoryState, MemoryDiagnostics
from backend.schemas import MemoryMode

class BaselineMemory(nn.Module):
    """
    Baseline memory module that stores nothing and reads zeros.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.mode = MemoryMode.baseline
        self.d_model = d_model
        
    def reset(self, batch_size: int) -> MemoryState:
        # S is [B, d_model, d_model] of zeros
        S = torch.zeros(batch_size, self.d_model, self.d_model)
        return MemoryState(S=S)
        
    def write(self, state: MemoryState, x: torch.Tensor, y: torch.Tensor) -> MemoryState:
        return state
        
    def read(self, state: MemoryState, query: torch.Tensor) -> torch.Tensor:
        # query: [B, d_model]
        # read: [B, d_model]
        return torch.zeros_like(query)
        
    def diagnostics(self, state: MemoryState) -> MemoryDiagnostics:
        return MemoryDiagnostics(
            state_norm=0.0,
            effective_rank=0.0,
            entropy=0.0,
            overlap=0.0,
            interference_score=0.0,
            reconstruction_error=0.0,
            activation_collision=0.0,
            sparsity_degradation=0.0
        )

class SimpleHebbianMemory(nn.Module):
    """
    Simple Hebbian associative memory.
    S_t = λ S_{t-1} + η (x_t ⊗ y_t)
    """
    def __init__(self, d_model: int, decay: float = 1.0, plasticity: float = 1.0):
        super().__init__()
        self.mode = MemoryMode.simple_hebbian
        self.d_model = d_model
        
        # Learnable parameters mapped through constraints (λ in [0,1], η >= 0)
        # For simplicity in hackathon, we initialize them as fixed or learnable logits
        self.lambda_logit = nn.Parameter(torch.tensor(decay).logit()) if decay < 1.0 else nn.Parameter(torch.tensor(10.0))
        self.eta_raw = nn.Parameter(torch.tensor(plasticity))
        
    @property
    def lam(self):
        return torch.sigmoid(self.lambda_logit)
        
    @property
    def eta(self):
        return torch.nn.functional.softplus(self.eta_raw)
        
    def reset(self, batch_size: int) -> MemoryState:
        # S is [B, d_model, d_model]
        S = torch.zeros(batch_size, self.d_model, self.d_model)
        return MemoryState(S=S)
        
    def write(self, state: MemoryState, x: torch.Tensor, y: torch.Tensor) -> MemoryState:
        """
        x: [B, d_model] key
        y: [B, d_model] value
        S: [B, d_model, d_model]
        """
        # x_t ⊗ y_t -> shape [B, d_model, d_model]
        # y is the value, x is the key. Typically update is V * K^T
        # so S = S + eta * (y.unsqueeze(-1) @ x.unsqueeze(1))
        # Wait, MEMORY_SPEC says x_t ⊗ y_t. Let's use y @ x.T
        update = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
        S_new = self.lam * state.S + self.eta * update
        return MemoryState(S=S_new)
        
    def read(self, state: MemoryState, query: torch.Tensor) -> torch.Tensor:
        """
        query: [B, d_model]
        read: [B, d_model]
        """
        # S @ q -> shape [B, d_model]
        # query is a key. S * q -> (y @ x^T) @ q = y @ (x^T @ q)
        return torch.bmm(state.S, query.unsqueeze(2)).squeeze(2)
        
    def diagnostics(self, state: MemoryState) -> MemoryDiagnostics:
        # Calculate diagnostics
        norm = torch.linalg.matrix_norm(state.S, ord='fro').mean().item()
        
        # Effective rank via SVD
        try:
            U, S_vals, V = torch.linalg.svd(state.S.detach())
            # S_vals is [B, d_model]
            max_s = S_vals.max(dim=1, keepdim=True).values
            # threshold e.g. 1% of max
            threshold = 0.01 * max_s
            eff_rank = (S_vals > threshold).float().sum(dim=1).mean().item()
            
            # Entropy of singular values
            s_norm = S_vals / (S_vals.sum(dim=1, keepdim=True) + 1e-9)
            entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-9), dim=1).mean().item()
        except Exception:
            eff_rank = 0.0
            entropy = 0.0
            
        return MemoryDiagnostics(
            state_norm=norm,
            effective_rank=eff_rank,
            entropy=entropy,
            overlap=0.0, # Needs history of keys
            interference_score=0.0, # Needs evaluation logic
            reconstruction_error=0.0,
            activation_collision=0.0,
            sparsity_degradation=0.0
        )
