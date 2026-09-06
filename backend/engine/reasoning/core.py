import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from backend.engine.base import MemoryState, MemoryModule, ReasoningTrace
from backend.schemas import ReasoningConfig

class ReasoningBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, memory_module: MemoryModule):
        super().__init__()
        self.d_model = d_model
        self.memory_module = memory_module
        
        # Small MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )
        
    def forward(self, H_r: torch.Tensor, memory: MemoryState, reread_memory: bool) -> torch.Tensor:
        if reread_memory:
            # Re-read from memory using current H_r as query
            mem_read = self.memory_module.read(memory, H_r)
        else:
            # Just read zeros or rely on original H_0
            mem_read = torch.zeros_like(H_r)
            
        # Concat H_r and mem_read
        x = torch.cat([H_r, mem_read], dim=-1)
        return self.mlp(x)

class LatentReasoningEngine(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, memory_module: MemoryModule):
        super().__init__()
        self.d_model = d_model
        self.block = ReasoningBlock(d_model, d_hidden, memory_module)
        self.norm = nn.LayerNorm(d_model)
        
        # Adaptive halting projector
        self.halt_proj = nn.Linear(d_model, 1)
        
    def forward(self, H_0: torch.Tensor, memory: MemoryState, cfg: ReasoningConfig) -> ReasoningTrace:
        H_r = H_0
        trajectory = [H_0]
        convergence_deltas = []
        
        halting_step = None
        accumulated_p = torch.zeros(H_0.size(0), 1, device=H_0.device)
        
        for r in range(cfg.depth):
            # Compute update
            update = self.block(H_r, memory, cfg.reread_memory)
            H_next = self.norm(H_r + update)
            
            # Convergence
            delta = torch.norm(H_next - H_r, dim=-1) / (torch.norm(H_r, dim=-1) + 1e-9)
            convergence_deltas.append(delta.mean().item()) # Simplified to mean across batch for tracing
            
            H_r = H_next
            trajectory.append(H_r)
            
            # Adaptive halting check
            if cfg.adaptive_halt and halting_step is None:
                p_r = torch.sigmoid(self.halt_proj(H_r))
                accumulated_p += p_r
                # If all items in batch cross threshold, we can halt early
                if (accumulated_p >= 0.99).all():
                    halting_step = r + 1
                    break
                    
        # Final trace
        return ReasoningTrace(
            H=trajectory,
            convergence_deltas=convergence_deltas,
            halting_step=halting_step,
            final_prediction=H_r # The decoder handles final translation
        )
