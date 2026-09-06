import torch
import torch.nn as nn
from backend.engine.base import MemoryState
from backend.engine.memory.implementations import SimpleHebbianMemory
from backend.schemas import MemoryMode

class InterferenceMitigatedMemory(SimpleHebbianMemory):
    def __init__(self, d_model: int, decay: float = 1.0, plasticity: float = 1.0, top_k_svd: int = 5):
        super().__init__(d_model, decay, plasticity)
        self.mode = MemoryMode.interference_mitigated
        self.top_k_svd = top_k_svd
        
        # Concept-aware gating
        self.gate_proj = nn.Linear(d_model, 1)
        
    def write(self, state: MemoryState, x: torch.Tensor, y: torch.Tensor) -> MemoryState:
        """
        Orthogonalization and selective gating before update.
        """
        B = x.size(0)
        
        # 1. Base update candidate
        w_t = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
        
        # 2. Orthogonalization via top-k principal directions projection
        # If memory is empty, skip orthogonalization
        if torch.norm(state.S).item() > 1e-5:
            try:
                U, S_vals, V = torch.linalg.svd(state.S.detach())
                # V is [B, d_model, d_model]. Top-k directions
                Vk = V[:, :self.top_k_svd, :] # [B, k, d_model]
                # Project w_t onto Vk and subtract
                # w_t is [B, d_model, d_model]. We can orthogonalize the key x.
                x_proj = torch.bmm(Vk.transpose(1, 2), torch.bmm(Vk, x.unsqueeze(2))).squeeze(2)
                x_ortho = x - x_proj
                w_t = torch.bmm(y.unsqueeze(2), x_ortho.unsqueeze(1))
            except Exception:
                pass # SVD failed, fallback to standard update
                
        # 3. Gating (selective plasticity)
        gate_score = torch.sigmoid(self.gate_proj(x)).unsqueeze(2) # [B, 1, 1]
        eta_eff = self.eta * (1 - gate_score)
        
        S_new = self.lam * state.S + eta_eff * w_t
        
        # Row normalization
        norms = torch.linalg.vector_norm(S_new, ord=2, dim=2, keepdim=True)
        S_new = S_new / torch.clamp(norms, min=1.0)
        
        return MemoryState(S=S_new)
