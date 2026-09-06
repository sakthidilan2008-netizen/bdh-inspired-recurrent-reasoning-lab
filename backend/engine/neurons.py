import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from backend.engine.base import ActivationRecord

class SparseActivation(nn.Module):
    def __init__(self, activation_type: str, top_k: Optional[int] = None):
        super().__init__()
        self.activation_type = activation_type
        self.top_k = top_k
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == 'relu':
            x = F.relu(x)
        elif self.activation_type == 'relu2':
            x = F.relu(x) ** 2
            # Pre-scale to avoid blowup, as specified in MODEL_SPEC
            n = x.size(-1)
            x = x / math.sqrt(n)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")
            
        if self.top_k is not None and self.top_k < x.size(-1):
            # top-k masking
            topk_vals, topk_idx = torch.topk(x, self.top_k, dim=-1)
            mask = torch.zeros_like(x).scatter_(-1, topk_idx, 1.0)
            x = x * mask
            
        # Hard sparsity control (clip ceiling)
        x = torch.clamp(x, max=10.0)
        return x

def create_block_sparse_mask(n: int, n_blocks: int) -> torch.Tensor:
    mask = torch.zeros(n, n)
    block_size = n // n_blocks
    for i in range(n_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, n)
        mask[start:end, start:end] = 1.0
    return mask

def create_small_world_mask(n: int, k: int, p: float) -> torch.Tensor:
    # A simple ring lattice with random rewiring approximation
    mask = torch.zeros(n, n)
    for i in range(n):
        for j in range(1, k // 2 + 1):
            mask[i, (i + j) % n] = 1.0
            mask[i, (i - j) % n] = 1.0
    # Rewire
    rewire_mask = torch.rand(n, n) < p
    random_edges = torch.randint(0, n, (n, n))
    mask = torch.where(rewire_mask, torch.eye(n).byte(), mask.byte()).float() # Just a rough approximation
    return torch.clamp(mask + torch.eye(n), max=1.0) # Ensure self-connections

class NeuronLayer(nn.Module):
    """
    Sparse positive neuron layer with local connectivity and E/I dynamics.
    """
    def __init__(
        self,
        d_in: int,
        n_neurons: int,
        activation_type: str = 'relu',
        top_k: Optional[int] = None,
        topology: str = 'dense',
        n_blocks: int = 4,
        ei_ratio: float = 0.8
    ):
        super().__init__()
        self.d_in = d_in
        self.n_neurons = n_neurons
        
        # W_in, b_in
        self.W_in = nn.Parameter(torch.Tensor(n_neurons, d_in))
        self.b_in = nn.Parameter(torch.Tensor(n_neurons))
        nn.init.kaiming_uniform_(self.W_in, a=math.sqrt(5))
        nn.init.zeros_(self.b_in)
        
        self.activation = SparseActivation(activation_type, top_k)
        
        # E/I grouping
        self.n_excitatory = int(n_neurons * ei_ratio)
        self.n_inhibitory = n_neurons - self.n_excitatory
        
        # Topology mask (used if there's recurrent lateral connectivity, but 
        # usually W_in maps from d_in -> n. If we want lateral connections, we need W_lat)
        # Assuming the topology applies to an optional lateral layer or out projection.
        # For this hackathon scope, we'll keep the mask ready for reasoning block.
        if topology == 'block_sparse':
            self.register_buffer('topology_mask', create_block_sparse_mask(n_neurons, n_blocks))
        elif topology == 'small_world':
            self.register_buffer('topology_mask', create_small_world_mask(n_neurons, k=4, p=0.1))
        else:
            self.register_buffer('topology_mask', torch.ones(n_neurons, n_neurons))
            
        # Tracking frequency EMA
        self.register_buffer('freq_ema', torch.zeros(n_neurons))
        
    def forward(self, e_t: torch.Tensor, memory_read: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ActivationRecord]:
        # e_t: [B, d_in]
        # x_t = σ_sparse(W_in e_t + b_in)
        pre_act = F.linear(e_t, self.W_in, self.b_in)
        
        # Add memory_read influence if provided, assuming projected to n_neurons
        # This is a simplification.
        
        post_act = self.activation(pre_act)
        
        # Sparsity ratio
        sparsity = (post_act == 0).float().mean().item()
        
        # EMA frequency
        batch_freq = (post_act > 0).float().mean(dim=0)
        self.freq_ema = 0.9 * self.freq_ema + 0.1 * batch_freq.detach()
        
        record = ActivationRecord(
            pre=pre_act.detach(),
            post=post_act.detach(),
            sparsity=sparsity,
            neuron_freq_ema=self.freq_ema.clone()
        )
        
        return post_act, record
