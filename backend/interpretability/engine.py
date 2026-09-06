import torch
from typing import Dict, Any, List
from backend.engine.base import ReasoningTrace, MemoryState, ActivationRecord

class InterpretabilityEngine:
    """
    Collects telemetry from forward passes.
    """
    def __init__(self):
        pass
        
    def collect(
        self, 
        trace: ReasoningTrace, 
        memory: MemoryState, 
        activation_records: List[ActivationRecord]
    ) -> Dict[str, Any]:
        """
        Creates a TelemetryBundle.
        """
        # Memory metrics
        norm = torch.linalg.matrix_norm(memory.S, ord='fro').mean().item()
        
        # Reasoning metrics
        convergence = trace.convergence_deltas
        halting = trace.halting_step
        
        # Neuron metrics
        mean_sparsity = sum(r.sparsity for r in activation_records) / max(1, len(activation_records))
        
        # We also want to record freq_ema for visualization
        # Getting the last record's EMA is sufficient for global stats
        if activation_records:
            freq_ema = activation_records[-1].neuron_freq_ema.tolist()
        else:
            freq_ema = []
            
        return {
            "memory": {
                "norm": norm,
                "interference_score": 0.0, # Placeholder until failure analysis populates
            },
            "reasoning": {
                "trajectory_length": len(trace.H),
                "convergence_deltas": convergence,
                "adaptive_halting_step": halting
            },
            "neurons": {
                "mean_sparsity": mean_sparsity,
                "activation_frequencies": freq_ema
            }
        }
