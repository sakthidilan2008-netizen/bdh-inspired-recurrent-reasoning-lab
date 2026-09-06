import torch
import numpy as np
from typing import Dict, Any, List

def exact_match(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Returns 1.0 if grid matches exactly, 0.0 otherwise."""
    return float(torch.equal(pred, target))

def cell_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Returns fraction of cells matched."""
    if pred.shape != target.shape:
        return 0.0
    return (pred == target).float().mean().item()

def evaluate_predictions(predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict[str, float]:
    """
    Evaluates a batch of predictions against targets.
    """
    if not predictions or not targets:
        return {}
        
    exact_matches = [exact_match(p, t) for p, t in zip(predictions, targets)]
    cell_accs = [cell_accuracy(p, t) for p, t in zip(predictions, targets)]
    
    return {
        "exact_match": sum(exact_matches) / len(exact_matches),
        "cell_accuracy": sum(cell_accs) / len(cell_accs)
    }
