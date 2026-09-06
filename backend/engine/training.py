import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import List, Dict, Any
from backend.engine.model import ModelEngine
from backend.engine.decoder.hypothesis import TransformationHypothesisEngine
from backend.schemas import Task, ReasoningConfig
from backend.eval.metrics import evaluate_predictions

class Trainer:
    def __init__(
        self, 
        model: ModelEngine, 
        hypothesis_engine: TransformationHypothesisEngine,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.hypothesis_engine = hypothesis_engine
        
        # We only train the parts of the model that have parameters
        params = list(model.parameters()) + list(hypothesis_engine.parameters())
        self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        
        # Loss for grid cells (assuming categorical grid output for hypothesis generator training)
        # Actually hypothesis engine generator outputs logits for candidate selection.
        # This is a simplified reinforcement/supervised setup.
        # For this hackathon, we might train the reasoner/encoder with a surrogate loss or direct REINFORCE.
        # Since it's an end-to-end differentiable system up to the discrete hypothesis selection,
        # we can use straight-through estimators or just train the representation.
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, task: Task, reasoning_cfg: ReasoningConfig) -> Dict[str, float]:
        self.model.train()
        self.hypothesis_engine.train()
        self.optimizer.zero_grad()
        
        # For batch size 1 (single task episode)
        
        # 1. Forward model
        query_grid = torch.tensor([task.query.input_grid], dtype=torch.float32) # [1, H, W]
        # In a real model we would embed it properly. For now we assume a flat embed:
        query_flat = query_grid.view(1, -1)
        
        # Pad or truncate flat to match d_in
        if query_flat.size(1) < self.model.d_in:
            pad = torch.zeros(1, self.model.d_in - query_flat.size(1))
            query_flat = torch.cat([query_flat, pad], dim=1)
        elif query_flat.size(1) > self.model.d_in:
            query_flat = query_flat[:, :self.model.d_in]
        
        trace, mem_state = self.model(query_flat, [], reasoning_cfg) # Pass empty demos for now to test flow
        
        # 2. Hypothesis Generation
        # Get logits for candidates
        logits = self.hypothesis_engine.generator(trace.final_prediction) # [1, num_candidates]
        
        # In a fully supervised setting we would need the index of the correct transformation.
        # For this mock trainer, we just compute a dummy loss to ensure gradients flow.
        target = torch.tensor([0], device=logits.device) # Dummy target
        loss = self.criterion(logits, target)
        
        loss.backward()
        
        # Gradient clipping as specified in REASONING_SPEC.md
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {"loss": loss.item()}
