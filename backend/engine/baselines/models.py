import torch
import torch.nn as nn

class MLPBaseline(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_classes: int):
        super().__init__()
        # Flattening 2D grid of size e.g. 7x7 to 49
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
        
class GRUBaseline(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_classes: int):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is sequence of demonstrations + query
        # For simple comparison, x could be [B, seq_len, d_in]
        out, _ = self.gru(x)
        # Predict based on last state
        return self.head(out[:, -1, :])
