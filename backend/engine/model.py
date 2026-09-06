import torch
import torch.nn as nn
from typing import List, Tuple

from backend.engine.base import (
    MemoryModule,
    MemoryState,
    ReasoningEngine,
    ReasoningConfig,
    ActivationRecord,
)
from backend.engine.neurons import NeuronLayer


class ModelEngine(nn.Module):
    """
    Integrated experimental model pipeline.

    Pipeline:
        input features
            -> sparse neuron layer
            -> projection to model dimension
            -> memory
            -> latent recurrent reasoning

    This is a teaching/research-scale implementation inspired by
    publicly described principles. It is NOT a reproduction of any
    proprietary production system.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_neurons: int,
        memory_module: MemoryModule,
        reasoning_engine: ReasoningEngine,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.n_neurons = n_neurons

        self.memory = memory_module
        self.reasoner = reasoning_engine

        # ---------------------------------------------------------
        # 1. Input encoder
        # ---------------------------------------------------------
        self.encoder_linear = nn.Linear(d_in, d_model)

        # ---------------------------------------------------------
        # 2. Sparse neuron population
        # ---------------------------------------------------------
        self.neuron_layer = NeuronLayer(
            d_in=d_model,
            n_neurons=n_neurons,
            activation_type="relu",
            top_k=None,
            topology="block_sparse",
            n_blocks=4,
            ei_ratio=0.8,
        )

        # ---------------------------------------------------------
        # 3. Project sparse population back to model dimension
        # ---------------------------------------------------------
        self.neuron_projection = nn.Linear(
            n_neurons,
            d_model,
        )

        # Stores activation records generated during the episode.
        self.activation_records: List[ActivationRecord] = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dense input encoding.

        x:
            [B, d_in]

        returns:
            [B, d_model]
        """

        if x.size(-1) != self.d_in:
            raise ValueError(
                f"Expected last dimension {self.d_in}, "
                f"got {x.size(-1)}"
            )

        return self.encoder_linear(x)

    def neural_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input through the sparse neuron population.

        Dense embedding:
            e_t

        Sparse population:
            z_t

        Model representation:
            h_t
        """

        e_t = self.encode(x)

        sparse_activity, record = self.neuron_layer(e_t)

        self.activation_records.append(record)

        projected = self.neuron_projection(sparse_activity)

        return projected

    def forward(
        self,
        query: torch.Tensor,
        demonstrations: List[Tuple[torch.Tensor, torch.Tensor]],
        reasoning_cfg: ReasoningConfig,
    ):
        """
        Main integrated forward pass.

        1. Reset memory.
        2. Encode demonstrations through sparse neurons.
        3. Write transformed representations into memory.
        4. Encode query through sparse neurons.
        5. Run latent reasoning.
        6. Return reasoning trace, memory state and activation records.
        """

        B = query.size(0)

        # Start a fresh activation trace for every episode.
        self.activation_records = []

        # ---------------------------------------------------------
        # 1. Reset memory
        # ---------------------------------------------------------
        mem_state = self.memory.reset(B)

        # ---------------------------------------------------------
        # 2. Ingest demonstrations
        # ---------------------------------------------------------
        for dem_in, dem_out in demonstrations:

            encoded_in = self.neural_encode(dem_in)
            encoded_out = self.neural_encode(dem_out)

            mem_state = self.memory.write(
                mem_state,
                encoded_in,
                encoded_out,
            )

        # ---------------------------------------------------------
        # 3. Encode query
        # ---------------------------------------------------------
        encoded_query = self.neural_encode(query)

        # ---------------------------------------------------------
        # 4. Latent reasoning
        # ---------------------------------------------------------
        trace = self.reasoner(
            encoded_query,
            mem_state,
            reasoning_cfg,
        )

        return (
            trace,
            mem_state,
            self.activation_records,
        )