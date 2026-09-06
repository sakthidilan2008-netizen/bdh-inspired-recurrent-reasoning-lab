import random
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.schemas import (
    DifficultyConfig,
    MemoryMode,
    ReasoningConfig,
    Task,
)

from backend.tasks.generators import (
    ColorTransformationGenerator,
    TranslationGenerator,
    BoundaryPropagationGenerator,
)

from backend.engine.memory import get_memory_module
from backend.engine.memory.mitigated import InterferenceMitigatedMemory
from backend.engine.reasoning.core import LatentReasoningEngine
from backend.engine.model import ModelEngine
from backend.engine.decoder.hypothesis import TransformationHypothesisEngine
from backend.interpretability.engine import InterpretabilityEngine


app = FastAPI(
    title="BDH-Inspired Research Playground",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class EpisodeRequest(BaseModel):
    family: str = "ColorTransformation"
    grid_size: Optional[list[int]] = None
    n_demonstrations: int = 3
    memory_mode: MemoryMode = MemoryMode.simple_hebbian
    reasoning_depth: int = 4
    adaptive_halt: bool = False
    reread_memory: bool = True
    seed: int = 42


# ---------------------------------------------------------
# Task generation
# ---------------------------------------------------------

def create_generator(family: str):
    if family == "ColorTransformation":
        return ColorTransformationGenerator()

    if family == "Translation":
        return TranslationGenerator()

    if family == "BoundaryPropagation":
        return BoundaryPropagationGenerator()

    raise ValueError(
        f"Unsupported family '{family}'. "
        "Available families: ColorTransformation, Translation, BoundaryPropagation"
    )


def generate_task(req: EpisodeRequest):
    random.seed(req.seed)
    torch.manual_seed(req.seed)

    generator = create_generator(req.family)

    difficulty = DifficultyConfig(
        grid_size=req.grid_size,
        n_demonstrations=req.n_demonstrations,
    )

    demos, query, target = generator.generate_episode(
        difficulty=difficulty,
        seed=req.seed,
    )

    return Task(
        task_id=f"playground-{req.seed}",
        family=req.family,
        difficulty=difficulty,
        split="validation",
        seed=req.seed,
        demonstrations=demos,
        query=query,
        target=target,
        metadata={
            "generator_version": "v1",
            "concept_labels": [req.family],
        },
    )


# ---------------------------------------------------------
# Grid → model feature adapter
# ---------------------------------------------------------

def grid_to_features(grid, d_in: int = 64):
    """
    Teaching-scale adapter.

    Converts an ARC grid into a fixed-size feature vector.
    This is NOT claimed to be an official BDH encoder.

    Features:
      - normalized flattened cells
      - zero padding / truncation
    """

    x = torch.tensor(grid, dtype=torch.float32)

    # Normalize ARC colors into approximately [0, 1].
    x = x / 9.0

    flat = x.flatten()

    if flat.numel() >= d_in:
        flat = flat[:d_in]
    else:
        padded = torch.zeros(d_in, dtype=torch.float32)
        padded[:flat.numel()] = flat
        flat = padded

    return flat.unsqueeze(0)


# ---------------------------------------------------------
# Memory construction
# ---------------------------------------------------------

def create_memory(mode: MemoryMode, d_model: int):
    if mode == MemoryMode.interference_mitigated:
        return InterferenceMitigatedMemory(
            d_model=d_model,
            decay=0.95,
            plasticity=0.10,
            top_k_svd=min(8, d_model),
        )

    return get_memory_module(mode, d_model)


# ---------------------------------------------------------
# Real episode execution
# ---------------------------------------------------------

def run_episode(req: EpisodeRequest):
    task = generate_task(req)

    d_in = 64
    d_model = 32
    n_neurons = 512

    memory = create_memory(req.memory_mode, d_model)

    reasoner = LatentReasoningEngine(
        d_model=d_model,
        d_hidden=64,
        memory_module=memory,
    )

    model = ModelEngine(
        d_in=d_in,
        d_model=d_model,
        n_neurons=n_neurons,
        memory_module=memory,
        reasoning_engine=reasoner,
    )

    hypothesis_engine = TransformationHypothesisEngine(
        d_model=d_model,
    )

    interpretability = InterpretabilityEngine()

    # -----------------------------------------------------
    # Build encoded demonstrations
    # -----------------------------------------------------

    demonstrations = []

    for demo in task.demonstrations:
        demo_in = grid_to_features(demo.input_grid, d_in)
        demo_out = grid_to_features(demo.output_grid, d_in)

        demonstrations.append(
            (demo_in, demo_out)
        )

    query_features = grid_to_features(
        task.query.input_grid,
        d_in,
    )

    # -----------------------------------------------------
    # Run actual memory + latent reasoning
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Run actual memory + latent reasoning
    # -----------------------------------------------------

    with torch.no_grad():
        trace, memory_state, activation_records = model(
            query=query_features,
            demonstrations=demonstrations,
            reasoning_cfg=ReasoningConfig(
                depth=max(1, req.reasoning_depth),
                adaptive_halt=req.adaptive_halt,
                reread_memory=req.reread_memory,
            ),
        )

        # The final latent state is the representation supplied
        # to the structured hypothesis verifier.
        H_R = trace.H[-1]

        prediction, hypothesis_trace = hypothesis_engine(
            H_R=H_R,
            query=task.query,
            demos=task.demonstrations,
        )
    # -----------------------------------------------------
    # Real telemetry
    # -----------------------------------------------------

    memory_diag = memory.diagnostics(memory_state)

    # ModelEngine now provides real NeuronLayer activation records.
    telemetry = interpretability.collect(
        trace=trace,
        memory=memory_state,
        activation_records=activation_records,
    )

    # Replace placeholder interference score with the
    # actual memory diagnostic value.
    telemetry["memory"]["interference_score"] = (
        memory_diag.interference_score
    )

    telemetry["memory"]["effective_rank"] = (
        memory_diag.effective_rank
    )

    telemetry["memory"]["entropy"] = (
        memory_diag.entropy
    )

    telemetry["memory"]["reconstruction_error"] = (
        memory_diag.reconstruction_error
    )

    # -----------------------------------------------------
    # Evaluation happens AFTER prediction
    # -----------------------------------------------------

    target = torch.tensor(
        task.target,
        dtype=torch.long,
    )

    prediction = prediction.to(torch.long)

    if prediction.shape == target.shape:
        correct_cells = int(
            (prediction == target).sum().item()
        )

        total_cells = int(target.numel())

        cell_accuracy = (
            correct_cells / total_cells
            if total_cells > 0
            else 0.0
        )

        exact_match = bool(
            torch.equal(prediction, target)
        )
    else:
        correct_cells = 0
        total_cells = int(target.numel())

        cell_accuracy = 0.0
        exact_match = False

    return {
        "task": {
            "task_id": task.task_id,
            "family": task.family,
            "seed": task.seed,
            "difficulty": task.difficulty.model_dump(),
            "demonstrations": [
                {
                    "input_grid": d.input_grid,
                    "output_grid": d.output_grid,
                }
                for d in task.demonstrations
            ],
            "query": {
                "input_grid": task.query.input_grid,
            },
        },

        "prediction": prediction.tolist(),

        "evaluation": {
            "exact_match": exact_match,
            "cell_accuracy": cell_accuracy,
            "correct_cells": correct_cells,
            "total_cells": total_cells,
        },

        "telemetry": telemetry,

        "hypothesis_trace": hypothesis_trace,

        "configuration": {
            "memory_mode": req.memory_mode.value,
            "reasoning_depth": req.reasoning_depth,
            "adaptive_halt": req.adaptive_halt,
            "reread_memory": req.reread_memory,
        },
    }


# ---------------------------------------------------------
# API endpoints
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "BDH-Inspired Research Playground",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post("/playground/episode")
def playground_episode(req: EpisodeRequest):
    return run_episode(req)


@app.get("/failure-atlas")
def failure_atlas():
    return {
        "status": "experimental",
        "message": (
            "Failure thresholds are not hard-coded here. "
            "Run empirical benchmark sweeps to populate "
            "the failure atlas."
        ),
    }