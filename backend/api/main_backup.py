from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import uuid

from backend.schemas import DifficultyConfig, Split, ReasoningConfig, MemoryMode
from backend.tasks import generate_task

app = FastAPI(title="BDH Research Lab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/tasks/generate")
def api_generate_task(family: str, difficulty: DifficultyConfig, split: Split, seed: int):
    try:
        task = generate_task(family, split, difficulty, seed)
        return task
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/playground/episode")
def api_playground_episode(family: str, difficulty: DifficultyConfig, seed: int):
    # Synchronously run one episode and return trace
    task = generate_task(family, Split.validation, difficulty, seed)
    
    # Mock response for now
    return {
        "task": task,
        "prediction": task.target, # Oracle prediction for UI stub
        "telemetry": {
            "memory": {"norm": 1.2, "interference_score": 0.05},
            "reasoning": {"trajectory_length": 4, "convergence_deltas": [0.8, 0.4, 0.1, 0.01]},
            "neurons": {"mean_sparsity": 0.95}
        }
    }

@app.get("/api/v1/failure-atlas/{task_family}")
def api_failure_atlas(task_family: str):
    # Mock data for failure atlas
    return {
        "family": task_family,
        "thresholds": {
            "sequence_length": 6,
            "nesting_depth": 3
        }
    }
