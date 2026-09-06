from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class Split(str, Enum):
    train = "train"
    validation = "validation"
    held_out_test = "held_out_test"
    ood_test = "ood_test"

class DifficultyConfig(BaseModel):
    grid_size: Optional[List[int]] = None
    n_colors: Optional[int] = None
    n_demonstrations: Optional[int] = None
    nesting_depth: Optional[int] = None
    sequence_length: Optional[int] = None
    noise_level: Optional[float] = None

class Demonstration(BaseModel):
    input_grid: List[List[int]]
    output_grid: List[List[int]]

class Query(BaseModel):
    input_grid: List[List[int]]

class TaskMetadata(BaseModel):
    generator_version: str
    concept_labels: List[str]

class Task(BaseModel):
    task_id: str
    family: str
    difficulty: DifficultyConfig
    split: Split
    seed: int
    demonstrations: List[Demonstration]
    query: Query
    target: Optional[List[List[int]]] = None
    metadata: TaskMetadata

class MemoryMode(str, Enum):
    baseline = "baseline"
    simple_hebbian = "simple_hebbian"
    fast_weight_associative = "fast_weight_associative"
    normalized_associative = "normalized_associative"
    interference_mitigated = "interference_mitigated"

class ReasoningConfig(BaseModel):
    depth: int = 1
    adaptive_halt: bool = False
    reread_memory: bool = True
