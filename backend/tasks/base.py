import random
import numpy as np
from typing import List, Tuple
from backend.schemas import Task, Demonstration, Query, DifficultyConfig, TaskMetadata, Split
from abc import ABC, abstractmethod
import uuid

class TaskGenerator(ABC):
    """
    Base class for synthetic ARC-style task generators.
    """
    family: str = "Base"
    
    @abstractmethod
    def generate_episode(self, difficulty: DifficultyConfig, seed: int) -> Tuple[List[Demonstration], Query, List[List[int]]]:
        """
        Returns (demonstrations, query, target_grid)
        """
        pass
        
    def sample(self, split: Split, difficulty: DifficultyConfig, seed: int) -> Task:
        random.seed(seed)
        np.random.seed(seed)
        
        demos, query, target = self.generate_episode(difficulty, seed)
        
        # OOD or test splits might hide the target if serving purely inference client
        target_field = target if split in (Split.train, Split.validation) else None
        
        # Include for hackathon evaluation on backend
        # We will keep target internally for evaluation but the API might omit it
        
        task_id = f"task_{self.family}_{uuid.uuid4().hex[:8]}"
        
        return Task(
            task_id=task_id,
            family=self.family,
            difficulty=difficulty,
            split=split,
            seed=seed,
            demonstrations=demos,
            query=query,
            target=target, # We keep it here, eval strips it if needed
            metadata=TaskMetadata(
                generator_version="1.0",
                concept_labels=[self.family]
            )
        )
