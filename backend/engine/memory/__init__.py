from backend.engine.memory.implementations import BaselineMemory, SimpleHebbianMemory
from backend.schemas import MemoryMode

def get_memory_module(mode: MemoryMode, d_model: int):
    if mode == MemoryMode.baseline:
        return BaselineMemory(d_model)
    elif mode == MemoryMode.simple_hebbian:
        return SimpleHebbianMemory(d_model)
    else:
        raise NotImplementedError(f"Memory mode {mode} not yet implemented")
