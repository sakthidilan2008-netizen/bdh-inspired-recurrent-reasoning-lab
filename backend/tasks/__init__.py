from backend.tasks.generators import ColorTransformationGenerator, TranslationGenerator
from backend.schemas import Task, Split, DifficultyConfig

# Registry of available families
GENERATORS = {
    "ColorTransformation": ColorTransformationGenerator(),
    "Translation": TranslationGenerator()
}

def generate_task(family: str, split: Split, difficulty: DifficultyConfig, seed: int) -> Task:
    if family not in GENERATORS:
        raise ValueError(f"Task family {family} not found in registry")
    return GENERATORS[family].sample(split, difficulty, seed)
