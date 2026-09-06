import random
import numpy as np

from backend.tasks.generators import BoundaryPropagationGenerator
from backend.schemas import DifficultyConfig
from backend.engine.decoder.hypothesis import BoundaryPropagationTransformation


def print_grid(title, grid):
    print(f"\n{title}")
    for row in grid:
        print(" ".join(str(x) for x in row))


generator = BoundaryPropagationGenerator()

difficulty = DifficultyConfig(
    grid_size=[7, 7],
    n_demonstrations=3
)

demos, query, target = generator.generate_episode(
    difficulty=difficulty,
    seed=42
)

transform = BoundaryPropagationTransformation()

for i, demo in enumerate(demos, start=1):
    x = np.array(demo.input_grid)
    expected = np.array(demo.output_grid)

    predicted = transform(
        __import__("torch").tensor(x)
    ).numpy()

    diff = predicted != expected

    print(f"\n{'=' * 50}")
    print(f"DEMO {i}")
    print(f"{'=' * 50}")

    print_grid("INPUT", x)
    print_grid("EXPECTED OUTPUT", expected)
    print_grid("PREDICTED OUTPUT", predicted)

    print(f"\nCorrect cells: {(~diff).sum()}/{diff.size}")
    print(f"Accuracy: {(~diff).mean():.4f}")

    if diff.any():
        print("\nMISMATCH LOCATIONS:")
        for r, c in zip(*np.where(diff)):
            print(
                f"row={r}, col={c}, "
                f"expected={expected[r,c]}, "
                f"predicted={predicted[r,c]}"
            )