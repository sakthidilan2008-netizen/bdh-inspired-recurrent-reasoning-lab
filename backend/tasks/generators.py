import numpy as np
import random
from typing import List, Tuple
from backend.tasks.base import TaskGenerator
from backend.schemas import Demonstration, Query, DifficultyConfig

class ColorTransformationGenerator(TaskGenerator):
    family = "ColorTransformation"
    
    def generate_episode(self, difficulty: DifficultyConfig, seed: int) -> Tuple[List[Demonstration], Query, List[List[int]]]:
        n_demos = difficulty.n_demonstrations or 3
        grid_h, grid_w = difficulty.grid_size or [5, 5]
        
        # pick a source color and a target color for this episode
        bg_color = 0
        colors = list(range(1, 10))
        random.shuffle(colors)
        src_color, tgt_color = colors[0], colors[1]
        
        demos = []
        for _ in range(n_demos):
            in_grid, out_grid = self._make_pair(grid_h, grid_w, src_color, tgt_color, bg_color)
            demos.append(Demonstration(input_grid=in_grid.tolist(), output_grid=out_grid.tolist()))
            
        q_in, q_out = self._make_pair(grid_h, grid_w, src_color, tgt_color, bg_color)
        query = Query(input_grid=q_in.tolist())
        target = q_out.tolist()
        
        return demos, query, target
        
    def _make_pair(self, h: int, w: int, src: int, tgt: int, bg: int) -> Tuple[np.ndarray, np.ndarray]:
        grid = np.full((h, w), bg, dtype=int)
        # Random noise of other colors
        num_noise = max(1, (h * w) // 5)
        for _ in range(num_noise):
            r, c = random.randint(0, h-1), random.randint(0, w-1)
            grid[r, c] = random.randint(1, 9)
            
        # Add source color
        num_src = max(1, (h * w) // 10)
        for _ in range(num_src):
            r, c = random.randint(0, h-1), random.randint(0, w-1)
            grid[r, c] = src
            
        out_grid = grid.copy()
        out_grid[grid == src] = tgt
        return grid, out_grid

class TranslationGenerator(TaskGenerator):
    family = "Translation"
    # Simplified: move a 2x2 object right by 1
    
    def generate_episode(self, difficulty: DifficultyConfig, seed: int) -> Tuple[List[Demonstration], Query, List[List[int]]]:
        n_demos = difficulty.n_demonstrations or 3
        grid_h, grid_w = difficulty.grid_size or [7, 7]
        
        obj_color = random.randint(1, 9)
        bg_color = 0
        direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        
        demos = []
        for _ in range(n_demos):
            in_grid, out_grid = self._make_pair(grid_h, grid_w, obj_color, bg_color, direction)
            demos.append(Demonstration(input_grid=in_grid.tolist(), output_grid=out_grid.tolist()))
            
        q_in, q_out = self._make_pair(grid_h, grid_w, obj_color, bg_color, direction)
        query = Query(input_grid=q_in.tolist())
        target = q_out.tolist()
        
        return demos, query, target
        
    def _make_pair(self, h: int, w: int, color: int, bg: int, dir: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        grid = np.full((h, w), bg, dtype=int)
        obj_h, obj_w = 2, 2
        
        # Valid start positions
        dr, dc = dir
        min_r = max(0, -dr)
        max_r = min(h - obj_h, h - obj_h - dr)
        min_c = max(0, -dc)
        max_c = min(w - obj_w, w - obj_w - dc)
        
        # if max is less than min due to small grid, just center it and no-op
        if max_r < min_r or max_c < min_c:
            r, c = h//2, w//2
            dr, dc = 0, 0
        else:
            r = random.randint(min_r, max_r)
            c = random.randint(min_c, max_c)
            
        grid[r:r+obj_h, c:c+obj_w] = color
        
        out_grid = np.full((h, w), bg, dtype=int)
        out_grid[r+dr:r+obj_h+dr, c+dc:c+obj_w+dc] = color
        
        return grid, out_grid
class BoundaryPropagationGenerator(TaskGenerator):
    """
    Boundary Propagation task.

    Rule:
    A colored cell placed on a grid boundary propagates inward along
    its row or column.

    - Top/bottom boundary cell -> fills its entire column.
    - Left/right boundary cell -> fills its entire row.

    This generator intentionally uses a local RNG so that the same
    seed produces the same episode without changing the randomness
    of the existing task generators.
    """

    family = "BoundaryPropagation"

    def generate_episode(
        self,
        difficulty: DifficultyConfig,
        seed: int
    ) -> Tuple[List[Demonstration], Query, List[List[int]]]:

        rng = random.Random(seed)

        n_demos = difficulty.n_demonstrations or 3
        grid_h, grid_w = difficulty.grid_size or [7, 7]

        # Keep the task large enough to make propagation meaningful.
        grid_h = max(5, grid_h)
        grid_w = max(5, grid_w)

        bg_color = 0
        prop_color = rng.randint(1, 9)

        demos = []

        for _ in range(n_demos):
            in_grid, out_grid = self._make_pair(
                grid_h,
                grid_w,
                prop_color,
                rng
            )

            demos.append(
                Demonstration(
                    input_grid=in_grid.tolist(),
                    output_grid=out_grid.tolist()
                )
            )

        q_in, q_out = self._make_pair(
            grid_h,
            grid_w,
            prop_color,
            rng
        )

        query = Query(input_grid=q_in.tolist())
        target = q_out.tolist()

        return demos, query, target

    def _make_pair(
        self,
        h: int,
        w: int,
        color: int,
        rng: random.Random
    ) -> Tuple[np.ndarray, np.ndarray]:

        grid = np.zeros((h, w), dtype=int)

        # Choose whether this example contains:
        #   1. top/bottom boundary seeds
        #   2. left/right boundary seeds
        #   3. a mixture of both
        pattern_type = rng.choice(
            ["vertical", "horizontal", "mixed"]
        )

        seeds = []

        if pattern_type in ("vertical", "mixed"):
            # Select one or two columns and place seeds
            # on the top or bottom boundary.
            n_vertical = rng.randint(1, 2)

            for _ in range(n_vertical):
                col = rng.randint(0, w - 1)
                row = rng.choice([0, h - 1])

                grid[row, col] = color
                seeds.append(("vertical", row, col))

        if pattern_type in ("horizontal", "mixed"):
            n_horizontal = rng.randint(1, 2)

            for _ in range(n_horizontal):
        # Avoid corners so boundary orientation is unambiguous.
                row = rng.randint(1, h - 2)
                col = rng.choice([0, w - 1])

                grid[row, col] = color
                seeds.append(("horizontal", row, col))

        out_grid = grid.copy()

        # Propagate boundary seeds.
        for direction, row, col in seeds:

            if direction == "vertical":
                # Boundary cell propagates through its column.
                out_grid[:, col] = color

            elif direction == "horizontal":
                # Boundary cell propagates through its row.
                out_grid[row, :] = color

        return grid, out_grid
