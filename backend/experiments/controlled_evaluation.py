import csv
import sys
from pathlib import Path

from backend.api.main import EpisodeRequest, run_episode


FAMILIES = [
    "ColorTransformation",
    "Translation",
    "BoundaryPropagation",
]

MEMORY_MODES = [
    "baseline",
    "simple_hebbian",
]

REASONING_DEPTHS = [
    1,
    4,
    8,
]

SEEDS = list(range(1, 21))

GRID_SIZE = [5, 5]
N_DEMONSTRATIONS = 3


def run_controlled_evaluation():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "controlled_evaluation.csv"

    fieldnames = [
        "family",
        "memory_mode",
        "reasoning_depth",
        "grid_rows",
        "grid_cols",
        "demonstrations",
        "seed",
        "exact_match",
        "cell_accuracy",
        "correct_cells",
        "total_cells",
        "memory_norm",
        "effective_rank",
        "entropy",
        "interference_score",
        "reconstruction_error",
        "mean_sparsity",
        "reasoning_steps",
        "halting_step",
    ]

    total_runs = (
        len(FAMILIES)
        * len(MEMORY_MODES)
        * len(REASONING_DEPTHS)
        * len(SEEDS)
    )

    completed = 0

    print("=" * 70)
    print("CONTROLLED EVALUATION")
    print("=" * 70)
    print(f"Families:          {FAMILIES}")
    print(f"Memory modes:      {MEMORY_MODES}")
    print(f"Reasoning depths:  {REASONING_DEPTHS}")
    print(f"Seeds:             {SEEDS[0]}-{SEEDS[-1]}")
    print(f"Grid size:         {GRID_SIZE}")
    print(f"Demonstrations:    {N_DEMONSTRATIONS}")
    print(f"Total episodes:    {total_runs}")
    print("=" * 70)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for family in FAMILIES:
            for memory_mode in MEMORY_MODES:
                for depth in REASONING_DEPTHS:
                    for seed in SEEDS:

                        req = EpisodeRequest(
                            family=family,
                            grid_size=GRID_SIZE,
                            n_demonstrations=N_DEMONSTRATIONS,
                            memory_mode=memory_mode,
                            reasoning_depth=depth,
                            adaptive_halt=False,
                            reread_memory=True,
                            seed=seed,
                        )

                        result = run_episode(req)

                        evaluation = result["evaluation"]
                        telemetry = result["telemetry"]

                        memory = telemetry.get("memory", {})
                        neurons = telemetry.get("neurons", {})
                        reasoning = telemetry.get("reasoning", {})

                        row = {
                            "family": family,
                            "memory_mode": memory_mode,
                            "reasoning_depth": depth,
                            "grid_rows": GRID_SIZE[0],
                            "grid_cols": GRID_SIZE[1],
                            "demonstrations": N_DEMONSTRATIONS,
                            "seed": seed,
                            "exact_match": evaluation["exact_match"],
                            "cell_accuracy": evaluation["cell_accuracy"],
                            "correct_cells": evaluation["correct_cells"],
                            "total_cells": evaluation["total_cells"],
                            "memory_norm": memory.get("norm", 0.0),
                            "effective_rank": memory.get("effective_rank", 0.0),
                            "entropy": memory.get("entropy", 0.0),
                            "interference_score": memory.get(
                                "interference_score", 0.0
                            ),
                            "reconstruction_error": memory.get(
                                "reconstruction_error", 0.0
                            ),
                            "mean_sparsity": neurons.get(
                                "mean_sparsity", 0.0
                            ),
                            "reasoning_steps": reasoning.get(
                                "trajectory_length", 0
                            ),
                            "halting_step": reasoning.get(
                                "halting_step", -1
                            ),
                        }

                        writer.writerow(row)
                        f.flush()

                        completed += 1

                        print(
                            f"[{completed:3d}/{total_runs}] "
                            f"{family:22s} "
                            f"{memory_mode:18s} "
                            f"depth={depth:2d} "
                            f"seed={seed:2d} "
                            f"accuracy={evaluation['cell_accuracy']:.3f} "
                            f"exact={evaluation['exact_match']}"
                        )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Saved to: {output_file}")
    print(f"Episodes: {completed}")


if __name__ == "__main__":
    run_controlled_evaluation()