import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from backend.schemas import Demonstration, Query


class Transformation(nn.Module):
    """Base class for a transformation hypothesis."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TranslationTransformation(Transformation):
    def __init__(self, dr: int, dc: int):
        super().__init__()
        self.dr = dr
        self.dc = dc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return torch.roll(
                x,
                shifts=(self.dr, self.dc),
                dims=(0, 1),
            )

        if x.dim() == 3:
            return torch.roll(
                x,
                shifts=(self.dr, self.dc),
                dims=(1, 2),
            )

        return x


class ColorSwapTransformation(Transformation):
    def __init__(self, src: int, tgt: int):
        super().__init__()
        self.src = src
        self.tgt = tgt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[x == self.src] = self.tgt
        return out

class BoundaryPropagationTransformation(Transformation):
    """
    Propagate original colored boundary seeds.

    Rule:
    - Original top/bottom boundary seed -> fill its column.
    - Original left/right boundary seed -> fill its row.

    Important:
    Boundary cells are detected from the ORIGINAL input,
    not from the progressively modified output. This prevents
    newly created propagation cells from being mistaken for
    additional seeds.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()

        if x.dim() != 2:
            return out

        h, w = x.shape

        # Snapshot the original grid.
        original = x.clone()

        # Detect ORIGINAL top/bottom seeds.
        for c in range(w):
            if int(original[0, c].item()) != 0:
                out[:, c] = original[0, c]

            if int(original[h - 1, c].item()) != 0:
                out[:, c] = original[h - 1, c]

        # Detect ORIGINAL left/right seeds.
        for r in range(1, h - 1):
            if int(original[r, 0].item()) != 0:
                out[r, :] = original[r, 0]

            if int(original[r, w - 1].item()) != 0:
                out[r, :] = original[r, w - 1]

        return out
class TransformationHypothesisEngine(nn.Module):
    """
    Structured transformation hypothesis engine.

    The latent state H_R is used to generate transformation logits,
    while demonstration consistency provides the actual verification.

    The ground-truth query target is NEVER used during hypothesis
    generation or selection.
    """

    def __init__(self, d_model: int, n_classes: int = 10):
        super().__init__()

        self.d_model = d_model

        # Translation: 3 x 3 = 9 candidates
        # Color mapping: 9 x 9 = 81 candidates
        # Boundary propagation: 1 candidate

        self.num_candidates = 91

        self.generator = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_candidates),
        )

    def _instantiate_transformations(
        self,
        logits: torch.Tensor,
    ) -> List[Tuple[Transformation, float]]:
        """
        Instantiate the complete candidate space.

        Candidate ordering:

        0-8:
            translations (-1,0,1) x (-1,0,1)

        9-89:
            all non-background color mappings 1..9 -> 1..9

        Returns candidates together with their generator logit.
        """

        candidates = []

        # -------------------------------------------------
        # Translation candidates
        # -------------------------------------------------

        candidate_index = 0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                generator_score = (
                    logits[candidate_index].item()
                    if candidate_index < logits.numel()
                    else 0.0
                )

                candidates.append(
                    (
                        TranslationTransformation(dr, dc),
                        generator_score,
                    )
                )

                candidate_index += 1

        # -------------------------------------------------
        # Arbitrary color mappings
        # -------------------------------------------------

        for src in range(1, 10):
            for tgt in range(1, 10):

                generator_score = (
                    logits[candidate_index].item()
                    if candidate_index < logits.numel()
                    else 0.0
                )

                candidates.append(
                    (
                        ColorSwapTransformation(src, tgt),
                        generator_score,
                    )

                )

                candidate_index += 1

        # -------------------------------------------------
        # Boundary propagation
        # -------------------------------------------------

        boundary_logit = (
            logits[candidate_index].item()
            if candidate_index < logits.numel()
            else 0.0
        )

        candidates.append(
            (
                BoundaryPropagationTransformation(),
                boundary_logit,
            )
        )

        return candidates

    def _score_demonstration(
        self,
        transform: Transformation,
        demo: Demonstration,
    ) -> float:
        """
        Compute cell-level accuracy for one demonstration.
        """

        in_grid = torch.tensor(
            demo.input_grid,
            dtype=torch.long,
        )

        out_grid = torch.tensor(
            demo.output_grid,
            dtype=torch.long,
        )

        pred = transform(in_grid)

        if pred.shape != out_grid.shape:
            return 0.0

        return (
            (pred == out_grid)
            .float()
            .mean()
            .item()
        )

    def _verify_against_demonstrations(
        self,
        transform: Transformation,
        demos: List[Demonstration],
    ) -> Dict[str, Any]:
        """
        Verify a candidate independently against every demonstration.

        Consistency is deliberately based on the weakest demonstration
        score so that a transformation cannot win by doing extremely
        well on one demonstration while failing another.
        """

        if not demos:
            return {
                "demo_scores": [],
                "minimum_score": 1.0,
                "mean_score": 1.0,
                "exact_consistency": True,
            }

        demo_scores = [
            self._score_demonstration(transform, demo)
            for demo in demos
        ]

        minimum_score = min(demo_scores)
        mean_score = sum(demo_scores) / len(demo_scores)

        exact_consistency = all(
            score >= 1.0 - 1e-9
            for score in demo_scores
        )

        return {
            "demo_scores": demo_scores,
            "minimum_score": minimum_score,
            "mean_score": mean_score,
            "exact_consistency": exact_consistency,
        }
    def _type_name(self, transform: Transformation) -> str:
        if isinstance(transform, TranslationTransformation):
            return (
                f"Translation(dr={transform.dr}, "
                f"dc={transform.dc})"
            )

        if isinstance(transform, ColorSwapTransformation):
            return (
                f"ColorSwap(src={transform.src}, "
                f"tgt={transform.tgt})"
            )

        if isinstance(transform, BoundaryPropagationTransformation):
            return "BoundaryPropagation"

        return type(transform).__name__
    
    def forward(
        self,
        H_R: torch.Tensor,
        query: Query,
        demos: List[Demonstration],
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Generate, verify, rank, and apply transformation hypotheses.

        Selection priority:

        1. Exact consistency across all demonstrations.
        2. Minimum demonstration score.
        3. Mean demonstration score.
        4. Generator logit.

        The query target is not accessed here.
        """

        if H_R.dim() == 1:
            H_R = H_R.unsqueeze(0)

        logits = self.generator(H_R)[0]

        candidates = self._instantiate_transformations(
            logits
        )

        ranked = []

        for transform, generator_logit in candidates:

            verification = self._verify_against_demonstrations(
                transform,
                demos,
            )

            ranked.append(
                {
                    "transform": transform,
                    "type": self._type_name(transform),
                    "demo_scores": verification["demo_scores"],
                    "minimum_score": verification["minimum_score"],
                    "mean_score": verification["mean_score"],
                    "exact_consistency": verification["exact_consistency"],
                    "generator_logit": generator_logit,
                }
            )

        # -------------------------------------------------
        # Rigorous ranking
        # -------------------------------------------------

        ranked.sort(
            key=lambda item: (
                item["exact_consistency"],
                item["minimum_score"],
                item["mean_score"],
                item["generator_logit"],
            ),
            reverse=True,
        )

        best = ranked[0]

        best_transform = best["transform"]

        query_grid = torch.tensor(
            query.input_grid,
            dtype=torch.long,
        )

        best_prediction = best_transform(
            query_grid
        )

        # -------------------------------------------------
        # Interpretable verification trace
        # -------------------------------------------------

        trace = []

        for rank, item in enumerate(ranked[:10], start=1):
            trace.append(
                {
                    "rank": rank,
                    "type": item["type"],
                    "score": item["mean_score"],
                    "minimum_score": item["minimum_score"],
                    "demo_scores": item["demo_scores"],
                    "exact_consistency": item["exact_consistency"],
                    "generator_logit": item["generator_logit"],
                }
            )

        return best_prediction, trace