from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ABCResult:
    """Result of deterministic Artificial Bee Colony maximization."""

    best_position: np.ndarray
    best_score: float
    history: list[dict[str, float]]
    evaluations: list[dict[str, object]]


def _validate_problem(
    bounds: Sequence[tuple[float, float]],
    initial_position: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
    high = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    if low.ndim != 1 or low.size == 0:
        raise ValueError("at least one parameter bound is required")
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        raise ValueError("all bounds must be finite")
    if np.any(high <= low):
        raise ValueError("each upper bound must exceed its lower bound")

    initial = np.asarray(initial_position, dtype=np.float64)
    if initial.shape != low.shape:
        raise ValueError("initial_position has the wrong dimension")
    if np.any(~np.isfinite(initial)):
        raise ValueError("initial_position must contain finite values")
    return low, high, np.clip(initial, low, high)


def _fitness_probabilities(scores: np.ndarray) -> np.ndarray:
    """Convert maximization scores to standard positive ABC fitness values."""
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty vector")
    if np.any(~np.isfinite(scores)):
        raise ValueError("objective scores must be finite")

    fitness = np.where(
        scores >= 0.0,
        1.0 + scores,
        1.0 / (1.0 + np.abs(scores)),
    )
    total = float(np.sum(fitness))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(scores.size, 1.0 / scores.size, dtype=np.float64)
    return fitness / total


def _neighbor(
    positions: np.ndarray,
    source_index: int,
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, int, int, float]:
    """Generate one canonical ABC neighbour in a single random dimension."""
    source_count, dimensions = positions.shape
    partner = int(rng.integers(0, source_count - 1))
    if partner >= source_index:
        partner += 1
    dimension = int(rng.integers(0, dimensions))
    phi = float(rng.uniform(-1.0, 1.0))

    candidate = positions[source_index].copy()
    candidate[dimension] = (
        positions[source_index, dimension]
        + phi
        * (
            positions[source_index, dimension]
            - positions[partner, dimension]
        )
    )
    candidate = np.clip(candidate, low, high)
    return candidate, partner, dimension, phi


def artificial_bee_colony_maximize(
    objective: Callable[[np.ndarray], tuple[float, dict[str, object]]],
    bounds: Sequence[tuple[float, float]],
    *,
    initial_position: Sequence[float],
    food_sources: int = 5,
    cycles: int = 4,
    seed: int = 2026,
    limit: int | None = None,
    onlooker_bees: int | None = None,
) -> ABCResult:
    """Maximize an objective with a deterministic Artificial Bee Colony search.

    The validated pre-optimization setting is inserted as food source zero.
    The global best is retained independently of scout replacement, so the ABC
    result can never be worse than the supplied starting configuration on the
    evaluated objective.

    Parameters
    ----------
    objective:
        Callable returning ``(score, details)`` for one parameter vector.
    bounds:
        Inclusive lower and upper bounds for every optimized parameter.
    initial_position:
        Existing validated configuration used as the first food source.
    food_sources:
        Number of employed bees / candidate food sources.
    cycles:
        Number of employed-onlooker-scout cycles.
    seed:
        Random seed for reproducibility.
    limit:
        Abandonment count before a source becomes a scout. The default is
        ``food_sources * number_of_dimensions``.
    onlooker_bees:
        Number of onlooker trials per cycle. Defaults to ``food_sources``.
    """
    if food_sources < 2:
        raise ValueError("food_sources must be at least 2")
    if cycles < 1:
        raise ValueError("cycles must be at least 1")

    low, high, initial = _validate_problem(bounds, initial_position)
    dimensions = int(low.size)
    abandonment_limit = (
        int(limit) if limit is not None else int(food_sources * dimensions)
    )
    if abandonment_limit < 1:
        raise ValueError("limit must be at least 1")
    onlooker_count = food_sources if onlooker_bees is None else int(onlooker_bees)
    if onlooker_count < 1:
        raise ValueError("onlooker_bees must be at least 1")

    rng = np.random.default_rng(seed)
    positions = rng.uniform(low, high, size=(food_sources, dimensions))
    positions[0] = initial
    scores = np.full(food_sources, -np.inf, dtype=np.float64)
    trials = np.zeros(food_sources, dtype=np.int64)

    best_position = initial.copy()
    best_score = -np.inf
    history: list[dict[str, float]] = []
    evaluations: list[dict[str, object]] = []

    def evaluate(
        position: np.ndarray,
        *,
        cycle: int,
        phase: str,
        bee: int,
        source: int,
        partner: int | None = None,
        dimension: int | None = None,
        phi: float | None = None,
    ) -> tuple[float, dict[str, object]]:
        nonlocal best_position, best_score
        score, details = objective(position.copy())
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("objective returned a non-finite score")
        record: dict[str, object] = {
            "cycle": cycle,
            "phase": phase,
            "bee": bee,
            "source": source,
            "position": position.tolist(),
            "score": score,
            "details": details,
        }
        if partner is not None:
            record["partner"] = partner
        if dimension is not None:
            record["dimension"] = dimension
        if phi is not None:
            record["phi"] = phi
        evaluations.append(record)
        if score > best_score:
            best_score = score
            best_position = position.copy()
        return score, details

    # Initial food-source evaluation includes the frozen validated candidate.
    for source in range(food_sources):
        scores[source], _ = evaluate(
            positions[source], cycle=-1, phase="initial", bee=source, source=source
        )

    for cycle in range(cycles):
        # Employed-bee phase.
        for source in range(food_sources):
            candidate, partner, dimension, phi = _neighbor(
                positions, source, rng, low, high
            )
            candidate_score, _ = evaluate(
                candidate,
                cycle=cycle,
                phase="employed",
                bee=source,
                source=source,
                partner=partner,
                dimension=dimension,
                phi=phi,
            )
            if candidate_score > scores[source]:
                positions[source] = candidate
                scores[source] = candidate_score
                trials[source] = 0
            else:
                trials[source] += 1

        # Onlooker-bee phase.
        for bee in range(onlooker_count):
            probabilities = _fitness_probabilities(scores)
            source = int(rng.choice(food_sources, p=probabilities))
            candidate, partner, dimension, phi = _neighbor(
                positions, source, rng, low, high
            )
            candidate_score, _ = evaluate(
                candidate,
                cycle=cycle,
                phase="onlooker",
                bee=bee,
                source=source,
                partner=partner,
                dimension=dimension,
                phi=phi,
            )
            if candidate_score > scores[source]:
                positions[source] = candidate
                scores[source] = candidate_score
                trials[source] = 0
            else:
                trials[source] += 1

        # Scout phase. The global best is kept even when an exhausted source is
        # abandoned, which protects the validated starting configuration.
        scout_count = 0
        for source in range(food_sources):
            if trials[source] < abandonment_limit:
                continue
            positions[source] = rng.uniform(low, high, size=dimensions)
            scores[source], _ = evaluate(
                positions[source],
                cycle=cycle,
                phase="scout",
                bee=scout_count,
                source=source,
            )
            trials[source] = 0
            scout_count += 1

        history.append(
            {
                "cycle": float(cycle),
                "best_score": float(best_score),
                "mean_score": float(np.mean(scores)),
                "max_trial_count": float(np.max(trials)),
                "scout_count": float(scout_count),
            }
        )

    return ABCResult(
        best_position=best_position,
        best_score=float(best_score),
        history=history,
        evaluations=evaluations,
    )


__all__ = ["ABCResult", "artificial_bee_colony_maximize"]
