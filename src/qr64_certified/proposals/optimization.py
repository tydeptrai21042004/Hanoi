from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class PSOResult:
    best_position: np.ndarray
    best_score: float
    history: list[dict[str, float]]
    evaluations: list[dict[str, object]]


def particle_swarm_maximize(
    objective: Callable[[np.ndarray], tuple[float, dict[str, object]]],
    bounds: Sequence[tuple[float, float]],
    *,
    initial_position: Sequence[float],
    particles: int = 5,
    iterations: int = 4,
    seed: int = 2026,
    inertia: float = 0.65,
    cognitive: float = 1.45,
    social: float = 1.45,
) -> PSOResult:
    """Small deterministic PSO used for parameter selection.

    The frozen pre-optimization setting is inserted as the first particle, so
    optimization cannot silently discard the reported baseline.
    """
    if particles < 2 or iterations < 1:
        raise ValueError("particles >= 2 and iterations >= 1 are required")
    low = np.asarray([b[0] for b in bounds], dtype=np.float64)
    high = np.asarray([b[1] for b in bounds], dtype=np.float64)
    if np.any(high <= low):
        raise ValueError("each upper bound must exceed its lower bound")
    initial = np.asarray(initial_position, dtype=np.float64)
    if initial.shape != low.shape:
        raise ValueError("initial_position has the wrong dimension")

    rng = np.random.default_rng(seed)
    positions = rng.uniform(low, high, size=(particles, low.size))
    positions[0] = np.clip(initial, low, high)
    velocities = rng.uniform(-(high - low), high - low, size=positions.shape) * 0.12
    velocities[0] = 0.0

    pbest = positions.copy()
    pbest_scores = np.full(particles, -np.inf, dtype=np.float64)
    gbest = positions[0].copy()
    gbest_score = -np.inf
    history: list[dict[str, float]] = []
    evaluations: list[dict[str, object]] = []

    for iteration in range(iterations):
        iteration_scores: list[float] = []
        for index in range(particles):
            score, details = objective(positions[index].copy())
            score = float(score)
            record = {
                "iteration": iteration,
                "particle": index,
                "position": positions[index].tolist(),
                "score": score,
                "details": details,
            }
            evaluations.append(record)
            iteration_scores.append(score)
            if score > pbest_scores[index]:
                pbest_scores[index] = score
                pbest[index] = positions[index]
            if score > gbest_score:
                gbest_score = score
                gbest = positions[index].copy()

        history.append(
            {
                "iteration": float(iteration),
                "best_score": float(gbest_score),
                "mean_score": float(np.mean(iteration_scores)),
            }
        )
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (
            inertia * velocities
            + cognitive * r1 * (pbest - positions)
            + social * r2 * (gbest[None, :] - positions)
        )
        positions = np.clip(positions + velocities, low, high)

    return PSOResult(
        best_position=gbest,
        best_score=float(gbest_score),
        history=history,
        evaluations=evaluations,
    )
