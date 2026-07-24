import numpy as np
from qr64_certified.optimization import particle_swarm_maximize


def test_pso_keeps_or_improves_initial_candidate():
    def objective(x):
        score = -float(np.sum((x - 0.25) ** 2))
        return score, {"score": score}
    initial = [0.9, 0.9]
    initial_score = objective(np.asarray(initial))[0]
    result = particle_swarm_maximize(
        objective, [(0.0, 1.0), (0.0, 1.0)], initial_position=initial,
        particles=4, iterations=3, seed=1,
    )
    assert result.best_score >= initial_score
    assert len(result.history) == 3
