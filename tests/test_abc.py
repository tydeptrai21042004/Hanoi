import numpy as np

from qr64_certified.optimization import artificial_bee_colony_maximize


def _quadratic_objective(x):
    score = -float(np.sum((x - 0.25) ** 2))
    return score, {"score": score}


def test_abc_keeps_or_improves_initial_candidate():
    initial = [0.9, 0.9]
    initial_score = _quadratic_objective(np.asarray(initial))[0]
    result = artificial_bee_colony_maximize(
        _quadratic_objective,
        [(0.0, 1.0), (0.0, 1.0)],
        initial_position=initial,
        food_sources=4,
        cycles=3,
        seed=1,
    )
    assert result.best_score >= initial_score
    assert len(result.history) == 3
    assert np.all(result.best_position >= 0.0)
    assert np.all(result.best_position <= 1.0)
    phases = {entry["phase"] for entry in result.evaluations}
    assert {"initial", "employed", "onlooker"}.issubset(phases)


def test_abc_is_deterministic_for_fixed_seed():
    kwargs = dict(
        objective=_quadratic_objective,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        initial_position=[0.8, 0.7],
        food_sources=5,
        cycles=4,
        seed=2026,
        limit=3,
        onlooker_bees=5,
    )
    first = artificial_bee_colony_maximize(**kwargs)
    second = artificial_bee_colony_maximize(**kwargs)
    np.testing.assert_allclose(first.best_position, second.best_position)
    assert first.best_score == second.best_score
    assert first.history == second.history
    assert first.evaluations == second.evaluations


def test_abc_scout_phase_respects_bounds():
    def flat_objective(x):
        return 0.0, {"sum": float(np.sum(x))}

    result = artificial_bee_colony_maximize(
        flat_objective,
        [(-2.0, -1.0), (3.0, 4.0)],
        initial_position=[-1.5, 3.5],
        food_sources=3,
        cycles=2,
        seed=7,
        limit=1,
        onlooker_bees=3,
    )
    scouts = [entry for entry in result.evaluations if entry["phase"] == "scout"]
    assert scouts
    for entry in result.evaluations:
        x = np.asarray(entry["position"])
        assert -2.0 <= x[0] <= -1.0
        assert 3.0 <= x[1] <= 4.0
