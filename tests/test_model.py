from __future__ import annotations

import numpy as np

from wc2026.model import PredictorParams, PredictorState, TeamRating, predict_match, score_matrix


def dummy_state() -> PredictorState:
    return PredictorState(
        params=PredictorParams(),
        teams={
            "A": TeamRating("A", 1600, 1.1, 0.9, 50, 10),
            "B": TeamRating("B", 1500, 1.0, 1.0, 50, 10),
        },
        trained_rows=100,
    )


def test_score_matrix_sums_to_one() -> None:
    matrix = score_matrix(dummy_state(), "A", "B")
    assert matrix.shape == (8, 8)
    assert np.isclose(matrix.sum(), 1.0)


def test_match_probabilities_sum_to_one() -> None:
    pred = predict_match(dummy_state(), "A", "B")
    total = pred["prob_team1_win"] + pred["prob_draw"] + pred["prob_team2_win"]
    assert abs(total - 1.0) < 1e-5
    assert len(pred["top_scores"]) == 5
