from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_predictions_contract_if_present() -> None:
    path = Path("outputs/predictions.json")
    if not path.exists():
        return
    df = pd.read_json(path)
    assert len(df) == 104
    assert {"match_id", "team1", "team2", "prob_team1_win", "prob_draw", "prob_team2_win"}.issubset(df.columns)
    assert df["match_id"].is_unique
