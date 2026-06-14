from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc2026.data import china_kickoff


def test_predictions_contract_if_present() -> None:
    path = Path("outputs/predictions.json")
    if not path.exists():
        return
    df = pd.read_json(path)
    assert len(df) == 104
    assert {"match_id", "team1", "team2", "prob_team1_win", "prob_draw", "prob_team2_win"}.issubset(df.columns)
    assert df["match_id"].is_unique


def test_host_date_converts_to_china_date() -> None:
    kickoff = china_kickoff("2026-06-14", "12:00 UTC-5")
    assert kickoff is not None
    assert kickoff.strftime("%Y-%m-%d %H:%M") == "2026-06-15 01:00"
