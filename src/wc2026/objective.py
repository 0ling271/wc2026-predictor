from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .model import PredictorParams, _apply_score_shape_adjustments, _headline_score_for_outcome, load_model
from .paths import OUTPUTS_DIR, PROCESSED_DIR, ensure_dirs


def load_adjustments(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["match_id"]): item for item in payload["matches"]}


def objective_predict(date: str, adjustments_path: Path | None = None) -> pd.DataFrame:
    ensure_dirs()
    predictions_path = OUTPUTS_DIR / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError("Run predict first: missing outputs/predictions.json")
    predictions = pd.read_json(predictions_path, convert_dates=False)
    adjustments_path = adjustments_path or PROCESSED_DIR / f"objective_adjustments_{date}.json"
    adjustments = load_adjustments(adjustments_path)
    rows = []
    date_column = "china_date" if "china_date" in predictions.columns else "date"
    for row in predictions[predictions[date_column].astype(str).eq(date)].itertuples(index=False):
        item = adjustments.get(int(row.match_id))
        if not item:
            continue
        rows.append(_adjust_row(row._asdict(), item))
    output = pd.DataFrame(rows)
    if {"china_date", "china_time"}.issubset(output.columns) and not output.empty:
        output["_kickoff_sort"] = pd.to_datetime(
            output["china_date"].astype(str) + " " + output["china_time"].astype(str),
            errors="coerce",
        )
        output = output.sort_values(["_kickoff_sort", "match_id"], na_position="last").drop(columns=["_kickoff_sort"])
    output.to_csv(OUTPUTS_DIR / f"objective_predictions_{date}.csv", index=False, encoding="utf-8")
    (OUTPUTS_DIR / f"objective_predictions_{date}.json").write_text(
        output.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def _adjust_row(row: dict, item: dict) -> dict:
    team1 = row["team1"]
    team2 = row["team2"]
    team1_adj = float(item.get("team1_xg_delta", 0.0))
    team2_adj = float(item.get("team2_xg_delta", 0.0))
    tempo = float(item.get("tempo_delta", 0.0))
    mu1 = max(0.15, float(row["expected_goals1"]) + team1_adj + tempo / 2.0)
    mu2 = max(0.15, float(row["expected_goals2"]) + team2_adj + tempo / 2.0)
    matrix = _score_matrix(mu1, mu2)
    p1 = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    p2 = float(np.triu(matrix, 1).sum())
    base = np.array([float(row["prob_team1_win"]), float(row["prob_draw"]), float(row["prob_team2_win"])])
    objective = np.array([p1, draw, p2])
    weight = float(item.get("objective_weight", 0.55))
    blended = (1.0 - weight) * base + weight * objective
    blended = blended / blended.sum()
    flat = sorted(
        [(float(matrix[i, j]), i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])],
        reverse=True,
    )
    outcome = int(np.argmax(blended))
    score = _headline_score_for_outcome(flat, outcome, mu1, mu2)
    return {
        "match_id": int(row["match_id"]),
        "date": row["date"],
        "time": row["time"],
        "china_date": row.get("china_date", row["date"]),
        "china_time": row.get("china_time", ""),
        "ground": row["ground"],
        "team1": team1,
        "team2": team2,
        "base_score": row["most_likely_score"],
        "base_prediction": row["predicted_winner"],
        "base_prob_team1_win": row["prob_team1_win"],
        "base_prob_draw": row["prob_draw"],
        "base_prob_team2_win": row["prob_team2_win"],
        "adjusted_expected_goals1": round(mu1, 3),
        "adjusted_expected_goals2": round(mu2, 3),
        "adjusted_prob_team1_win": round(float(blended[0]), 6),
        "adjusted_prob_draw": round(float(blended[1]), 6),
        "adjusted_prob_team2_win": round(float(blended[2]), 6),
        "adjusted_score": f"{score[1]}-{score[2]}",
        "adjusted_prediction": [team1, "Draw", team2][outcome],
        "objective_notes": " | ".join(item.get("notes", [])),
        "sources": " | ".join(item.get("sources", [])),
    }


def _score_matrix(mu1: float, mu2: float) -> np.ndarray:
    try:
        params = load_model().params
    except Exception:
        params = PredictorParams()
    max_goals = params.max_goals
    probs1 = poisson.pmf(np.arange(max_goals + 1), mu1)
    probs2 = poisson.pmf(np.arange(max_goals + 1), mu2)
    matrix = np.outer(probs1, probs2)
    matrix = _apply_score_shape_adjustments(matrix, params)
    return matrix / matrix.sum()
