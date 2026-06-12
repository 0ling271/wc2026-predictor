from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .model import PredictorState, predict_match, score_matrix
from .paths import OUTPUTS_DIR, ensure_dirs
from .simulate import TeamRecord, _add_result, _best_thirds, _resolve_knockout_team


def run_monte_carlo(state: PredictorState, fixtures: pd.DataFrame, simulations: int = 10000, seed: int = 42) -> pd.DataFrame:
    ensure_dirs()
    rng = np.random.default_rng(seed)
    group_fixtures = fixtures[fixtures["match_id"] <= 72]
    teams = sorted(set(group_fixtures["team1"]).union(group_fixtures["team2"]))
    teams = [team for team in teams if team]
    counters: dict[str, Counter] = {team: Counter() for team in teams}

    score_cache = _score_cache(state, fixtures)
    pred_cache: dict[tuple[str, str, str], dict] = {}
    for _ in range(simulations):
        placements, group_table = _simulate_groups(fixtures, score_cache, rng, counters)
        best_thirds = _best_thirds(placements, group_table)
        used_thirds: set[str] = set()
        winners: dict[int, str] = {}
        losers: dict[int, str] = {}
        for row in fixtures[fixtures["match_id"] > 72].itertuples(index=False):
            team1 = _resolve_knockout_team(str(row.team1), placements, best_thirds, used_thirds, winners, losers)
            team2 = _resolve_knockout_team(str(row.team2), placements, best_thirds, used_thirds, winners, losers)
            winner = _sample_knockout_winner(state, team1, team2, row.ground, rng, pred_cache)
            loser = team2 if winner == team1 else team1
            winners[int(row.match_id)] = winner
            losers[int(row.match_id)] = loser
            _mark_stage(counters, winner, int(row.match_id))
            _mark_stage(counters, loser, int(row.match_id), loser=True)

    rows = []
    for team in sorted(counters):
        row = {"team": team}
        for key in ["advance_r32", "advance_r16", "quarterfinal", "semifinal", "final", "champion"]:
            row[f"prob_{key}"] = round(counters[team][key] / simulations, 6)
        rows.append(row)
    odds = pd.DataFrame(rows).sort_values("prob_champion", ascending=False).reset_index(drop=True)
    odds.to_csv(OUTPUTS_DIR / "tournament_odds.csv", index=False, encoding="utf-8")
    (OUTPUTS_DIR / "tournament_odds.json").write_text(odds.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    return odds


def _score_cache(state: PredictorState, fixtures: pd.DataFrame) -> dict[int, tuple[np.ndarray, list[tuple[int, int]]]]:
    cache = {}
    for row in fixtures[fixtures["match_id"] <= 72].itertuples(index=False):
        matrix = score_matrix(state, row.team1, row.team2, row.ground)
        scores = [(i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
        cache[int(row.match_id)] = (matrix.reshape(-1), scores)
    return cache


def _simulate_groups(fixtures: pd.DataFrame, score_cache: dict, rng: np.random.Generator, counters: dict[str, Counter]):
    records: dict[str, TeamRecord] = {}
    for row in fixtures[fixtures["match_id"] <= 72].itertuples(index=False):
        if pd.notna(row.actual_goals1) and pd.notna(row.actual_goals2):
            g1, g2 = int(row.actual_goals1), int(row.actual_goals2)
        else:
            probs, scores = score_cache[int(row.match_id)]
            g1, g2 = scores[int(rng.choice(len(scores), p=probs))]
        _add_result(records, str(row.group), row.team1, row.team2, g1, g2)
    table = pd.DataFrame([r.__dict__ | {"gd": r.gd} for r in records.values()])
    table["group_letter"] = table["group"].str.extract(r"Group ([A-L])")[0]
    table = table.sort_values(
        ["group_letter", "points", "gd", "gf", "wins", "team"],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)
    placements = {}
    for group, group_df in table.groupby("group_letter", sort=True):
        ordered = list(group_df["team"])
        placements[group] = ordered
        for team in ordered[:2]:
            counters[team]["advance_r32"] += 1
    best_thirds = _best_thirds(placements, table)
    for team in best_thirds:
        counters[team]["advance_r32"] += 1
    return placements, table


def _sample_knockout_winner(
    state: PredictorState,
    team1: str,
    team2: str,
    ground: str,
    rng: np.random.Generator,
    pred_cache: dict[tuple[str, str, str], dict],
) -> str:
    key = (team1, team2, str(ground))
    if key not in pred_cache:
        pred_cache[key] = predict_match(state, team1, team2, ground)
    pred = pred_cache[key]
    probs = np.array([pred["prob_team1_win"], pred["prob_draw"], pred["prob_team2_win"]], dtype=float)
    probs = probs / probs.sum()
    outcome = int(rng.choice(3, p=probs))
    if outcome == 0:
        return team1
    if outcome == 2:
        return team2
    coin = pred["expected_goals1"] / max(pred["expected_goals1"] + pred["expected_goals2"], 1e-9)
    return team1 if rng.random() < coin else team2


def _mark_stage(counters: dict[str, Counter], team: str, match_id: int, loser: bool = False) -> None:
    if match_id <= 88:
        counters[team]["advance_r16"] += 1 if not loser else 0
    elif match_id <= 96:
        counters[team]["quarterfinal"] += 1 if not loser else 0
    elif match_id <= 100:
        counters[team]["semifinal"] += 1 if not loser else 0
    elif match_id in {101, 102}:
        counters[team]["final"] += 1 if not loser else 0
    if match_id == 104 and not loser:
        counters[team]["champion"] += 1
