from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pandas as pd

from .model import PredictorState, choose_winner, predict_match
from .paths import OUTPUTS_DIR, ensure_dirs


GROUP_RE = re.compile(r"Group ([A-L])")


@dataclass
class TeamRecord:
    team: str
    group: str
    points: float = 0
    gf: float = 0
    ga: float = 0
    wins: float = 0
    draws: float = 0
    losses: float = 0

    @property
    def gd(self) -> float:
        return self.gf - self.ga


def _predicted_score(pred: dict) -> tuple[int, int]:
    a, b = pred["most_likely_score"].split("-")
    return int(a), int(b)


def _group_letter(group: str) -> str:
    match = GROUP_RE.search(str(group))
    return match.group(1) if match else ""


def _add_result(records: dict[str, TeamRecord], group: str, t1: str, t2: str, g1: int, g2: int) -> None:
    for team in [t1, t2]:
        records.setdefault(team, TeamRecord(team=team, group=group))
    r1, r2 = records[t1], records[t2]
    r1.gf += g1
    r1.ga += g2
    r2.gf += g2
    r2.ga += g1
    if g1 > g2:
        r1.points += 3
        r1.wins += 1
        r2.losses += 1
    elif g2 > g1:
        r2.points += 3
        r2.wins += 1
        r1.losses += 1
    else:
        r1.points += 1
        r2.points += 1
        r1.draws += 1
        r2.draws += 1


def build_group_tables(fixtures: pd.DataFrame, predictions: list[dict]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    records: dict[str, TeamRecord] = {}
    pred_by_id = {p["match_id"]: p for p in predictions}
    group_matches = fixtures[fixtures["match_id"] <= 72]
    for row in group_matches.itertuples(index=False):
        pred = pred_by_id[int(row.match_id)]
        if pd.notna(row.actual_goals1) and pd.notna(row.actual_goals2):
            g1, g2 = int(row.actual_goals1), int(row.actual_goals2)
        else:
            g1, g2 = _predicted_score(pred)
        _add_result(records, str(row.group), row.team1, row.team2, g1, g2)

    table = pd.DataFrame([r.__dict__ | {"gd": r.gd} for r in records.values()])
    table["group_letter"] = table["group"].map(_group_letter)
    table = table.sort_values(
        ["group_letter", "points", "gd", "gf", "wins", "team"],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)
    placements: dict[str, list[str]] = {}
    for group, group_df in table.groupby("group_letter", sort=True):
        placements[group] = list(group_df["team"])
    return placements, table


def _best_thirds(placements: dict[str, list[str]], table: pd.DataFrame) -> list[str]:
    third_rows = []
    for group, teams in placements.items():
        if len(teams) >= 3:
            third_rows.append(table[(table["group_letter"] == group) & (table["team"] == teams[2])].iloc[0])
    if not third_rows:
        return []
    thirds = pd.DataFrame(third_rows)
    thirds = thirds.sort_values(["points", "gd", "gf", "wins", "team"], ascending=[False, False, False, False, True])
    return list(thirds.head(8)["team"])


def _resolve_slot(slot: str, placements: dict[str, list[str]], best_thirds: list[str], used_thirds: set[str]) -> str:
    slot = str(slot)
    if re.fullmatch(r"[12][A-L]", slot):
        pos = int(slot[0]) - 1
        group = slot[1]
        return placements[group][pos]
    if slot.startswith("3") and "/" in slot:
        allowed = slot[1:].split("/")
        for team in best_thirds:
            group = next((g for g, teams in placements.items() if len(teams) >= 3 and teams[2] == team), "")
            if group in allowed and team not in used_thirds:
                used_thirds.add(team)
                return team
        for group in allowed:
            candidate = placements[group][2]
            if candidate not in used_thirds:
                used_thirds.add(candidate)
                return candidate
        for team in best_thirds:
            if team not in used_thirds:
                used_thirds.add(team)
                return team
        for group in allowed:
            if group in placements and len(placements[group]) >= 3:
                return placements[group][2]
    return slot


def predict_tournament(state: PredictorState, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    predictions: list[dict] = []
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}

    for row in fixtures[fixtures["match_id"] <= 72].itertuples(index=False):
        pred = predict_match(state, row.team1, row.team2, row.ground)
        pred.update(_row_meta(row))
        if pd.notna(row.actual_goals1) and pd.notna(row.actual_goals2):
            pred["status"] = "actual"
            pred["predicted_winner"] = _result_label(pred)
            pred["actual_result"] = _actual_winner(row.team1, row.team2, int(row.actual_goals1), int(row.actual_goals2))
            pred["actual_score"] = f"{int(row.actual_goals1)}-{int(row.actual_goals2)}"
        else:
            pred["status"] = "predicted"
            pred["predicted_winner"] = _result_label(pred)
            pred["actual_result"] = ""
            pred["actual_score"] = ""
        predictions.append(pred)

    placements, group_table = build_group_tables(fixtures, predictions)
    best_thirds = _best_thirds(placements, group_table)
    used_thirds: set[str] = set()

    for row in fixtures[fixtures["match_id"] > 72].itertuples(index=False):
        team1 = _resolve_knockout_team(str(row.team1), placements, best_thirds, used_thirds, winners, losers)
        team2 = _resolve_knockout_team(str(row.team2), placements, best_thirds, used_thirds, winners, losers)
        pred = predict_match(state, team1, team2, row.ground)
        pred.update(_row_meta(row))
        pred["team1"] = team1
        pred["team2"] = team2
        pred["status"] = "predicted"
        pred["actual_score"] = ""
        pred["actual_result"] = ""
        winner = choose_winner(pred)
        loser = team2 if winner == team1 else team1
        winners[int(row.match_id)] = winner
        losers[int(row.match_id)] = loser
        pred["predicted_winner"] = winner
        predictions.append(pred)

    df = pd.DataFrame(predictions).sort_values("match_id").reset_index(drop=True)
    df["top_scores_json"] = df["top_scores"].map(lambda value: json.dumps(value, ensure_ascii=False))
    df.drop(columns=["top_scores"]).to_csv(OUTPUTS_DIR / "predictions.csv", index=False, encoding="utf-8")
    (OUTPUTS_DIR / "predictions.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    group_table.to_csv(OUTPUTS_DIR / "group_tables.csv", index=False, encoding="utf-8")
    bracket = {
        "champion": winners.get(104, ""),
        "final": {"match_id": 104, "winner": winners.get(104, ""), "loser": losers.get(104, "")},
        "winners": winners,
    }
    (OUTPUTS_DIR / "bracket.json").write_text(json.dumps(bracket, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, group_table


def _row_meta(row) -> dict:
    return {
        "match_id": int(row.match_id),
        "round": row.round,
        "date": row.date,
        "time": row.time,
        "group": row.group,
        "ground": row.ground,
    }


def _actual_winner(team1: str, team2: str, g1: int, g2: int) -> str:
    if g1 > g2:
        return team1
    if g2 > g1:
        return team2
    return "Draw"


def _result_label(pred: dict) -> str:
    probs = {
        pred["team1"]: pred["prob_team1_win"],
        "Draw": pred["prob_draw"],
        pred["team2"]: pred["prob_team2_win"],
    }
    return max(probs, key=probs.get)


def _resolve_knockout_team(
    slot: str,
    placements: dict[str, list[str]],
    best_thirds: list[str],
    used_thirds: set[str],
    winners: dict[int, str],
    losers: dict[int, str],
) -> str:
    if re.fullmatch(r"W\d+", slot):
        return winners[int(slot[1:])]
    if re.fullmatch(r"L\d+", slot):
        return losers[int(slot[1:])]
    return _resolve_slot(slot, placements, best_thirds, used_thirds)
