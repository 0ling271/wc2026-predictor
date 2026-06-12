from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .data import is_host_advantage
from .paths import MODELS_DIR, ensure_dirs


IMPORTANT_TOURNAMENTS = {
    "FIFA World Cup": 1.7,
    "FIFA World Cup qualification": 1.35,
    "UEFA Euro": 1.35,
    "Copa America": 1.3,
    "African Cup of Nations": 1.25,
    "AFC Asian Cup": 1.2,
    "CONCACAF Gold Cup": 1.2,
    "Oceania Nations Cup": 1.15,
    "UEFA Nations League": 1.1,
    "Friendly": 0.7,
}


@dataclass
class TeamRating:
    team: str
    elo: float
    attack: float
    defense: float
    matches: int
    recent_matches: int


@dataclass
class PredictorParams:
    base_home_goals: float = 1.34
    base_away_goals: float = 1.10
    elo_goal_scale: float = 0.0032
    attack_weight: float = 0.32
    defense_weight: float = 0.24
    host_goal_bonus: float = 0.13
    ml_blend_weight: float = 0.22
    max_goals: int = 7


@dataclass
class PredictorState:
    params: PredictorParams
    teams: dict[str, TeamRating]
    trained_rows: int


def tournament_weight(name: str) -> float:
    for key, weight in IMPORTANT_TOURNAMENTS.items():
        if key.lower() in str(name).lower():
            return weight
    return 1.0


def fit_model(history: pd.DataFrame, fixtures: pd.DataFrame) -> PredictorState:
    ensure_dirs()
    teams = sorted(set(fixtures["team1"]).union(fixtures["team2"]))
    teams = [t for t in teams if t and not _is_placeholder(t)]
    elo = {team: 1500.0 for team in teams}
    scored = {team: 0.0 for team in teams}
    conceded = {team: 0.0 for team in teams}
    weighted_matches = {team: 0.0 for team in teams}
    match_counts = {team: 0 for team in teams}
    recent_counts = {team: 0 for team in teams}

    cutoff = pd.Timestamp("2026-06-11")
    usable = history[history["date"] < cutoff].copy()
    if usable.empty:
        raise ValueError("No historical rows available before 2026-06-11")
    max_date = usable["date"].max()

    for row in usable.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        if home not in elo and away not in elo:
            continue
        for team in [home, away]:
            if team not in elo:
                elo[team] = 1500.0
                scored[team] = 0.0
                conceded[team] = 0.0
                weighted_matches[team] = 0.0
                match_counts[team] = 0
                recent_counts[team] = 0

        age_years = max((max_date - row.date).days / 365.25, 0.0)
        recency = math.exp(-age_years / 4.5)
        weight = recency * tournament_weight(row.tournament)
        hg, ag = int(row.home_score), int(row.away_score)
        result = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        expected = 1.0 / (1.0 + 10 ** ((elo[away] - elo[home]) / 400.0))
        goal_margin = abs(hg - ag)
        goal_mult = 1.0 if goal_margin <= 1 else 1.5 if goal_margin == 2 else (1.75 + (goal_margin - 3) / 8.0)
        k = 22.0 * tournament_weight(row.tournament) * goal_mult
        delta = k * (result - expected)
        elo[home] += delta
        elo[away] -= delta

        scored[home] += weight * hg
        conceded[home] += weight * ag
        scored[away] += weight * ag
        conceded[away] += weight * hg
        weighted_matches[home] += weight
        weighted_matches[away] += weight
        match_counts[home] += 1
        match_counts[away] += 1
        if age_years <= 2.0:
            recent_counts[home] += 1
            recent_counts[away] += 1

    global_goals = (usable["home_score"].sum() + usable["away_score"].sum()) / max(len(usable) * 2, 1)
    ratings: dict[str, TeamRating] = {}
    for team in teams:
        denom = max(weighted_matches.get(team, 0.0), 1.0)
        gf = scored.get(team, 0.0) / denom
        ga = conceded.get(team, 0.0) / denom
        attack = gf / max(global_goals, 0.1)
        defense = ga / max(global_goals, 0.1)
        ratings[team] = TeamRating(
            team=team,
            elo=round(float(elo.get(team, 1500.0)), 3),
            attack=round(float(np.clip(attack, 0.45, 1.9)), 4),
            defense=round(float(np.clip(defense, 0.45, 1.9)), 4),
            matches=int(match_counts.get(team, 0)),
            recent_matches=int(recent_counts.get(team, 0)),
        )

    state = PredictorState(params=_load_calibrated_params(), teams=ratings, trained_rows=int(len(usable)))
    save_model(state)
    try:
        from .ml import fit_ml_outcome_model

        fit_ml_outcome_model(usable, state)
    except Exception:
        pass
    return state


def _load_calibrated_params() -> PredictorParams:
    params = PredictorParams()
    path = MODELS_DIR / "calibration_adjustments.json"
    if not path.exists():
        return params
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return params
    multipliers = payload.get("multipliers", {})
    params.base_home_goals *= float(multipliers.get("base_home_goals", 1.0))
    params.base_away_goals *= float(multipliers.get("base_away_goals", 1.0))
    params.elo_goal_scale *= float(multipliers.get("elo_goal_scale", 1.0))
    params.host_goal_bonus += float(payload.get("host_goal_bonus_delta", 0.0))
    return params


def save_model(state: PredictorState, path: Path | None = None) -> None:
    ensure_dirs()
    path = path or MODELS_DIR / "model.json"
    payload = {
        "params": asdict(state.params),
        "teams": {team: asdict(rating) for team, rating in state.teams.items()},
        "trained_rows": state.trained_rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(path: Path | None = None) -> PredictorState:
    path = path or MODELS_DIR / "model.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = PredictorParams(**payload["params"])
    teams = {team: TeamRating(**rating) for team, rating in payload["teams"].items()}
    return PredictorState(params=params, teams=teams, trained_rows=int(payload["trained_rows"]))


def _is_placeholder(team: str) -> bool:
    return bool(team) and (
        team.startswith(("W", "L"))
        or bool(set(team).intersection({"/"}))
        or bool(pd.notna(team) and team[:1].isdigit())
    )


def expected_goals(state: PredictorState, team1: str, team2: str, ground: str = "") -> tuple[float, float]:
    p = state.params
    r1 = state.teams.get(team1, TeamRating(team1, 1500.0, 1.0, 1.0, 0, 0))
    r2 = state.teams.get(team2, TeamRating(team2, 1500.0, 1.0, 1.0, 0, 0))
    elo_component = (r1.elo - r2.elo) * p.elo_goal_scale
    host1 = p.host_goal_bonus if is_host_advantage(team1, ground) else 0.0
    host2 = p.host_goal_bonus if is_host_advantage(team2, ground) else 0.0
    mu1 = p.base_home_goals * math.exp(elo_component + p.attack_weight * math.log(r1.attack) - p.defense_weight * math.log(r2.defense)) + host1
    mu2 = p.base_away_goals * math.exp(-elo_component + p.attack_weight * math.log(r2.attack) - p.defense_weight * math.log(r1.defense)) + host2
    return float(np.clip(mu1, 0.15, 4.8)), float(np.clip(mu2, 0.15, 4.8))


def score_matrix(state: PredictorState, team1: str, team2: str, ground: str = "") -> np.ndarray:
    mu1, mu2 = expected_goals(state, team1, team2, ground)
    max_goals = state.params.max_goals
    probs1 = poisson.pmf(np.arange(max_goals + 1), mu1)
    probs2 = poisson.pmf(np.arange(max_goals + 1), mu2)
    matrix = np.outer(probs1, probs2)
    matrix = matrix / matrix.sum()
    return matrix


def predict_match(state: PredictorState, team1: str, team2: str, ground: str = "") -> dict:
    matrix = score_matrix(state, team1, team2, ground)
    max_goals = state.params.max_goals
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    poisson_probs = np.array([home_win, draw, away_win], dtype=float)
    ml_probs = None
    try:
        from .ml import predict_ml_probs

        ml_probs = predict_ml_probs(state, team1, team2, ground)
    except Exception:
        ml_probs = None
    if ml_probs is not None:
        weight = float(np.clip(state.params.ml_blend_weight, 0.0, 0.5))
        blended = (1.0 - weight) * poisson_probs + weight * ml_probs
        home_win, draw, away_win = [float(value) for value in blended / blended.sum()]
    mu1, mu2 = expected_goals(state, team1, team2, ground)
    flat = []
    for g1 in range(max_goals + 1):
        for g2 in range(max_goals + 1):
            flat.append((float(matrix[g1, g2]), g1, g2))
    top = sorted(flat, reverse=True)[:5]
    likely = top[0]
    return {
        "team1": team1,
        "team2": team2,
        "expected_goals1": round(mu1, 3),
        "expected_goals2": round(mu2, 3),
        "prob_team1_win": round(home_win, 6),
        "prob_draw": round(draw, 6),
        "prob_team2_win": round(away_win, 6),
        "model_blend": "poisson+ml" if ml_probs is not None else "poisson",
        "most_likely_score": f"{likely[1]}-{likely[2]}",
        "top_scores": [
            {"score": f"{g1}-{g2}", "probability": round(prob, 6)} for prob, g1, g2 in top
        ],
    }


def choose_winner(pred: dict) -> str:
    p1 = pred["prob_team1_win"] + pred["prob_draw"] * 0.5
    p2 = pred["prob_team2_win"] + pred["prob_draw"] * 0.5
    if abs(p1 - p2) < 1e-9:
        return pred["team1"] if pred["expected_goals1"] >= pred["expected_goals2"] else pred["team2"]
    return pred["team1"] if p1 >= p2 else pred["team2"]
