from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import is_host_advantage
from .model import PredictorState, TeamRating, tournament_weight
from .paths import MODELS_DIR, ensure_dirs


ML_MODEL_PATH = MODELS_DIR / "ml_outcome.joblib"
_ML_CACHE: dict | None = None


def _rating(state: PredictorState, team: str) -> TeamRating:
    return state.teams.get(team, TeamRating(team, 1500.0, 1.0, 1.0, 0, 0))


def match_features(state: PredictorState, team1: str, team2: str, neutral: bool = True, ground: str = "", tournament: str = "") -> list[float]:
    r1 = _rating(state, team1)
    r2 = _rating(state, team2)
    return [
        (r1.elo - r2.elo) / 400.0,
        r1.attack - r2.attack,
        r1.defense - r2.defense,
        np.log(max(r1.attack, 0.05)) - np.log(max(r2.attack, 0.05)),
        np.log(max(r1.defense, 0.05)) - np.log(max(r2.defense, 0.05)),
        (r1.recent_matches - r2.recent_matches) / 20.0,
        0.0 if neutral else 1.0,
        1.0 if is_host_advantage(team1, ground) else 0.0,
        1.0 if is_host_advantage(team2, ground) else 0.0,
        tournament_weight(tournament),
    ]


def fit_ml_outcome_model(history: pd.DataFrame, state: PredictorState) -> dict:
    ensure_dirs()
    teams = set(state.teams)
    df = history[history["home_team"].isin(teams) & history["away_team"].isin(teams)].copy()
    df = df[df["date"] >= pd.Timestamp("2014-01-01")]
    if len(df) < 300:
        return {"status": "skipped", "rows": int(len(df))}

    x_rows = []
    y_rows = []
    weights = []
    max_date = df["date"].max()
    for row in df.itertuples(index=False):
        x_rows.append(
            match_features(
                state,
                row.home_team,
                row.away_team,
                neutral=bool(row.neutral),
                tournament=row.tournament,
            )
        )
        if row.home_score > row.away_score:
            y_rows.append(0)
        elif row.home_score == row.away_score:
            y_rows.append(1)
        else:
            y_rows.append(2)
        age_years = max((max_date - row.date).days / 365.25, 0.0)
        weights.append(np.exp(-age_years / 5.0) * tournament_weight(row.tournament))

    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)
    sample_weight = np.asarray(weights, dtype=float)

    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    rf = RandomForestClassifier(
        n_estimators=240,
        min_samples_leaf=16,
        max_depth=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    logit.fit(x, y, logisticregression__sample_weight=sample_weight)
    rf.fit(x, y, sample_weight=sample_weight)
    payload = {"logit": logit, "rf": rf, "rows": int(len(df)), "classes": [0, 1, 2]}
    joblib.dump(payload, ML_MODEL_PATH)
    return {"status": "ok", "rows": int(len(df))}


def predict_ml_probs(state: PredictorState, team1: str, team2: str, ground: str = "", tournament: str = "FIFA World Cup") -> np.ndarray | None:
    if not ML_MODEL_PATH.exists():
        return None
    global _ML_CACHE
    if _ML_CACHE is None:
        _ML_CACHE = joblib.load(ML_MODEL_PATH)
    payload = _ML_CACHE
    features = np.asarray([match_features(state, team1, team2, neutral=True, ground=ground, tournament=tournament)], dtype=float)
    probs = []
    for model_name in ["logit", "rf"]:
        model = payload[model_name]
        model_probs = model.predict_proba(features)[0]
        class_probs = np.zeros(3, dtype=float)
        for idx, cls in enumerate(model.classes_):
            class_probs[int(cls)] = model_probs[idx]
        probs.append(class_probs)
    blended = np.mean(probs, axis=0)
    return blended / blended.sum()
