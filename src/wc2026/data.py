from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .paths import PROCESSED_DIR, RAW_DIR, ensure_dirs


OPENFOOTBALL_2026_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FIFA_FIXTURES_URL = "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures"
ELO_URL = "https://www.eloratings.net/2026_World_Cup"

HOSTS = {"Mexico", "USA", "Canada"}
HOST_GROUNDS = {
    "Mexico City": "Mexico",
    "Guadalajara": "Mexico",
    "Monterrey": "Mexico",
    "Toronto": "Canada",
    "Vancouver": "Canada",
    "Los Angeles": "USA",
    "San Francisco Bay Area": "USA",
    "Seattle": "USA",
    "Atlanta": "USA",
    "Boston": "USA",
    "Dallas": "USA",
    "Houston": "USA",
    "Kansas City": "USA",
    "Miami": "USA",
    "New York/New Jersey": "USA",
    "Philadelphia": "USA",
}

ALIASES = {
    "Korea Republic": "South Korea",
    "Korea Rep.": "South Korea",
    "United States": "USA",
    "United States of America": "USA",
    "Czechia": "Czech Republic",
    "Cote d'Ivoire": "Ivory Coast",
    "Cote dIvoire": "Ivory Coast",
    "Cote d Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Bosnia-Herzegovina": "Bosnia & Herzegovina",
}


@dataclass(frozen=True)
class SourceStatus:
    name: str
    path: str
    rows: int
    status: str


def normalize_team(name: str | None) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"\s+", " ", str(name).strip())
    return ALIASES.get(cleaned, cleaned)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _get(url: str) -> requests.Response:
    response = requests.get(url, timeout=30, headers={"User-Agent": "wc2026-predictor/0.1"})
    response.raise_for_status()
    return response


def refresh_data() -> list[SourceStatus]:
    ensure_dirs()
    statuses: list[SourceStatus] = []

    wc_path = RAW_DIR / "openfootball_worldcup_2026.json"
    try:
        payload = _get(OPENFOOTBALL_2026_URL).json()
        wc_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        statuses.append(SourceStatus("openfootball_2026", str(wc_path), len(payload.get("matches", [])), "ok"))
    except Exception as exc:
        statuses.append(SourceStatus("openfootball_2026", str(wc_path), 0, f"failed: {exc}"))

    hist_path = RAW_DIR / "international_results.csv"
    try:
        content = _get(RESULTS_URL).text
        hist_path.write_text(content, encoding="utf-8")
        rows = max(content.count("\n") - 1, 0)
        statuses.append(SourceStatus("international_results", str(hist_path), rows, "ok"))
    except Exception as exc:
        statuses.append(SourceStatus("international_results", str(hist_path), 0, f"failed: {exc}"))

    fifa_path = RAW_DIR / "fifa_scores_fixtures.html"
    try:
        html = _get(FIFA_FIXTURES_URL).text
        fifa_path.write_text(html, encoding="utf-8")
        statuses.append(SourceStatus("fifa_scores_fixtures", str(fifa_path), 1, "ok"))
    except Exception as exc:
        statuses.append(SourceStatus("fifa_scores_fixtures", str(fifa_path), 0, f"failed: {exc}"))

    elo_path = RAW_DIR / "world_football_elo_2026.html"
    try:
        html = _get(ELO_URL).text
        elo_path.write_text(html, encoding="utf-8")
        statuses.append(SourceStatus("world_football_elo", str(elo_path), 1, "ok"))
    except Exception as exc:
        statuses.append(SourceStatus("world_football_elo", str(elo_path), 0, f"failed: {exc}"))

    fixtures = load_fixtures()
    fixtures.to_csv(PROCESSED_DIR / "fixtures_2026.csv", index=False)
    history = load_history()
    history.to_csv(PROCESSED_DIR / "history.csv", index=False)
    pd.DataFrame([s.__dict__ for s in statuses]).to_csv(PROCESSED_DIR / "source_status.csv", index=False)
    return statuses


def load_fixtures() -> pd.DataFrame:
    path = RAW_DIR / "openfootball_worldcup_2026.json"
    if not path.exists():
        raise FileNotFoundError("Run refresh first: missing data/raw/openfootball_worldcup_2026.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for idx, match in enumerate(data.get("matches", []), start=1):
        score = match.get("score") or {}
        ft = score.get("ft") or [None, None]
        rows.append(
            {
                "match_id": int(match.get("num") or idx),
                "round": match.get("round", ""),
                "date": match.get("date", ""),
                "time": match.get("time", ""),
                "team1": normalize_team(match.get("team1")),
                "team2": normalize_team(match.get("team2")),
                "group": match.get("group", ""),
                "ground": match.get("ground", ""),
                "actual_goals1": ft[0] if len(ft) > 0 else None,
                "actual_goals2": ft[1] if len(ft) > 1 else None,
            }
        )
    return pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)


def load_history() -> pd.DataFrame:
    path = RAW_DIR / "international_results.csv"
    if not path.exists():
        raise FileNotFoundError("Run refresh first: missing data/raw/international_results.csv")
    df = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")
    df = df[list(required)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def parse_fifa_fixture_text() -> str:
    path = RAW_DIR / "fifa_scores_fixtures.html"
    if not path.exists():
        return ""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    return soup.get_text(" ", strip=True)


def is_host_advantage(team: str, ground: str) -> bool:
    if team not in HOSTS:
        return False
    for prefix, country in HOST_GROUNDS.items():
        if str(ground).startswith(prefix):
            return country == team
    return False
