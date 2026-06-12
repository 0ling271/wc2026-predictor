from __future__ import annotations

import json

import pandas as pd
import typer

from .calibrate import calibration_report
from .data import load_fixtures, load_history, refresh_data
from .model import fit_model, load_model
from .monte_carlo import run_monte_carlo
from .paths import OUTPUTS_DIR, ensure_dirs
from .simulate import predict_tournament


app = typer.Typer(help="2026 World Cup predictor")


@app.command()
def refresh() -> None:
    """Download and process free public data sources."""
    statuses = refresh_data()
    for status in statuses:
        typer.echo(f"{status.name}: {status.status} ({status.rows})")


@app.command()
def train() -> None:
    """Train the transparent Elo + Poisson model."""
    ensure_dirs()
    fixtures = load_fixtures()
    history = load_history()
    state = fit_model(history, fixtures)
    typer.echo(f"trained_rows={state.trained_rows} teams={len(state.teams)}")


@app.command()
def predict(simulations: int = typer.Option(1000, help="Monte Carlo tournament simulations.")) -> None:
    """Predict all 104 matches and tournament path."""
    ensure_dirs()
    fixtures = load_fixtures()
    state = load_model()
    predictions, groups = predict_tournament(state, fixtures)
    odds = run_monte_carlo(state, fixtures, simulations=simulations)
    typer.echo(f"predictions={len(predictions)} groups={groups['group_letter'].nunique()}")
    typer.echo(f"monte_carlo_simulations={simulations} teams={len(odds)}")
    typer.echo(f"saved={OUTPUTS_DIR / 'predictions.csv'}")


@app.command()
def calibrate() -> None:
    """Compare completed matches, update parameters, then regenerate predictions."""
    report = calibration_report()
    fixtures = load_fixtures()
    state = load_model()
    predictions, _ = predict_tournament(state, fixtures)
    odds = run_monte_carlo(state, fixtures, simulations=1000)
    typer.echo(report)
    typer.echo(f"regenerated_predictions={len(predictions)}")
    typer.echo(f"regenerated_tournament_odds={len(odds)}")


@app.command()
def status() -> None:
    """Print local project status."""
    paths = {
        "fixtures": "data/processed/fixtures_2026.csv",
        "history": "data/processed/history.csv",
        "model": "models/model.json",
        "predictions": "outputs/predictions.json",
        "calibration": "reports/calibration_latest.md",
    }
    for name, path in paths.items():
        typer.echo(f"{name}: {'ok' if pd.io.common.file_exists(path) else 'missing'}")


@app.command()
def serve() -> None:
    """Show how to launch the local dashboard."""
    typer.echo("Run: streamlit run app.py")


@app.command()
def export_summary() -> None:
    """Write a compact summary JSON for automation jobs."""
    pred_path = OUTPUTS_DIR / "predictions.json"
    if not pred_path.exists():
        raise typer.BadParameter("Run predict first")
    df = pd.read_json(pred_path)
    summary = {
        "matches": int(len(df)),
        "completed": int((df["status"] == "actual").sum()),
        "champion_pick": json.loads((OUTPUTS_DIR / "bracket.json").read_text(encoding="utf-8")).get("champion"),
    }
    odds_path = OUTPUTS_DIR / "tournament_odds.json"
    if odds_path.exists():
        odds = pd.read_json(odds_path)
        if not odds.empty:
            summary["monte_carlo_champion_pick"] = str(odds.iloc[0]["team"])
            summary["monte_carlo_champion_probability"] = float(odds.iloc[0]["prob_champion"])
    (OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(summary)
