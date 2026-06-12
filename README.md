# WC2026 Predictor

Local, free-data-first predictor for all 104 FIFA World Cup 2026 matches.

## Quick Start

```powershell
python -m pip install -e .
python -m wc2026 refresh
python -m wc2026 train
python -m wc2026 predict --simulations 1000
python -m wc2026 calibrate
streamlit run app.py
```

## Outputs

- `outputs/predictions.csv`
- `outputs/predictions.json`
- `outputs/group_tables.csv`
- `outputs/bracket.json`
- `outputs/tournament_odds.csv`
- `reports/calibration_latest.md`

## Model

The model is intentionally transparent but now uses a mainstream hybrid workflow:

- Elo-style ratings quantify team strength.
- A Poisson score matrix estimates exact-score probabilities.
- Logistic regression and random forest models calibrate 1X2 outcome probabilities from rating and team-strength features.
- Monte Carlo simulation repeatedly plays the full 104-match tournament to estimate advancement, final, and champion probabilities. The efficient default is 1,000 runs; increase `--simulations` for slower but smoother odds.
- Post-match calibration writes persistent parameter adjustments after comparing predictions with actual results.

## Data Sources

- OpenFootball 2026 World Cup JSON
- martj42 international results CSV
- FIFA scores and fixtures page
- World Football Elo page, cached for future enhancement
