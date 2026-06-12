# WC2026 Predictor

Local, free-data-first predictor for all 104 FIFA World Cup 2026 matches.

## Quick Start

```powershell
python -m pip install -e .
python -m wc2026 refresh
python -m wc2026 train
python -m wc2026 predict
python -m wc2026 calibrate
streamlit run app.py
```

## Outputs

- `outputs/predictions.csv`
- `outputs/predictions.json`
- `outputs/group_tables.csv`
- `outputs/bracket.json`
- `reports/calibration_latest.md`

## Model

The model is intentionally transparent: historical international results train team Elo, attack, and defense ratings. Match score probabilities come from a Poisson score matrix with host advantage and recent strength baked into ratings. Knockout matches are resolved by deterministic bracket simulation from the predicted group tables.

## Data Sources

- OpenFootball 2026 World Cup JSON
- martj42 international results CSV
- FIFA scores and fixtures page
- World Football Elo page, cached for future enhancement
