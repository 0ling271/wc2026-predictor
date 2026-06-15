from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import is_host_advantage
from .model import load_model, save_model
from .paths import MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, ensure_dirs


@dataclass
class CalibrationAdjustment:
    completed_matches: int
    home_goal_ratio: float
    away_goal_ratio: float
    home_goal_multiplier: float
    away_goal_multiplier: float
    elo_goal_scale_multiplier: float
    host_goal_bonus_delta: float
    score_total_shrink: float
    score_diff_shrink: float
    low_score_draw_boost: float
    one_goal_margin_boost: float
    note: str


def calibration_report() -> str:
    ensure_dirs()
    pred_path = OUTPUTS_DIR / "predictions.json"
    if not pred_path.exists():
        raise FileNotFoundError("Run predict first: missing outputs/predictions.json")
    df = pd.read_json(pred_path)
    completed = df[(df["status"] == "actual") & (df["actual_score"].fillna("") != "")].copy()
    lines = ["# 模型校准误差报告", ""]
    lines.append(f"已评估比赛数: {len(completed)}")
    if completed.empty:
        lines.append("")
        lines.append("暂无带实际比分的已赛比赛。")
        report = "\n".join(lines)
        (REPORTS_DIR / "calibration_latest.md").write_text(report, encoding="utf-8")
        return report

    details = _build_details(completed)
    details.to_csv(REPORTS_DIR / "calibration_details.csv", index=False, encoding="utf-8")

    brier = float(details["brier_score"].mean())
    log_loss = float(details["log_loss"].mean())
    score_mae = float(details["score_mae"].mean())
    exact_hit_rate = float(details["exact_score_hit"].mean())
    result_hit_rate = float(details["result_hit"].mean())
    expected_total = float(details["predicted_expected_total_goals"].sum())
    actual_total = float(details["actual_total_goals"].sum())
    total_ratio = actual_total / max(expected_total, 1e-9)

    adjustment = _write_adjustments(details)
    _apply_adjustment_to_current_model(adjustment)

    lines += [
        "",
        f"胜平负命中率: {result_hit_rate:.1%}",
        f"精确比分命中率: {exact_hit_rate:.1%}",
        f"Brier 分数: {brier:.4f}",
        f"对数损失: {log_loss:.4f}",
        f"比分平均绝对误差: {score_mae:.4f}",
        f"预测总进球/实际总进球: {expected_total:.2f} / {actual_total:.2f}",
        f"总进球校准比率: {total_ratio:.3f}",
        "",
        "## 本轮参数微调",
        "",
        f"- 球队1基础进球倍率: {adjustment.home_goal_multiplier:.4f}",
        f"- 球队2基础进球倍率: {adjustment.away_goal_multiplier:.4f}",
        f"- Elo 进球影响倍率: {adjustment.elo_goal_scale_multiplier:.4f}",
        f"- 主场进球加成调整: {adjustment.host_goal_bonus_delta:+.4f}",
        f"- 冷门/平局保护: diff_shrink={adjustment.score_diff_shrink:.3f}, draw_boost={adjustment.low_score_draw_boost:.3f}",
        f"- 说明: {adjustment.note}",
        "",
        "## 逐场误差",
        "",
        "| 场次 | 比赛 | 预测比分 | 实际比分 | 胜平负是否命中 | 精确比分是否命中 | 期望总进球 | 实际总进球 |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in details.itertuples(index=False):
        lines.append(
            f"| {row.match_id} | {row.match} | {row.predicted_score} | {row.actual_score} | "
            f"{'是' if row.result_hit else '否'} | {'是' if row.exact_score_hit else '否'} | "
            f"{row.predicted_expected_total_goals:.2f} | {row.actual_total_goals:.0f} |"
        )

    report = "\n".join(lines)
    (REPORTS_DIR / "calibration_latest.md").write_text(report, encoding="utf-8")
    return report


def _build_details(completed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in completed.itertuples(index=False):
        actual_g1, actual_g2 = [int(x) for x in str(row.actual_score).split("-")]
        pred_g1, pred_g2 = [int(x) for x in str(row.most_likely_score).split("-")]
        probs = np.array([row.prob_team1_win, row.prob_draw, row.prob_team2_win], dtype=float)
        probs = np.clip(probs / probs.sum(), 1e-9, 1.0)
        actual_vec = np.array([actual_g1 > actual_g2, actual_g1 == actual_g2, actual_g2 > actual_g1], dtype=float)
        actual_result_idx = int(actual_vec.argmax())
        actual_result = [row.team1, "Draw", row.team2][actual_result_idx]
        predicted_result = row.predicted_winner
        rows.append(
            {
                "match_id": int(row.match_id),
                "match": f"{row.team1} vs {row.team2}",
                "ground": row.ground,
                "team1": row.team1,
                "team2": row.team2,
                "predicted_score": row.most_likely_score,
                "actual_score": row.actual_score,
                "predicted_result": predicted_result,
                "actual_result": actual_result,
                "result_hit": predicted_result == actual_result,
                "exact_score_hit": pred_g1 == actual_g1 and pred_g2 == actual_g2,
                "expected_goals1": float(row.expected_goals1),
                "expected_goals2": float(row.expected_goals2),
                "actual_goals1": actual_g1,
                "actual_goals2": actual_g2,
                "predicted_expected_total_goals": float(row.expected_goals1 + row.expected_goals2),
                "actual_total_goals": actual_g1 + actual_g2,
                "score_mae": (abs(pred_g1 - actual_g1) + abs(pred_g2 - actual_g2)) / 2.0,
                "expected_goal_mae": (abs(row.expected_goals1 - actual_g1) + abs(row.expected_goals2 - actual_g2)) / 2.0,
                "brier_score": float(np.mean((probs - actual_vec) ** 2)),
                "log_loss": float(-np.log(probs[actual_result_idx])),
                "predicted_draw_probability": float(probs[1]),
                "favorite_miss": bool(probs.max() >= 0.45 and predicted_result != actual_result),
                "team1_is_host": is_host_advantage(row.team1, row.ground),
                "team2_is_host": is_host_advantage(row.team2, row.ground),
            }
        )
    return pd.DataFrame(rows)


def _write_adjustments(details: pd.DataFrame) -> CalibrationAdjustment:
    n = int(len(details))
    shrink = n / (n + 18.0)
    home_ratio = float(details["actual_goals1"].sum() / max(details["expected_goals1"].sum(), 1e-9))
    away_ratio = float(details["actual_goals2"].sum() / max(details["expected_goals2"].sum(), 1e-9))
    home_multiplier = float(np.clip(1.0 + shrink * (home_ratio - 1.0), 0.94, 1.06))
    away_multiplier = float(np.clip(1.0 + shrink * (away_ratio - 1.0), 0.94, 1.06))

    result_hit_rate = float(details["result_hit"].mean())
    elo_multiplier = 1.0 if result_hit_rate >= 0.5 else 0.99
    draw_rate = float((details["actual_result"] == "Draw").mean())
    predicted_draw_rate = float(details["predicted_draw_probability"].mean())
    favorite_miss_rate = float(details["favorite_miss"].mean())
    total_ratio = float(details["actual_total_goals"].sum() / max(details["predicted_expected_total_goals"].sum(), 1e-9))
    score_total_shrink = float(np.clip(0.68 * (1.0 + shrink * (total_ratio - 1.0) * 0.45), 0.58, 0.78))
    draw_gap = max(draw_rate - predicted_draw_rate, 0.0)
    low_score_draw_boost = float(np.clip(1.08 + shrink * draw_gap * 0.75, 1.04, 1.22))
    favorite_gap = max(favorite_miss_rate - 0.28, 0.0)
    score_diff_shrink = float(np.clip(0.68 * (1.0 - shrink * favorite_gap * 0.45), 0.58, 0.72))
    one_goal_margin_boost = float(np.clip(1.04 + shrink * favorite_gap * 0.18, 1.02, 1.12))

    host_rows = details[details["team1_is_host"] | details["team2_is_host"]]
    host_delta = 0.0
    if not host_rows.empty:
        host_errors = []
        for row in host_rows.itertuples(index=False):
            if row.team1_is_host:
                host_errors.append(float(row.actual_goals1 - row.expected_goals1))
            if row.team2_is_host:
                host_errors.append(float(row.actual_goals2 - row.expected_goals2))
        host_delta = float(np.clip(shrink * np.mean(host_errors) * 0.03, -0.025, 0.025))

    adjustment = CalibrationAdjustment(
        completed_matches=n,
        home_goal_ratio=home_ratio,
        away_goal_ratio=away_ratio,
        home_goal_multiplier=home_multiplier,
        away_goal_multiplier=away_multiplier,
        elo_goal_scale_multiplier=elo_multiplier,
        host_goal_bonus_delta=host_delta,
        score_total_shrink=score_total_shrink,
        score_diff_shrink=score_diff_shrink,
        low_score_draw_boost=low_score_draw_boost,
        one_goal_margin_boost=one_goal_margin_boost,
        note="样本较少时使用强收缩，避免两三场比赛导致参数剧烈摆动。",
    )
    payload = {
        "completed_matches": adjustment.completed_matches,
        "ratios": {
            "home_goal_ratio": adjustment.home_goal_ratio,
            "away_goal_ratio": adjustment.away_goal_ratio,
        },
        "multipliers": {
            "base_home_goals": adjustment.home_goal_multiplier,
            "base_away_goals": adjustment.away_goal_multiplier,
            "elo_goal_scale": adjustment.elo_goal_scale_multiplier,
        },
        "host_goal_bonus_delta": adjustment.host_goal_bonus_delta,
        "score": {
            "score_total_shrink": adjustment.score_total_shrink,
            "score_diff_shrink": adjustment.score_diff_shrink,
            "low_score_draw_boost": adjustment.low_score_draw_boost,
            "one_goal_margin_boost": adjustment.one_goal_margin_boost,
        },
        "diagnostics": {
            "draw_rate": draw_rate,
            "predicted_draw_rate": predicted_draw_rate,
            "favorite_miss_rate": favorite_miss_rate,
            "total_goal_ratio": total_ratio,
        },
        "note": adjustment.note,
    }
    (MODELS_DIR / "calibration_adjustments.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return adjustment


def _apply_adjustment_to_current_model(adjustment: CalibrationAdjustment) -> None:
    model_path = MODELS_DIR / "model.json"
    if not model_path.exists():
        return
    state = load_model(model_path)
    state.params.base_home_goals *= adjustment.home_goal_multiplier
    state.params.base_away_goals *= adjustment.away_goal_multiplier
    state.params.elo_goal_scale *= adjustment.elo_goal_scale_multiplier
    state.params.host_goal_bonus += adjustment.host_goal_bonus_delta
    state.params.score_total_shrink = adjustment.score_total_shrink
    state.params.score_diff_shrink = adjustment.score_diff_shrink
    state.params.low_score_draw_boost = adjustment.low_score_draw_boost
    state.params.one_goal_margin_boost = adjustment.one_goal_margin_boost
    save_model(state, model_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload["last_calibration"] = asdict(adjustment)
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
