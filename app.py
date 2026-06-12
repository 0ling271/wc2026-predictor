from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
PRED_PATH = ROOT / "outputs" / "predictions.json"
GROUP_PATH = ROOT / "outputs" / "group_tables.csv"
BRACKET_PATH = ROOT / "outputs" / "bracket.json"
REPORT_PATH = ROOT / "reports" / "calibration_latest.md"
DETAIL_PATH = ROOT / "reports" / "calibration_details.csv"
ODDS_PATH = ROOT / "outputs" / "tournament_odds.csv"


def status_cn(value: str) -> str:
    return {"actual": "已赛", "predicted": "预测"}.get(str(value), str(value))


def result_cn(value: str) -> str:
    return "平局" if value == "Draw" else str(value)


def sort_by_kickoff(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    time_text = ordered.get("time", "").astype(str)
    hour_minute = time_text.str.extract(r"(\d{1,2}):(\d{2})").fillna("0").astype(int)
    ordered["_kickoff_sort"] = pd.to_datetime(ordered["date"], errors="coerce") + pd.to_timedelta(
        hour_minute[0] * 60 + hour_minute[1], unit="m"
    )
    return ordered.sort_values(["_kickoff_sort", "match_id"], na_position="last").drop(columns=["_kickoff_sort"])


def probability_label(row: pd.Series) -> str:
    return f"{row['team1']}胜 {row['prob_team1_win']:.1%} / 平 {row['prob_draw']:.1%} / {row['team2']}胜 {row['prob_team2_win']:.1%}"


def round_probability_chart(row: pd.Series) -> go.Figure:
    labels = [f"{row['team1']} 胜", "平局", f"{row['team2']} 胜"]
    values = [row["prob_team1_win"], row["prob_draw"], row["prob_team2_win"]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#1677ff", "#8c8c8c", "#d4380d"],
            text=[f"{v:.1%}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="胜平负概率",
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(values) * 1.25],
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    return fig


def score_chart(row: pd.Series) -> go.Figure:
    score_df = pd.DataFrame(row["top_scores"]).sort_values("probability", ascending=True)
    fig = px.bar(
        score_df,
        x="probability",
        y="score",
        orientation="h",
        text=score_df["probability"].map(lambda value: f"{value:.1%}"),
        labels={"probability": "概率", "score": "比分"},
        title="最可能比分 Top 5",
        color_discrete_sequence=["#13a8a8"],
    )
    fig.update_layout(
        xaxis_tickformat=".0%",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        showlegend=False,
    )
    return fig


def filtered_probability_overview(view: pd.DataFrame) -> go.Figure:
    chart_df = view.head(30).copy()
    chart_df["比赛"] = chart_df.apply(
        lambda r: f"{r['date']} {r.get('time', '')} | {r['team1']} vs {r['team2']}", axis=1
    )
    fig = go.Figure()
    fig.add_bar(name="球队1胜", x=chart_df["比赛"], y=chart_df["prob_team1_win"], marker_color="#1677ff")
    fig.add_bar(name="平局", x=chart_df["比赛"], y=chart_df["prob_draw"], marker_color="#8c8c8c")
    fig.add_bar(name="球队2胜", x=chart_df["比赛"], y=chart_df["prob_team2_win"], marker_color="#d4380d")
    fig.update_layout(
        title="按比赛日期排序的胜平负概率概览（最多显示前30场）",
        barmode="stack",
        yaxis_tickformat=".0%",
        xaxis_tickangle=-35,
        margin=dict(l=20, r=20, t=50, b=135),
        height=450,
        legend_title_text="结果",
    )
    return fig


st.set_page_config(page_title="2026世界杯预测", layout="wide")
st.title("2026世界杯全赛程预测")

if not PRED_PATH.exists():
    st.warning("请先运行：`python -m wc2026 refresh && python -m wc2026 train && python -m wc2026 predict`")
    st.stop()

pred = sort_by_kickoff(pd.read_json(PRED_PATH)).reset_index(drop=True)
bracket = json.loads(BRACKET_PATH.read_text(encoding="utf-8")) if BRACKET_PATH.exists() else {}

completed = int((pred["status"] == "actual").sum())
predicted = len(pred) - completed
champion = bracket.get("champion", "")
odds_df = pd.read_csv(ODDS_PATH) if ODDS_PATH.exists() else pd.DataFrame()
mc_champion = odds_df.iloc[0]["team"] if not odds_df.empty else ""

metric_cols = st.columns(4)
metric_cols[0].metric("总比赛数", len(pred))
metric_cols[1].metric("已赛", completed)
metric_cols[2].metric("待预测/未赛", predicted)
metric_cols[3].metric("当前冠军预测", champion)
if mc_champion:
    st.caption(f"蒙特卡洛夺冠概率最高：{mc_champion}（{odds_df.iloc[0]['prob_champion']:.1%}）")

with st.sidebar:
    st.header("筛选")
    rounds = ["全部"] + list(pred["round"].dropna().unique())
    selected_round = st.selectbox("阶段", rounds)
    groups = ["全部"] + sorted([g for g in pred["group"].dropna().unique() if g])
    selected_group = st.selectbox("小组", groups)
    selected_status = st.selectbox("状态", ["全部", "已赛", "预测"])
    team_query = st.text_input("球队搜索", "")

view = pred.copy()
if selected_round != "全部":
    view = view[view["round"] == selected_round]
if selected_group != "全部":
    view = view[view["group"] == selected_group]
if selected_status != "全部":
    reverse_status = {"已赛": "actual", "预测": "predicted"}
    view = view[view["status"] == reverse_status[selected_status]]
if team_query.strip():
    needle = team_query.strip().lower()
    view = view[
        view["team1"].str.lower().str.contains(needle, na=False)
        | view["team2"].str.lower().str.contains(needle, na=False)
    ]

view = sort_by_kickoff(view).reset_index(drop=True)
st.subheader("赛程预测总览")
if view.empty:
    st.info("当前筛选条件下没有比赛。")
    st.stop()

st.plotly_chart(filtered_probability_overview(view), use_container_width=True)

if not odds_df.empty:
    st.subheader("蒙特卡洛赛事推演")
    top_odds = odds_df.head(16).copy()
    fig = px.bar(
        top_odds.sort_values("prob_champion", ascending=True),
        x="prob_champion",
        y="team",
        orientation="h",
        text=top_odds.sort_values("prob_champion", ascending=True)["prob_champion"].map(lambda value: f"{value:.1%}"),
        labels={"prob_champion": "夺冠概率", "team": "球队"},
        title="夺冠概率 Top 16",
        color_discrete_sequence=["#722ed1"],
    )
    fig.update_layout(xaxis_tickformat=".0%", height=520, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        odds_df.rename(
            columns={
                "team": "球队",
                "prob_advance_r32": "晋级32强",
                "prob_advance_r16": "晋级16强",
                "prob_quarterfinal": "晋级8强",
                "prob_semifinal": "晋级4强",
                "prob_final": "晋级决赛",
                "prob_champion": "夺冠",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

display = view[
    [
        "match_id",
        "date",
        "time",
        "round",
        "group",
        "team1",
        "team2",
        "expected_goals1",
        "expected_goals2",
        "prob_team1_win",
        "prob_draw",
        "prob_team2_win",
        "most_likely_score",
        "predicted_winner",
        "actual_result",
        "actual_score",
        "status",
    ]
].copy()
display["status"] = display["status"].map(status_cn)
display["predicted_winner"] = display["predicted_winner"].map(result_cn)
display["actual_result"] = display["actual_result"].map(
    lambda value: "" if pd.isna(value) or value == "" else result_cn(value)
)
display = display.rename(
    columns={
        "match_id": "场次",
        "date": "日期",
        "time": "开球时间",
        "round": "阶段",
        "group": "小组",
        "team1": "球队1",
        "team2": "球队2",
        "expected_goals1": "球队1期望进球",
        "expected_goals2": "球队2期望进球",
        "prob_team1_win": "球队1胜率",
        "prob_draw": "平局概率",
        "prob_team2_win": "球队2胜率",
        "most_likely_score": "最可能比分",
        "predicted_winner": "预测结果",
        "actual_result": "实际结果",
        "actual_score": "实际比分",
        "status": "状态",
    }
)
st.dataframe(display, use_container_width=True, hide_index=True)

st.subheader("单场比赛可视化")
match_labels = view.apply(
    lambda r: f"{r['date']} {r.get('time', '')} | {r['team1']} vs {r['team2']}（{r['most_likely_score']}）",
    axis=1,
)
selected_label = st.selectbox("选择比赛", match_labels)
selected_index = int(match_labels[match_labels == selected_label].index[0])
selected = view.loc[selected_index]

detail_cols = st.columns(4)
detail_cols[0].metric("最可能比分", selected["most_likely_score"])
detail_cols[1].metric("预测结果", result_cn(selected["predicted_winner"]))
detail_cols[2].metric(f"{selected['team1']} 期望进球", f"{selected['expected_goals1']:.2f}")
detail_cols[3].metric(f"{selected['team2']} 期望进球", f"{selected['expected_goals2']:.2f}")
st.caption(probability_label(selected))

chart_cols = st.columns(2)
chart_cols[0].plotly_chart(round_probability_chart(selected), use_container_width=True)
chart_cols[1].plotly_chart(score_chart(selected), use_container_width=True)

if GROUP_PATH.exists():
    st.subheader("小组积分预测")
    group_table = pd.read_csv(GROUP_PATH)
    group_display = group_table.rename(
        columns={
            "team": "球队",
            "group": "小组",
            "points": "积分",
            "gf": "进球",
            "ga": "失球",
            "wins": "胜",
            "draws": "平",
            "losses": "负",
            "gd": "净胜球",
            "group_letter": "组别",
        }
    )
    st.dataframe(group_display, use_container_width=True, hide_index=True)

if REPORT_PATH.exists():
    st.subheader("模型校准报告")
    st.markdown(REPORT_PATH.read_text(encoding="utf-8"))

if DETAIL_PATH.exists():
    st.subheader("逐场误差明细")
    detail = pd.read_csv(DETAIL_PATH)
    detail_display = detail[
        [
            "match_id",
            "match",
            "predicted_score",
            "actual_score",
            "result_hit",
            "exact_score_hit",
            "predicted_expected_total_goals",
            "actual_total_goals",
            "score_mae",
            "expected_goal_mae",
        ]
    ].rename(
        columns={
            "match_id": "场次",
            "match": "比赛",
            "predicted_score": "预测比分",
            "actual_score": "实际比分",
            "result_hit": "胜平负命中",
            "exact_score_hit": "精确比分命中",
            "predicted_expected_total_goals": "预测期望总进球",
            "actual_total_goals": "实际总进球",
            "score_mae": "比分误差",
            "expected_goal_mae": "期望进球误差",
        }
    )
    detail_display["胜平负命中"] = detail_display["胜平负命中"].map({True: "是", False: "否"})
    detail_display["精确比分命中"] = detail_display["精确比分命中"].map({True: "是", False: "否"})
    st.dataframe(detail_display, use_container_width=True, hide_index=True)
