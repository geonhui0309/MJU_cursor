"""분석 결과 시각화 (차트·워드클라우드)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Streamlit / Cloud 호환
plt.rcParams["font.family"] = [
    "AppleGothic",
    "Malgun Gothic",
    "NanumGothic",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

CHART_COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#64748b"]


def likert_summary_df(quant_results: dict[str, Any]) -> pd.DataFrame:
    """리커트 문항별 평균·긍정/부정 비율."""
    rows = []
    for col, res in quant_results.items():
        if col.startswith("_") or not isinstance(res, dict):
            continue
        if res.get("type") != "리커트 척도형":
            continue
        label = col if len(col) <= 28 else col[:26] + "…"
        rows.append(
            {
                "문항": label,
                "평균": res.get("mean", 0),
                "긍정%": res.get("positive_pct", 0),
                "부정%": res.get("negative_pct", 0),
            }
        )
    return pd.DataFrame(rows)


def choice_distribution_df(quant_results: dict[str, Any], max_questions: int = 3) -> pd.DataFrame | None:
    """단일/복수 선택 상위 선택지 (long format for bar chart)."""
    rows = []
    count = 0
    for col, res in quant_results.items():
        if col.startswith("_") or count >= max_questions:
            break
        if not isinstance(res, dict) or "distribution" not in res:
            continue
        qtype = res.get("type", "")
        if qtype not in ("단일 선택형", "복수 선택형"):
            continue
        q_label = col if len(col) <= 20 else col[:18] + "…"
        for d in res["distribution"][:5]:
            rows.append(
                {
                    "문항": q_label,
                    "선택지": str(d.get("choice", ""))[:24],
                    "비율": d.get("ratio", 0),
                }
            )
        count += 1
    return pd.DataFrame(rows) if rows else None


def keyword_chart_df(keyword_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if keyword_df is None or keyword_df.empty:
        return pd.DataFrame()
    col_kw = "키워드" if "키워드" in keyword_df.columns else keyword_df.columns[0]
    col_freq = "빈도" if "빈도" in keyword_df.columns else keyword_df.columns[1]
    df = keyword_df.nlargest(top_n, col_freq)[[col_kw, col_freq]].copy()
    df.columns = ["키워드", "빈도"]
    return df.set_index("키워드")


def sentiment_counts_df(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    if sentiment_df is None or sentiment_df.empty:
        return pd.DataFrame()
    if "basic_sentiment" in sentiment_df.columns:
        s = sentiment_df["basic_sentiment"].value_counts()
        return pd.DataFrame({"감성": s.index, "건수": s.values}).set_index("감성")
    if "감성" in sentiment_df.columns and "건수" in sentiment_df.columns:
        return sentiment_df.groupby("감성")["건수"].sum().to_frame()
    if "감성" in sentiment_df.columns:
        s = sentiment_df["감성"].value_counts()
        return pd.DataFrame({"건수": s.values}, index=s.index)
    return pd.DataFrame()


def journey_chart_df(journey_df: pd.DataFrame) -> pd.DataFrame:
    if journey_df is None or journey_df.empty:
        return pd.DataFrame()
    col_stage = "여정 단계" if "여정 단계" in journey_df.columns else journey_df.columns[0]
    col_cnt = "응답 수" if "응답 수" in journey_df.columns else journey_df.columns[1]
    df = journey_df[[col_stage, col_cnt]].copy()
    df.columns = ["단계", "응답 수"]
    return df.set_index("단계")


def text_structure_counts_df(text_structure_df: pd.DataFrame) -> pd.DataFrame:
    if text_structure_df is None or text_structure_df.empty:
        return pd.DataFrame()
    col = "text_structure" if "text_structure" in text_structure_df.columns else "텍스트 구조"
    if col not in text_structure_df.columns:
        return pd.DataFrame()
    s = text_structure_df[col].value_counts()
    return pd.DataFrame({"유형": s.index, "건수": s.values}).set_index("유형")


def cleaning_counts_series(cleaning_summary: dict) -> pd.Series:
    labels = cleaning_summary.get("label_counts") or {}
    if labels:
        return pd.Series(labels)
    return pd.Series(
        {
            "분석 포함": cleaning_summary.get("included_count", 0),
            "분석 제외": cleaning_summary.get("excluded_count", 0),
            "검토 필요": cleaning_summary.get("review_count", 0),
        }
    )


def hypothesis_verdict_df(hypothesis_df: pd.DataFrame) -> pd.DataFrame:
    if hypothesis_df is None or hypothesis_df.empty or "지지 여부" not in hypothesis_df.columns:
        return pd.DataFrame()
    s = hypothesis_df["지지 여부"].value_counts()
    return pd.DataFrame({"판정": s.index, "건수": s.values}).set_index("판정")


def action_priority_df(actions: list[dict]) -> pd.DataFrame:
    if not actions:
        return pd.DataFrame()
    df = pd.DataFrame(actions)
    if "우선순위" not in df.columns:
        return pd.DataFrame()
    order = {"높음": 0, "중간": 1, "낮음": 2}
    df["_ord"] = df["우선순위"].map(order).fillna(3)
    df = df.sort_values("_ord")
    labels = df["개선안"].astype(str).str.slice(0, 36)
    return pd.DataFrame({"건수": [1] * len(df)}, index=labels)


def make_wordcloud_figure(keyword_df: pd.DataFrame) -> plt.Figure | None:
    if not WORDCLOUD_AVAILABLE or keyword_df is None or keyword_df.empty:
        return None
    col_kw = "키워드" if "키워드" in keyword_df.columns else keyword_df.columns[0]
    col_freq = "빈도" if "빈도" in keyword_df.columns else keyword_df.columns[1]
    freq = dict(zip(keyword_df[col_kw].astype(str), keyword_df[col_freq].astype(int)))
    if not freq:
        return None

    wc = WordCloud(
        width=900,
        height=420,
        background_color="#0f172a",
        colormap="Blues",
        max_words=60,
        prefer_horizontal=0.85,
        margin=8,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("#0f172a")
    plt.tight_layout(pad=0.2)
    return fig


def make_likert_horizontal_figure(likert_df: pd.DataFrame) -> plt.Figure | None:
    if likert_df is None or likert_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(2.5, len(likert_df) * 0.55)))
    y = range(len(likert_df))
    ax.barh(y, likert_df["평균"], color=CHART_COLORS[0], height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(likert_df["문항"], fontsize=9)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("평균 점수 (1~5)")
    ax.set_title("리커트 척도 문항 평균", fontsize=11, color="#e2e8f0")
    ax.axvline(3, color="#64748b", linestyle="--", linewidth=0.8)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    plt.tight_layout()
    return fig


def make_pie_figure(series: pd.Series, title: str) -> plt.Figure | None:
    if series is None or series.empty or series.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.pie(
        series.values,
        labels=series.index,
        autopct="%1.0f%%",
        startangle=90,
        colors=CHART_COLORS[: len(series)],
        textprops={"color": "#e2e8f0", "fontsize": 8},
    )
    ax.set_title(title, fontsize=10, color="#e2e8f0")
    fig.patch.set_facecolor("#0f172a")
    return fig
