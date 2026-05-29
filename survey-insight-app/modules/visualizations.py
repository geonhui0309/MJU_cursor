"""분석 결과 시각화 (차트·워드클라우드)."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    from matplotlib import font_manager

    HAS_FM = True
except ImportError:
    HAS_FM = False

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

MODULE_DIR = Path(__file__).resolve().parent
APP_DIR = MODULE_DIR.parent
FONT_DIR = APP_DIR / "assets" / "fonts"
FONT_FILE = FONT_DIR / "NanumGothic.ttf"
FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
)

CHART_COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#64748b"]

_KOREAN_FONT_NAME: str | None = None


def ensure_korean_font() -> str | None:
    """한글 폰트 경로 확보 (로컬·Cloud)."""
    global _KOREAN_FONT_NAME
    if _KOREAN_FONT_NAME:
        return str(FONT_FILE) if FONT_FILE.exists() else None

    candidates = [
        FONT_FILE,
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/Library/Fonts/AppleGothic.ttf"),
        Path.home() / "Library/Fonts/NanumGothic.ttf",
    ]

    for path in candidates:
        if path.exists():
            _apply_font(path)
            return str(path)

    try:
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        if not FONT_FILE.exists():
            urllib.request.urlretrieve(FONT_URL, FONT_FILE)
        if FONT_FILE.exists():
            _apply_font(FONT_FILE)
            return str(FONT_FILE)
    except Exception:
        pass

    plt.rcParams["axes.unicode_minus"] = False
    return None


def _apply_font(path: Path) -> None:
    global _KOREAN_FONT_NAME
    if HAS_FM:
        try:
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            _KOREAN_FONT_NAME = prop.get_name()
            plt.rcParams["font.family"] = _KOREAN_FONT_NAME
        except Exception:
            plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def numbered_labels(labels: list[str], max_len: int = 80) -> tuple[list[str], list[str]]:
    """차트용 짧은 번호 라벨 + 하단 범례 문장."""
    legend: list[str] = []
    short: list[str] = []
    for i, raw in enumerate(labels, start=1):
        text = str(raw).strip() or f"항목 {i}"
        if len(text) > max_len:
            text = text[: max_len - 1] + "…"
        short.append(str(i))
        legend.append(f"{i}. {text}")
    return short, legend


def _style_axes(ax, title: str = "") -> None:
    fig = ax.figure
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    if title:
        ax.set_title(title, fontsize=10, color="#e2e8f0", pad=8)


def likert_summary_df(quant_results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for col, res in quant_results.items():
        if col.startswith("_") or not isinstance(res, dict):
            continue
        if res.get("type") != "리커트 척도형":
            continue
        rows.append(
            {
                "문항": str(col),
                "평균": res.get("mean", 0),
                "긍정%": res.get("positive_pct", 0),
                "부정%": res.get("negative_pct", 0),
            }
        )
    return pd.DataFrame(rows)


def choice_distribution_df(quant_results: dict[str, Any], max_questions: int = 3) -> pd.DataFrame | None:
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
        for d in res["distribution"][:5]:
            rows.append(
                {
                    "문항": str(col),
                    "선택지": str(d.get("choice", "")),
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


def numbered_index_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """인덱스를 1,2,3… 으로 바꾸고 범례 문장 반환."""
    if df is None or df.empty:
        return df, []
    labels = [str(x) for x in df.index.tolist()]
    short, legend = numbered_labels(labels)
    out = df.copy()
    out.index = short
    return out, legend


def sentiment_counts_df(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    if sentiment_df is None or sentiment_df.empty:
        return pd.DataFrame()
    if "basic_sentiment" in sentiment_df.columns:
        s = sentiment_df["basic_sentiment"].value_counts()
        return pd.DataFrame({"건수": s.values}, index=[str(x) for x in s.index])
    if "감성" in sentiment_df.columns and "건수" in sentiment_df.columns:
        return sentiment_df.groupby("감성")["건수"].sum().to_frame()
    if "감성" in sentiment_df.columns:
        s = sentiment_df["감성"].value_counts()
        return pd.DataFrame({"건수": s.values}, index=[str(x) for x in s.index])
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
    return pd.DataFrame({"건수": s.values}, index=[str(x) for x in s.index])


def cleaning_counts_series(cleaning_summary: dict) -> pd.Series:
    labels = cleaning_summary.get("label_counts") or {}
    if labels:
        return pd.Series({str(k): v for k, v in labels.items()})
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
    return pd.DataFrame({"건수": s.values}, index=[str(x) for x in s.index])


def make_wordcloud_figure(keyword_df: pd.DataFrame) -> plt.Figure | None:
    if not WORDCLOUD_AVAILABLE or keyword_df is None or keyword_df.empty:
        return None
    col_kw = "키워드" if "키워드" in keyword_df.columns else keyword_df.columns[0]
    col_freq = "빈도" if "빈도" in keyword_df.columns else keyword_df.columns[1]
    freq = dict(zip(keyword_df[col_kw].astype(str), keyword_df[col_freq].astype(int)))
    if not freq:
        return None

    font_path = ensure_korean_font()
    wc_kwargs: dict[str, Any] = dict(
        width=900,
        height=420,
        background_color="#0f172a",
        colormap="Blues",
        max_words=60,
        prefer_horizontal=0.85,
        margin=8,
    )
    if font_path:
        wc_kwargs["font_path"] = font_path

    wc = WordCloud(**wc_kwargs).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("#0f172a")
    plt.tight_layout(pad=0.2)
    return fig


def make_likert_horizontal_figure(likert_df: pd.DataFrame) -> tuple[plt.Figure | None, list[str]]:
    """리커트 막대 차트 + 번호 범례."""
    if likert_df is None or likert_df.empty:
        return None, []
    ensure_korean_font()

    labels = likert_df["문항"].astype(str).tolist()
    short, legend = numbered_labels(labels)

    fig, ax = plt.subplots(figsize=(10, max(2.2, len(likert_df) * 0.45)))
    y = range(len(likert_df))
    ax.barh(y, likert_df["평균"], color=CHART_COLORS[0], height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Average (1-5)", fontsize=9)
    _style_axes(ax, "Likert mean")
    ax.axvline(3, color="#64748b", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    return fig, legend


def make_pie_figure(series: pd.Series, title: str = "") -> tuple[plt.Figure | None, list[str]]:
    """파이 차트 — 슬라이스는 번호만, 범례는 호출측에서 표시."""
    if series is None or series.empty or series.sum() == 0:
        return None, []

    ensure_korean_font()
    labels = [str(x) for x in series.index.tolist()]
    short, legend = numbered_labels(labels)
    values = series.values

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    _, _, autotexts = ax.pie(
        values,
        labels=short,
        autopct="%1.0f%%",
        startangle=90,
        colors=CHART_COLORS[: len(values)],
        textprops={"color": "#e2e8f0", "fontsize": 9},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(8)
    _style_axes(ax, title or "Distribution")
    plt.tight_layout()
    return fig, legend


def make_grouped_choice_figure(choice_df: pd.DataFrame) -> tuple[plt.Figure | None, list[str]]:
    """선택지 비율 막대 — 문항·선택지 모두 번호 범례."""
    if choice_df is None or choice_df.empty:
        return None, []

    ensure_korean_font()
    row_labels = [
        f"{row['문항']} / {row['선택지']}" for _, row in choice_df.iterrows()
    ]
    short, legend = numbered_labels(row_labels, max_len=100)

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(choice_df) * 0.35)))
    y = range(len(choice_df))
    ax.barh(y, choice_df["비율"], color=CHART_COLORS[2], height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Ratio (%)", fontsize=9)
    _style_axes(ax, "Choice distribution")
    plt.tight_layout()
    return fig, legend
