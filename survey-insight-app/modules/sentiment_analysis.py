"""감성 분석."""

from __future__ import annotations

from typing import Any

import pandas as pd

BASIC_SENTIMENT = ("긍정", "중립", "부정")
DETAIL_SENTIMENT = (
    "만족", "불만", "혼란", "불안", "피로", "기대", "실망", "신뢰", "의심", "무관심"
)

SENTIMENT_RULES: dict[str, tuple[str, ...]] = {
    "만족": ("만족", "좋았", "편리", "괜찮"),
    "불만": ("불만", "불편", "짜증", "별로"),
    "혼란": ("모르겠", "헷갈", "어디서", "찾기 어렵"),
    "불안": ("불안", "걱정", "맞는지 모르", "확신"),
    "피로": ("많아", "피곤", "고를 게", "선택지"),
    "기대": ("있으면 좋", "바랐", "원해", "기대"),
    "실망": ("실망", "아쉽", "기대했"),
    "신뢰": ("믿", "신뢰", "안심"),
    "의심": ("의심", "불투명", "못 믿"),
    "무관심": ("잘 모르", "상관", "무관"),
}


def run_sentiment_analysis(
    df: pd.DataFrame,
    text_columns: list[str],
    excluded_ids: set[int] | None = None,
) -> pd.DataFrame:
    """주관식 감성 분류 및 요약 테이블."""
    excluded_ids = excluded_ids or set()
    detail_rows: list[dict[str, Any]] = []
    basic_counts = {"긍정": 0, "중립": 0, "부정": 0}

    for col in text_columns:
        if col not in df.columns:
            continue
        for _, row in df.iterrows():
            rid = int(row["response_id"]) if "response_id" in df.columns else _
            if rid in excluded_ids:
                continue
            text = str(row.get(col, "")).strip()
            if len(text) < 3:
                continue

            basic = _basic_sentiment(text)
            basic_counts[basic] += 1
            detail = _detail_sentiment(text)

            detail_rows.append(
                {
                    "response_id": rid,
                    "question": col,
                    "text": text[:200],
                    "basic_sentiment": basic,
                    "detail_sentiment": detail,
                    "improvement_link": _link_improvement(detail),
                }
            )

    summary = _aggregate_summary(detail_rows, basic_counts)
    summary_df = pd.DataFrame(summary)
    detail_df = pd.DataFrame(detail_rows)
    return summary_df if not summary_df.empty else detail_df


def _basic_sentiment(text: str) -> str:
    pos = sum(1 for k in ("좋", "만족", "편리", "추천", "유용") if k in text)
    neg = sum(1 for k in ("불편", "어렵", "불만", "복잡", "실망", "별로") if k in text)
    if pos > neg:
        return "긍정"
    if neg > pos:
        return "부정"
    return "중립"


def _detail_sentiment(text: str) -> str:
    for label, keywords in SENTIMENT_RULES.items():
        if any(k in text for k in keywords):
            return label
    basic = _basic_sentiment(text)
    if basic == "긍정":
        return "만족"
    if basic == "부정":
        return "불만"
    return "무관심"


def _link_improvement(detail: str) -> str:
    mapping = {
        "혼란": "온보딩/정보 구조 개선",
        "불안": "가격·정보 근거 제공",
        "피로": "선택 기준·옵션 단순화",
        "기대": "기능 로드맵 검토",
        "불만": "핵심 UX 마찰 지점 개선",
        "의심": "투명성·신뢰 요소 강화",
    }
    return mapping.get(detail, "사용자 경험 전반 점검")


def _aggregate_summary(detail_rows: list[dict], basic_counts: dict) -> list[dict]:
    if not detail_rows:
        return []

    by_detail: dict[str, list[dict]] = {}
    for row in detail_rows:
        d = row["detail_sentiment"]
        by_detail.setdefault(d, []).append(row)

    meanings = {
        "혼란": "사용 방법·정보 위치 이해 어려움",
        "불안": "신뢰 부족, 정보 부족",
        "피로": "과정이 길거나 선택지가 많음",
        "기대": "기능 추가·개선 희망",
        "불만": "현재 경험에 대한 부정적 평가",
    }

    summary = []
    for label, rows in sorted(by_detail.items(), key=lambda x: -len(x[1])):
        rep = rows[0]
        summary.append(
            {
                "감성": label,
                "의미": meanings.get(label, label),
                "건수": len(rows),
                "대표 응답": rep["text"][:150],
                "개선 연결점": rep["improvement_link"],
                "긍정": basic_counts.get("긍정", 0),
                "중립": basic_counts.get("중립", 0),
                "부정": basic_counts.get("부정", 0),
            }
        )
    return summary
