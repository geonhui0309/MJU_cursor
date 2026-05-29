"""사용자 여정 단계 매핑."""

from __future__ import annotations

from typing import Any

import pandas as pd

from modules.keyword_analysis import JOURNEY_KW

DEFAULT_STAGES = [
    "인지", "진입", "탐색", "선택", "실행", "확인", "재사용", "이탈",
]


def run_journey_mapping(
    df: pd.DataFrame,
    text_columns: list[str],
    custom_stages: list[str] | None = None,
    keyword_df: pd.DataFrame | None = None,
    sentiment_df: pd.DataFrame | None = None,
    excluded_ids: set[int] | None = None,
) -> pd.DataFrame:
    """응답을 여정 단계에 매핑하고 단계별 요약."""
    stages = custom_stages if custom_stages else DEFAULT_STAGES
    excluded_ids = excluded_ids or set()

    stage_entries: dict[str, list[dict]] = {s: [] for s in stages}

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
            stage = _map_to_stage(text, stages)
            stage_entries[stage].append({"response_id": rid, "text": text, "question": col})

    rows = []
    for stage in stages:
        entries = stage_entries.get(stage, [])
        if not entries:
            continue
        texts = [e["text"] for e in entries]
        problems = _top_keywords_in_texts(texts, ("불편", "어렵", "복잡", "문제"))
        emotions = _top_keywords_in_texts(texts, ("불안", "혼란", "피로", "불만", "기대"))
        kw = ", ".join(problems[:3]) if problems else "-"
        rows.append(
            {
                "여정 단계": stage,
                "응답 수": len(entries),
                "주요 문제": ", ".join(problems[:5]) or "-",
                "주요 감정": ", ".join(emotions[:3]) or "-",
                "관련 키워드": kw,
                "관련 응답 ID": ", ".join(str(e["response_id"]) for e in entries[:5]),
                "개선 기회": _improvement_for_stage(stage),
                "우선순위": _priority(len(entries), problems),
            }
        )

    return pd.DataFrame(rows)


def _map_to_stage(text: str, stages: list[str]) -> str:
    for stage, keywords in JOURNEY_KW.items():
        if stage not in stages:
            continue
        if any(k in text for k in keywords):
            return stage
    if "예약" in text or "신청" in text:
        return "실행" if "실행" in stages else stages[-2]
    if "다시" in text:
        return "재사용" if "재사용" in stages else stages[0]
    return stages[2] if len(stages) > 2 else stages[0]  # 기본: 탐색


def _top_keywords_in_texts(texts: list[str], keywords: tuple[str, ...]) -> list[str]:
    found = []
    for k in keywords:
        if sum(1 for t in texts if k in t) >= 1:
            found.append(k)
    return found


def _improvement_for_stage(stage: str) -> str:
    mapping = {
        "인지": "첫 화면·가치 메시지 개선",
        "진입": "온보딩·가입 흐름 단순화",
        "탐색": "정보 구조·검색 개선",
        "선택": "비교 기준·추천 정보 제공",
        "실행": "핵심 태스크 플로우 단순화",
        "확인": "상태·알림 명확화",
        "재사용": "리마인드·재방문 동기 설계",
        "이탈": "이탈 원인 분석·복귀 UX",
    }
    return mapping.get(stage, "해당 단계 UX 점검")


def _priority(count: int, problems: list[str]) -> str:
    if count >= 10 and len(problems) >= 2:
        return "높음"
    if count >= 5:
        return "중간"
    return "낮음"
