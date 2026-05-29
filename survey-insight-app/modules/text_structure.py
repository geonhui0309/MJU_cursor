"""텍스트 구조 분석."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

STRUCTURE_TYPES = [
    "문제 제기형",
    "원인 설명형",
    "감정 표현형",
    "요구사항 제안형",
    "비교형",
    "맥락 설명형",
    "모순형",
    "단순 반응형",
    "해석 불가형",
]


def run_text_structure_analysis(
    df: pd.DataFrame,
    text_columns: list[str],
    excluded_ids: set[int] | None = None,
) -> pd.DataFrame:
    """주관식 응답의 텍스트 구조 분류."""
    excluded_ids = excluded_ids or set()
    rows: list[dict[str, Any]] = []

    for col in text_columns:
        if col not in df.columns:
            continue
        for _, row in df.iterrows():
            rid = int(row["response_id"]) if "response_id" in df.columns else _
            if rid in excluded_ids:
                continue
            text = str(row.get(col, "")).strip()
            if not text or len(text) < 2:
                continue

            stype = _classify_structure(text)
            parsed = _parse_structure(text, stype)
            rows.append(
                {
                    "response_id": rid,
                    "question": col,
                    "original_answer": text[:500],
                    "text_structure": stype,
                    "main_claim": parsed.get("main_claim", ""),
                    "reason_or_cause": parsed.get("reason", ""),
                    "emotion": parsed.get("emotion", ""),
                    "improvement_request": parsed.get("improvement", ""),
                    "analysis_memo": parsed.get("memo", ""),
                }
            )

    return pd.DataFrame(rows)


def _classify_structure(text: str) -> str:
    if len(text) < 4 or re.match(r"^[\W\d]+$", text):
        return "해석 불가형"
    if len(text) < 8 and not any(c in text for c in ("다", "요", "음")):
        return "단순 반응형"

    if _has_contradiction(text):
        return "모순형"
    if any(k in text for k in ("보다", "대비", "비해", "vs")):
        return "비교형"
    if any(k in text for k in ("추가", "있으면", "개선", "바랐", "원해", "필요")):
        return "요구사항 제안형"
    if any(k in text for k in ("때문", "해서", "어려웠", "못 찾", "이유")):
        return "원인 설명형"
    if any(k in text for k in ("불편", "복잡", "어렵", "문제", "불만")):
        return "문제 제기형"
    if any(k in text for k in ("좋", "싫", "답답", "불안", "피곤", "짜증", "기대")):
        return "감정 표현형"
    if len(text) > 40:
        return "맥락 설명형"
    return "문제 제기형"


def _has_contradiction(text: str) -> bool:
    return ("좋" in text or "만족" in text) and ("안" in text or "불편" in text or "다시" in text)


def _parse_structure(text: str, stype: str) -> dict[str, str]:
    emotion = ""
    for e in ("불안", "혼란", "피로", "불만", "기대", "만족", "실망"):
        if e in text or (e == "불만" and "불편" in text):
            emotion = e
            break

    improvement = ""
    if stype == "요구사항 제안형":
        improvement = _extract_improvement(text)

    main_claim = text[:80] if len(text) > 80 else text
    reason = ""
    if "때문" in text:
        parts = text.split("때문")
        reason = parts[0][-30:] if parts else ""

    return {
        "main_claim": main_claim,
        "reason": reason,
        "emotion": emotion,
        "improvement": improvement,
        "memo": f"{stype} 패턴으로 분류됨",
    }


def _extract_improvement(text: str) -> str:
    for pat in ("추가", "개선", "단순", "줄여", "명확"):
        if pat in text:
            idx = text.find(pat)
            return text[max(0, idx - 5) : idx + 20]
    return text[:40]
