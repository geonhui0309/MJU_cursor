"""주관식 정성 분석."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from modules.schema_detector import get_columns_by_type


def run_qualitative_analysis(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
    excluded_ids: set[int] | None = None,
    cleaning_log: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """문항별·통합 정성 분석."""
    excluded_ids = excluded_ids or set()
    text_cols = get_columns_by_type(schema_df, "장문형") + get_columns_by_type(schema_df, "단답형")

    per_question: dict[str, Any] = {}
    all_texts: list[tuple[int, str, str]] = []

    for col in text_cols:
        if col not in df.columns:
            continue
        entries = _analyze_question(df, col, excluded_ids, cleaning_log)
        per_question[col] = entries
        for rid, text in entries.get("valid_responses", []):
            all_texts.append((rid, col, text))

    integrated = _integrate(all_texts)
    export_rows = _build_export(per_question, integrated)
    export_df = pd.DataFrame(export_rows)

    return {"per_question": per_question, "integrated": integrated}, export_df


def _analyze_question(
    df: pd.DataFrame,
    col: str,
    excluded_ids: set[int],
    cleaning_log: pd.DataFrame | None,
) -> dict[str, Any]:
    exclude_cells: set[tuple[int, str]] = set()
    if cleaning_log is not None and not cleaning_log.empty:
        ex = cleaning_log[
            (cleaning_log["question"] == col) & (cleaning_log["included_in_analysis"] == False)  # noqa: E712
        ]
        for _, row in ex.iterrows():
            exclude_cells.add((int(row["response_id"]), col))

    valid: list[tuple[int, str]] = []
    for _, row in df.iterrows():
        rid = int(row["response_id"]) if "response_id" in df.columns else _
        if rid in excluded_ids:
            continue
        if (rid, col) in exclude_cells:
            continue
        val = row.get(col)
        if pd.isna(val) or str(val).strip() == "":
            continue
        text = str(val).strip()
        if len(text) >= 3:
            valid.append((rid, text))

    themes = _extract_themes([t for _, t in valid])
    problems = _extract_patterns(valid, ("불편", "어렵", "복잡", "불만", "문제", "힘들"))
    improvements = _extract_patterns(valid, ("개선", "추가", "있으면", "원해", "필요", "바랍"))
    representatives = _pick_representatives(valid, 5)

    return {
        "response_count": len(valid),
        "summary": _summarize_texts([t for _, t in valid]),
        "recurring_problems": problems,
        "improvement_requests": improvements,
        "themes": themes,
        "representative_responses": representatives,
        "valid_responses": valid,
        "specificity": "높음" if _avg_len(valid) >= 30 else ("보통" if _avg_len(valid) >= 15 else "낮음"),
    }


def _extract_themes(texts: list[str]) -> list[dict[str, Any]]:
    """간단 키워드 기반 테마 추출."""
    theme_keywords = {
        "사용성": ("찾기", "어렵", "복잡", "단계", "버튼", "메뉴"),
        "정보/선택": ("비교", "가격", "정보", "선택", "기준"),
        "신뢰": ("믿", "신뢰", "후기", "리뷰"),
        "재사용": ("다시", "재사용", "의향"),
    }
    counts: dict[str, int] = {k: 0 for k in theme_keywords}
    for text in texts:
        for theme, kws in theme_keywords.items():
            if any(k in text for k in kws):
                counts[theme] += 1
    return [{"theme": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1]) if v > 0]


def _extract_patterns(
    valid: list[tuple[int, str]], keywords: tuple[str, ...]
) -> list[dict[str, Any]]:
    matched = [(rid, t) for rid, t in valid if any(k in t for k in keywords)]
    counter = Counter()
    for _, t in matched:
        for k in keywords:
            if k in t:
                counter[k] += 1
    return [
        {"keyword": k, "count": v, "example_ids": [rid for rid, t in matched if k in t][:3]}
        for k, v in counter.most_common(5)
    ]


def _pick_representatives(valid: list[tuple[int, str]], n: int) -> list[dict]:
    if not valid:
        return []
    sorted_v = sorted(valid, key=lambda x: len(x[1]), reverse=True)
    mid = len(sorted_v) // 2
    picks = [sorted_v[0], sorted_v[mid], sorted_v[-1]] if len(sorted_v) >= 3 else sorted_v
    seen = set()
    result = []
    for rid, text in picks[:n]:
        if rid in seen:
            continue
        seen.add(rid)
        result.append({"response_id": rid, "text": text[:300]})
    return result


def _summarize_texts(texts: list[str]) -> str:
    if not texts:
        return "유효한 주관식 응답이 없습니다."
    n = len(texts)
    avg = sum(len(t) for t in texts) / n
    return f"총 {n}건의 응답이 수집되었으며, 평균 응답 길이는 {avg:.0f}자입니다."


def _integrate(all_texts: list[tuple[int, str, str]]) -> dict[str, Any]:
    if not all_texts:
        return {"summary": "주관식 응답 없음", "top_problems": [], "top_improvements": []}
    texts = [t for _, _, t in all_texts]
    problems = _count_keywords(texts, ("불편", "어렵", "복잡", "불만"))
    improvements = _count_keywords(texts, ("개선", "추가", "필요", "원해"))
    return {
        "summary": f"전체 {len(texts)}건의 주관식 응답을 통합 분석했습니다.",
        "top_problems": problems[:5],
        "top_improvements": improvements[:5],
        "latent_needs": _infer_latent_needs(texts),
    }


def _count_keywords(texts: list[str], keywords: tuple[str, ...]) -> list[dict]:
    c = Counter()
    for t in texts:
        for k in keywords:
            if k in t:
                c[k] += 1
    return [{"keyword": k, "count": v} for k, v in c.most_common()]


def _infer_latent_needs(texts: list[str]) -> list[str]:
    needs = []
    if sum("비교" in t or "선택" in t for t in texts) >= 3:
        needs.append("선택 기준·비교 정보에 대한 니즈")
    if sum("가격" in t for t in texts) >= 2:
        needs.append("가격 투명성·신뢰에 대한 니즈")
    if sum("추천" in t or "자동" in t for t in texts) >= 2:
        needs.append("의사결정 지원(추천) 기능 니즈")
    return needs


def _avg_len(valid: list[tuple[int, str]]) -> float:
    if not valid:
        return 0
    return sum(len(t) for _, t in valid) / len(valid)


def _build_export(per_q: dict, integrated: dict) -> list[dict]:
    rows = []
    for col, data in per_q.items():
        for rep in data.get("representative_responses", []):
            rows.append(
                {
                    "question": col,
                    "response_id": rep["response_id"],
                    "representative_text": rep["text"],
                    "specificity": data.get("specificity"),
                }
            )
    rows.append({"question": "(통합)", "summary": integrated.get("summary")})
    return rows
