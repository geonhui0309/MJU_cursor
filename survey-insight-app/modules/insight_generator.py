"""핵심 인사이트·액션 아이템·후속 리서치 생성."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_insights(
    context: dict[str, Any],
    quant_results: dict,
    qual_results: dict,
    keyword_df: pd.DataFrame,
    journey_df: pd.DataFrame,
    hypothesis_df: pd.DataFrame,
    cleaning_summary: dict,
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Fact / Interpretation / Action 구조의 인사이트 생성.
    Returns: (insights, action_items, follow_up_research)
    """
    insights: list[dict] = []
    candidates = _build_candidates(quant_results, qual_results, keyword_df, journey_df, hypothesis_df)

    for i, cand in enumerate(candidates[:7]):
        if len(insights) >= 7:
            break
        insights.append(_format_insight(cand, i + 1, context))

    while len(insights) < 3:
        insights.append(_fallback_insight(len(insights) + 1, cleaning_summary, context))

    actions = _generate_actions(insights)
    follow_up = _generate_follow_up(insights, hypothesis_df, context)
    return insights, actions, follow_up


def _build_candidates(
    quant_results: dict,
    qual_results: dict,
    keyword_df: pd.DataFrame,
    journey_df: pd.DataFrame,
    hypothesis_df: pd.DataFrame,
) -> list[dict]:
    candidates: list[dict] = []

    # 키워드 기반
    if keyword_df is not None and not keyword_df.empty:
        neg_kw = keyword_df[keyword_df["감성 방향"].isin(["부정", "요구"])].head(5)
        for _, row in neg_kw.iterrows():
            candidates.append(
                {
                    "title": f"'{row['키워드']}' 관련 불편·요구가 두드러짐",
                    "problem": row.get("주요 맥락", ""),
                    "fact": f"키워드 '{row['키워드']}' {row['빈도']}회 언급 ({row.get('감성 방향')})",
                    "interpretation": f"응답자들이 {row['키워드']}와 관련된 경험에서 마찰을 느끼고 있습니다.",
                    "action": row.get("서비스 개선 연결점", "관련 UX 개선"),
                    "journey": row.get("카테고리", ""),
                    "priority": "높음" if row["빈도"] >= 5 else "중간",
                    "confidence": "중간",
                    "questions": row.get("관련 문항", ""),
                    "response_ids": "",
                    "representative": row.get("대표 응답", ""),
                }
            )

    # 여정 기반
    if journey_df is not None and not journey_df.empty:
        top_j = journey_df.sort_values("응답 수", ascending=False).head(3)
        for _, row in top_j.iterrows():
            candidates.append(
                {
                    "title": f"{row['여정 단계']} 단계에서 주요 마찰 발생",
                    "problem": row.get("주요 문제", ""),
                    "fact": f"{row['여정 단계']} 단계 응답 {row['응답 수']}건, 관련 ID: {row.get('관련 응답 ID', '')}",
                    "interpretation": f"사용자 여정의 {row['여정 단계']} 단계에서 개선 여지가 큽니다.",
                    "action": row.get("개선 기회", ""),
                    "journey": row["여정 단계"],
                    "priority": row.get("우선순위", "중간"),
                    "confidence": "중간",
                    "questions": "-",
                    "response_ids": row.get("관련 응답 ID", ""),
                    "representative": "",
                }
            )

    # 정량 기반
    for col, res in quant_results.items():
        if col.startswith("_") or not isinstance(res, dict):
            continue
        if res.get("type") == "리커트 척도형" and res.get("mean", 5) < 3.0:
            candidates.append(
                {
                    "title": f"'{col}' 점수가 낮아 개선 필요",
                    "problem": "만족도·평가 점수 저조",
                    "fact": res.get("interpretation", ""),
                    "interpretation": "해당 영역이 전반적 경험을 저해할 가능성이 있습니다.",
                    "action": f"'{col}' 관련 핵심 접점 UX 개선",
                    "journey": "실행",
                    "priority": "높음",
                    "confidence": "높음",
                    "questions": col,
                    "response_ids": "",
                    "representative": "",
                }
            )

    # 가설 기반
    if hypothesis_df is not None and not hypothesis_df.empty:
        supported = hypothesis_df[hypothesis_df["지지 여부"].isin(["지지됨", "부분 지지됨"])]
        for _, row in supported.head(2).iterrows():
            candidates.append(
                {
                    "title": f"가설 검증: {row['가설'][:40]}...",
                    "problem": row["가설"],
                    "fact": row.get("판단 근거", ""),
                    "interpretation": f"판단: {row['지지 여부']}",
                    "action": row.get("가설 수정 제안", "관련 영역 개선"),
                    "journey": "-",
                    "priority": "중간",
                    "confidence": "중간" if row["지지 여부"] == "부분 지지됨" else "높음",
                    "questions": row.get("관련 문항", ""),
                    "response_ids": "",
                    "representative": row.get("관련 정성 응답", "")[:150],
                }
            )

    integrated = qual_results.get("integrated", {})
    for prob in integrated.get("top_problems", [])[:2]:
        candidates.append(
            {
                "title": f"반복 언급 문제: '{prob.get('keyword', '')}'",
                "problem": prob.get("keyword", ""),
                "fact": f"키워드 '{prob.get('keyword')}' {prob.get('count')}회 반복",
                "interpretation": "주관식 응답에서 동일 문제가 반복적으로 관찰됩니다.",
                "action": "해당 문제 영역의 UX·정보 구조 점검",
                "journey": "탐색",
                "priority": "높음",
                "confidence": "중간",
                "questions": "주관식 통합",
                "response_ids": "",
                "representative": "",
            }
        )

    return candidates


def _format_insight(cand: dict, idx: int, context: dict) -> dict:
    return {
        "인사이트 제목": cand.get("title", f"인사이트 {idx}"),
        "사용자가 실제로 겪는 문제": cand.get("problem", ""),
        "Fact": cand.get("fact", ""),
        "Interpretation": cand.get("interpretation", ""),
        "Action": cand.get("action", ""),
        "서비스 개선 기회": cand.get("action", ""),
        "관련 여정 단계": cand.get("journey", ""),
        "관련 사용자 세그먼트": context.get("target_users", "전체"),
        "우선순위": cand.get("priority", "중간"),
        "판단 신뢰도": cand.get("confidence", "중간"),
        "주의할 점": "표본 수가 적은 세그먼트는 과대해석을 피하고 추가 검증이 필요합니다.",
        "관련 문항명": cand.get("questions", ""),
        "관련 응답 ID": cand.get("response_ids", ""),
        "대표 응답": cand.get("representative", ""),
        "추가 검증 필요": cand.get("confidence") != "높음",
    }


def _fallback_insight(idx: int, cleaning_summary: dict, context: dict) -> dict:
    rel = cleaning_summary.get("reliability", "보통")
    return {
        "인사이트 제목": f"데이터 품질 기반 분석 범위 안내 ({idx})",
        "사용자가 실제로 겪는 문제": context.get("known_problems", "입력된 문제 영역"),
        "Fact": f"분석 포함 응답 {cleaning_summary.get('included_count', 0)}건, 데이터 신뢰도: {rel}",
        "Interpretation": "현재 데이터로 도출 가능한 인사이트 범위가 제한될 수 있습니다.",
        "Action": "추가 응답 수집 또는 정제 검토 후 재분석을 권장합니다.",
        "서비스 개선 기회": "설문 설계·표본 확대",
        "관련 여정 단계": "-",
        "관련 사용자 세그먼트": context.get("target_users", "전체"),
        "우선순위": "낮음",
        "판단 신뢰도": "낮음",
        "주의할 점": "추가 데이터 필요",
        "관련 문항명": "-",
        "관련 응답 ID": "-",
        "대표 응답": "",
        "추가 검증 필요": True,
    }


def _generate_actions(insights: list[dict]) -> list[dict]:
    categories = {
        "높음": ("영향도가 높은 항목", "높음"),
        "중간": ("빠르게 개선 가능한 항목", "중간"),
        "낮음": ("추가 리서치가 필요한 항목", "낮음"),
    }
    actions = []
    for ins in insights:
        pri = ins.get("우선순위", "중간")
        cat, impact = categories.get(pri, categories["중간"])
        actions.append(
            {
                "개선안": ins.get("Action", ""),
                "관련 인사이트": ins.get("인사이트 제목", ""),
                "기대 효과": "해당 마찰 지점 완화",
                "실행 난이도": "중간" if pri == "높음" else "낮음",
                "영향도": impact,
                "우선순위": pri,
                "분류": cat,
                "추가 검증 방법": "사용성 테스트 또는 후속 설문",
            }
        )
    return actions


def _generate_follow_up(
    insights: list[dict], hypothesis_df: pd.DataFrame, context: dict
) -> list[str]:
    items = [
        "추가 설문 문항: 해당 불편이 발생한 구체적 화면·단계를 선택형으로 물어보세요.",
        "인터뷰 질문: 최근 사용 시 가장 망설였던 순간은 언제였나요?",
        "사용성 테스트: 핵심 태스크 완료 과정을 관찰하여 설문 응답과 실제 행동을 대조하세요.",
    ]
    if hypothesis_df is not None and not hypothesis_df.empty:
        pending = hypothesis_df[hypothesis_df["추가 검증 필요"] == True]  # noqa: E712
        for _, row in pending.head(2).iterrows():
            items.append(f"검증 필요 가설: {row['가설']}")
    if context.get("known_problems"):
        items.append(f"심화 조사 권장 영역: {context['known_problems']}")
    return items
