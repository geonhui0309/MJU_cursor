"""가설 기반 분석."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


VERDICT_LABELS = ("지지됨", "부분 지지됨", "반박됨", "판단 보류", "가설 수정 필요")


def run_hypothesis_analysis(
    hypotheses_text: str,
    schema_df: pd.DataFrame,
    quant_results: dict[str, Any],
    qual_results: dict[str, Any],
    keyword_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """입력 가설별 관련 근거 연결 및 판단."""
    hypotheses = _parse_hypotheses(hypotheses_text)
    if not hypotheses:
        return pd.DataFrame()

    all_keywords = []
    if keyword_df is not None and not keyword_df.empty:
        all_keywords = keyword_df["키워드"].tolist()

    qual_texts = _collect_qual_texts(qual_results)
    question_names = schema_df["문항명"].tolist() if not schema_df.empty else []

    rows = []
    for hyp in hypotheses:
        related_q = _match_questions(hyp, question_names)
        related_quant = _match_quant(hyp, quant_results, related_q)
        related_qual = _match_qual(hyp, qual_texts)
        verdict, reason, suggestion = _judge_hypothesis(
            hyp, related_quant, related_qual, all_keywords
        )
        rows.append(
            {
                "가설": hyp,
                "관련 문항": ", ".join(related_q[:5]) or "-",
                "관련 정량 데이터": related_quant[:300] if related_quant else "-",
                "관련 정성 응답": related_qual[:300] if related_qual else "-",
                "지지 여부": verdict,
                "판단 근거": reason,
                "가설 수정 제안": suggestion,
                "추가 검증 필요": verdict in ("판단 보류", "부분 지지됨", "가설 수정 필요"),
            }
        )

    return pd.DataFrame(rows)


def _parse_hypotheses(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    lines = [ln.strip() for ln in re.split(r"[\n;]+", text) if ln.strip()]
    # 번호·불릿 제거
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^[\d]+[\.\)]\s*", "", ln)
        ln = re.sub(r"^[-*•]\s*", "", ln)
        if len(ln) >= 5:
            cleaned.append(ln)
    return cleaned


def _collect_qual_texts(qual_results: dict) -> list[str]:
    texts = []
    per_q = qual_results.get("per_question", {})
    for data in per_q.values():
        for rid, t in data.get("valid_responses", []):
            texts.append(t)
    return texts


def _match_questions(hypothesis: str, questions: list[str]) -> list[str]:
    hyp_tokens = set(re.findall(r"[가-힣]{2,}", hypothesis))
    scored = []
    for q in questions:
        q_tokens = set(re.findall(r"[가-힣]{2,}", q))
        overlap = len(hyp_tokens & q_tokens)
        if overlap:
            scored.append((overlap, q))
    scored.sort(reverse=True)
    return [q for _, q in scored[:5]]


def _match_quant(hypothesis: str, quant_results: dict, related_q: list[str]) -> str:
    snippets = []
    keys = related_q if related_q else list(quant_results.keys())[:3]
    for k in keys:
        if k in quant_results and isinstance(quant_results[k], dict):
            interp = quant_results[k].get("interpretation", "")
            if interp:
                snippets.append(interp)
    if "_cross_analysis" in quant_results:
        for cross in quant_results["_cross_analysis"][:2]:
            if cross.get("interpretation"):
                snippets.append(cross["interpretation"])
    return " ".join(snippets)


def _match_qual(hypothesis: str, texts: list[str]) -> str:
    hyp_words = re.findall(r"[가-힣]{2,}", hypothesis)
    matched = [t for t in texts if any(w in t for w in hyp_words)]
    if not matched:
        # 키워드 유사 매칭
        theme_words = ("불편", "어렵", "복잡", "선택", "가격", "신뢰", "예약", "이탈")
        for w in theme_words:
            if w in hypothesis:
                matched = [t for t in texts if w in t]
                break
    return " | ".join(matched[:3])


def _judge_hypothesis(
    hyp: str,
    quant_evidence: str,
    qual_evidence: str,
    keywords: list[str],
) -> tuple[str, str, str]:
    score = 0
    if quant_evidence:
        score += 1
    if qual_evidence:
        score += 2
    if keywords and any(k in hyp for k in keywords[:10]):
        score += 1

    # 반박 키워드
    refute_signals = ("아니다", "반대", "낮지 않", "높은 편")
    if any(s in qual_evidence for s in refute_signals):
        score -= 1

    if score >= 3:
        verdict = "지지됨"
        reason = "정량·정성 근거가 가설과 방향성이 일치합니다."
        suggestion = "-"
    elif score == 2:
        verdict = "부분 지지됨"
        reason = "일부 데이터는 가설을 지지하나, 전체 응답에서 혼재된 신호가 있습니다."
        suggestion = "가설을 더 구체적인 여정 단계·세그먼트로 좁혀 재검증하세요."
    elif score == 1:
        verdict = "판단 보류"
        reason = "관련 근거가 제한적입니다. 표본 수·문항 매칭을 확인하세요."
        suggestion = "관련 문항을 추가하거나 표본을 늘린 후 재분석하세요."
    else:
        verdict = "가설 수정 필요"
        reason = "현재 데이터에서 가설을 뒷받침할 근거가 부족합니다."
        suggestion = "가설을 응답에서 실제로 반복되는 키워드·문제로 수정하세요."

    if "반박" in qual_evidence or score < 0:
        verdict = "반박됨"
        reason = "일부 응답이 가설과 반대 방향의 신호를 보입니다."
        suggestion = "가설 전제를 재검토하고 대안 가설을 설정하세요."

    return verdict, reason, suggestion
