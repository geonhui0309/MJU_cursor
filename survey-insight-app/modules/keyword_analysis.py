"""키워드 분석."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

# UX 리서치 맥락 키워드 사전
POSITIVE_CTX = ("좋", "편리", "유용", "만족", "추천", "빠르", "쉬웠")
NEGATIVE_CTX = ("불편", "어렵", "복잡", "불만", "불안", "피곤", "실망")
FUNCTION_KW = ("예약", "알림", "검색", "결제", "로그인", "메뉴", "버튼", "기능", "화면")
JOURNEY_KW = {
    "인지": ("알게", "처음", "광고", "소개"),
    "진입": ("가입", "시작", "첫"),
    "탐색": ("찾", "검색", "둘러"),
    "선택": ("비교", "선택", "고르"),
    "실행": ("예약", "신청", "결제", "등록"),
    "확인": ("확인", "알림", "상태"),
    "재사용": ("다시", "재방", "재사용"),
    "이탈": ("그만", "안 쓸", "삭제"),
}

STOPWORDS = {
    "그리고", "하지만", "정말", "너무", "있는", "없는", "이런", "저런", "것", "수",
    "등", "때", "더", "좀", "잘", "좀", "해서", "합니다", "습니다", "해요", "어요",
}


def run_keyword_analysis(
    df: pd.DataFrame,
    text_columns: list[str],
    excluded_ids: set[int] | None = None,
) -> pd.DataFrame:
    """맥락 기반 키워드 추출."""
    excluded_ids = excluded_ids or set()
    all_entries: list[tuple[int, str, str]] = []

    for col in text_columns:
        if col not in df.columns:
            continue
        for _, row in df.iterrows():
            rid = int(row["response_id"]) if "response_id" in df.columns else _
            if rid in excluded_ids:
                continue
            text = str(row.get(col, "")).strip()
            if len(text) >= 3:
                all_entries.append((rid, col, text))

    if not all_entries:
        return pd.DataFrame()

    keyword_stats = _build_keyword_table(all_entries)
    return keyword_stats


def _tokenize_korean(text: str) -> list[str]:
    """간단 한국어 토큰화 (공백 + 2글자 이상 명사 후보)."""
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    tokens = []
    for word in text.split():
        if len(word) >= 2 and word not in STOPWORDS:
            tokens.append(word)
    # 2-3글자 연속 한글 추출
    for m in re.finditer(r"[가-힣]{2,6}", text):
        w = m.group()
        if w not in STOPWORDS and len(w) >= 2:
            tokens.append(w)
    return tokens


def _build_keyword_table(entries: list[tuple[int, str, str]]) -> pd.DataFrame:
    kw_context: dict[str, list[tuple[int, str, str]]] = {}
    for rid, col, text in entries:
        tokens = _tokenize_korean(text)
        for tok in set(tokens):
            kw_context.setdefault(tok, []).append((rid, col, text))

    rows: list[dict[str, Any]] = []
    for kw, ctx_list in sorted(kw_context.items(), key=lambda x: -len(x[1]))[:40]:
        if len(ctx_list) < 2 and kw not in FUNCTION_KW:
            continue
        freq = len(ctx_list)
        sample_texts = [t for _, _, t in ctx_list[:3]]
        sentiment = _context_sentiment(sample_texts)
        category = _categorize_keyword(kw)
        improvement = _improvement_link(kw, sample_texts)
        rep = sample_texts[0][:120] if sample_texts else ""
        related_q = ", ".join(sorted({c for _, c, _ in ctx_list[:5]}))

        rows.append(
            {
                "키워드": kw,
                "빈도": freq,
                "주요 맥락": _summarize_context(sample_texts),
                "감성 방향": sentiment,
                "카테고리": category,
                "관련 문항": related_q,
                "대표 응답": rep,
                "서비스 개선 연결점": improvement,
            }
        )

    return pd.DataFrame(rows).sort_values("빈도", ascending=False).head(30)


def _context_sentiment(texts: list[str]) -> str:
    combined = " ".join(texts)
    pos = sum(1 for k in POSITIVE_CTX if k in combined)
    neg = sum(1 for k in NEGATIVE_CTX if k in combined)
    if pos > neg:
        return "긍정"
    if neg > pos:
        return "부정"
    if any(k in combined for k in ("추가", "있으면", "원해")):
        return "요구"
    return "중립"


def _categorize_keyword(kw: str) -> str:
    if kw in FUNCTION_KW or any(f in kw for f in FUNCTION_KW):
        return "기능"
    for stage, kws in JOURNEY_KW.items():
        if kw in kws or any(k in kw for k in kws):
            return f"여정-{stage}"
    if any(k in kw for k in POSITIVE_CTX):
        return "긍정 맥락"
    if any(k in kw for k in NEGATIVE_CTX):
        return "부정 맥락"
    return "일반"


def _improvement_link(kw: str, texts: list[str]) -> str:
    links = {
        "예약": "예약 플로우 단순화",
        "가격": "가격 정보 구조화",
        "비교": "비교 기준·정보 제공",
        "알림": "알림 기능 강화",
        "검색": "검색·탐색 UX 개선",
        "복잡": "프로세스 단순화",
        "추천": "추천 로직 설계",
    }
    for k, v in links.items():
        if k in kw or any(k in t for t in texts):
            return v
    return "관련 영역 UX 점검"


def _summarize_context(texts: list[str]) -> str:
    snippets = []
    for t in texts[:2]:
        snippets.append(t[:40] + ("..." if len(t) > 40 else ""))
    return " / ".join(snippets)
