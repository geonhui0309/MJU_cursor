"""서비스·경쟁사 웹 리서치 및 AI 종합 (동음이의어·무관 결과 필터)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

try:
    from duckduckgo_search import DDGS

    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

# 동음이의어·짧은 서비스명 → 검색·필터 강화
KNOWN_SERVICES: dict[str, dict[str, Any]] = {
    "당근": {
        "canonical": "당근마켓",
        "aliases": ("당근마켓", "당근 마켓", "karrot", "daangn", "당근마켓 앱"),
        "positive": (
            "중고", "거래", "앱", "마켓", "market", "karrot", "daangn",
            "이웃", "동네", "채팅", "사기", "직거래", "플랫폼", "ux", "리뷰",
        ),
        "negative": (
            "채소", "영양", "비타민", "요리", "레시피", "재배", "농산", "밭",
            "주스", "효능", "식품", "먹", "요리법", "씨앗", "당근즙", "비타민a",
            "carrot cake", "carrot soup",
        ),
        "search_extra": ("당근마켓 중고거래 앱", "karrot market korea app"),
    },
}

DIGITAL_HINTS = (
    "앱", "app", "서비스", "플랫폼", "웹", "사이트", "ux", "ui",
    "로그인", "회원", "예약", "결제", "중고", "거래", "리뷰", "사용자",
)
FOOD_AGRICULTURE_HINTS = (
    "채소", "음식", "요리", "영양", "재배", "농산", "레시피", "효능",
)

RELEVANCE_THRESHOLD = 1


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _known_key(service_name: str) -> str | None:
    name = service_name.strip()
    if name in KNOWN_SERVICES:
        return name
    for key, cfg in KNOWN_SERVICES.items():
        if key in name or name in cfg.get("canonical", ""):
            return key
    return None


def _tokens_from_description(description: str) -> list[str]:
    if not description:
        return []
    words = re.findall(r"[가-힣a-zA-Z0-9]{2,}", description)
    return [w for w in words if len(w) >= 2][:12]


def build_relevance_context(
    service_name: str,
    service_description: str = "",
) -> dict[str, Any]:
    """검색 쿼리·필터용 맥락 (동음이의어 대응)."""
    name = service_name.strip()
    desc = (service_description or "").strip()
    desc_lower = _norm(desc)
    known_key = _known_key(name)
    known = KNOWN_SERVICES.get(known_key or "", {})

    positive: set[str] = set()
    negative: set[str] = set()
    canonical = name
    aliases: list[str] = []

    if known:
        canonical = known.get("canonical", name)
        aliases = list(known.get("aliases", ()))
        positive.update(k.lower() for k in known.get("positive", ()))
        negative.update(k.lower() for k in known.get("negative", ()))

    positive.update(_norm(t) for t in _tokens_from_description(desc))
    for hint in DIGITAL_HINTS:
        if hint in desc_lower:
            positive.add(hint)

    # 설명에 디지털 맥락이 있으면 음식·농업 키워드는 강하게 제외
    has_digital_context = bool(positive & set(DIGITAL_HINTS)) or any(
        h in desc_lower for h in DIGITAL_HINTS
    )
    ambiguous_short_name = len(name) <= 3 or known_key is not None
    if ambiguous_short_name or has_digital_context:
        negative.update(h.lower() for h in FOOD_AGRICULTURE_HINTS)

    return {
        "service_name": name,
        "canonical_name": canonical,
        "description": desc,
        "aliases": aliases,
        "positive": positive,
        "negative": negative,
        "ambiguous": ambiguous_short_name,
        "known_key": known_key,
        "search_extra": list(known.get("search_extra", ())),
    }


def build_search_queries(ctx: dict[str, Any]) -> list[str]:
    """서비스 맥락이 드러나는 검색어 목록."""
    name = ctx["service_name"]
    canonical = ctx["canonical_name"]
    desc = ctx["description"][:80]
    aliases = ctx["aliases"]
    extra = ctx["search_extra"]

    primary = canonical if canonical != name else name
    queries: list[str] = []

    if desc:
        queries.append(f"{primary} {desc} 앱 서비스")
    queries.append(f"{primary} 앱 사용자 리뷰 불편")
    queries.append(f"{primary} 경쟁 서비스 비교")
    queries.append(f"{primary} UX 개선 이슈")

    for alias in aliases[:2]:
        queries.append(f"{alias} 서비스 소개")

    queries.extend(extra)

    # 동음이의어일 때 원 짧은 이름 단독 검색은 제외
    if not ctx["ambiguous"]:
        queries.insert(0, f"{name} 서비스 소개 특징")

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:6]


def score_snippet_relevance(
    hit: dict[str, str],
    ctx: dict[str, Any],
) -> int:
    """높을수록 서비스 관련. 음수면 무관."""
    text = _norm(
        f"{hit.get('title', '')} {hit.get('snippet', '')} {hit.get('url', '')}"
    )
    score = 0

    for kw in ctx["positive"]:
        if kw and kw in text:
            score += 2

    for kw in ctx["negative"]:
        if kw and kw in text:
            score -= 4

    canonical = _norm(ctx["canonical_name"])
    if canonical and canonical in text:
        score += 3

    for alias in ctx["aliases"]:
        if _norm(alias) in text:
            score += 2

    if ctx["service_name"] in text and ctx["ambiguous"]:
        # 짧은 이름만 있고 디지털 신호 없으면 감점
        if not any(p in text for p in ctx["positive"] if len(p) >= 2):
            score -= 2

    return score


def filter_relevant_snippets(
    snippets: list[dict[str, str]],
    ctx: dict[str, Any],
) -> tuple[list[dict[str, str]], int]:
    """관련 스니펫만 남기고 제외 건수 반환."""
    kept: list[dict[str, str]] = []
    excluded = 0
    threshold = RELEVANCE_THRESHOLD
    if ctx["ambiguous"]:
        threshold = max(threshold, 2)

    for hit in snippets:
        rel = score_snippet_relevance(hit, ctx)
        if rel >= threshold:
            kept.append({**hit, "relevance_score": rel})
        else:
            excluded += 1

    kept.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return kept, excluded


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """DuckDuckGo 텍스트 검색."""
    if not DDG_AVAILABLE:
        return []
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": h.get("title", ""),
                "snippet": (h.get("body") or h.get("snippet") or "")[:500],
                "url": h.get("href", h.get("link", "")),
            }
            for h in hits
        ]
    except Exception:
        return []


def gather_search_snippets(
    service_name: str,
    service_description: str = "",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """서비스 맥락 검색 후 관련 스니펫만 반환."""
    ctx = build_relevance_context(service_name, service_description)
    queries = build_search_queries(ctx)

    seen: set[str] = set()
    raw_hits: list[dict[str, str]] = []
    for q in queries:
        for hit in web_search(q, max_results=4):
            key = hit.get("title", "") + hit.get("url", "")
            if key in seen:
                continue
            seen.add(key)
            raw_hits.append({**hit, "query": q})

    filtered, excluded = filter_relevant_snippets(raw_hits, ctx)
    meta = {
        "service_name": ctx["service_name"],
        "canonical_name": ctx["canonical_name"],
        "raw_count": len(raw_hits),
        "filtered_count": len(filtered),
        "excluded_count": excluded,
        "ambiguous": ctx["ambiguous"],
    }
    return filtered[:16], meta


def _format_snippets_for_prompt(snippets: list[dict[str, str]]) -> str:
    if not snippets:
        return "(관련 웹 검색 결과 없음 — 서비스 설명을 구체적으로 입력해 주세요.)"
    lines = []
    for i, s in enumerate(snippets, 1):
        lines.append(
            f"[{i}] ({s.get('query', '')})\n"
            f"제목: {s.get('title', '')}\n"
            f"요약: {s.get('snippet', '')}\n"
            f"URL: {s.get('url', '')}"
        )
    return "\n\n".join(lines)


def synthesize_market_research(
    service_name: str,
    service_description: str,
    survey_purpose: str,
    known_problems: str,
    snippets: list[dict[str, str]],
    survey_summary: str,
    api_key: str,
    canonical_name: str = "",
    model: str = "gpt-4o-mini",
) -> str | None:
    """검색 스니펫 + 설문 맥락을 바탕으로 서비스·경쟁 리서치 리포트 작성."""
    if not api_key:
        return None

    target = canonical_name or service_name
    search_block = _format_snippets_for_prompt(snippets)
    user_msg = f"""다음 **디지털 서비스/앱**에 대해 UX 리서치 보조 리포트를 작성하세요.

## 대상 서비스 (동음이의어 주의)
- 입력 이름: {service_name}
- 분석 대상(실제 서비스): {target}
- 설명: {service_description}
- 설문 목적: {survey_purpose}
- 알고 싶은 문제: {known_problems or '-'}

## 설문 분석 요약 (참고)
{survey_summary[:2500]}

## 웹 검색 스니펫 (이미 서비스 관련만 필터됨)
{search_block}

## 필수 규칙
- 대상은 **{target}** 디지털 서비스/앱입니다. 채소·음식·농업·영양·요리 등 **동음이의어(예: 당근 채소)** 내용은 절대 쓰지 마세요.
- 스니펫에 없는 내용은 추측하지 말고 "추가 확인 필요"라고 하세요.
- 스니펫이 비어 있으면 일반 상식으로 채우지 말고, 설명·설문만으로 제한적으로 작성하세요.

## 작성 지침
1) **서비스 기본 파악**: 무엇을 하는 서비스인지, 주요 사용자, 핵심 가치 (스니펫 기반)
2) **경쟁·유사 서비스**: 2~4개 나열, 각각 한 줄 설명
3) **업계 공통 불편·이슈**: 설문 키워드와 검색 내용 연결
4) **경쟁사 해결 사례**: 있을 때만
5) **설문 결과와의 연결**
6) 한국어, Fact/Interpretation/Action 구조 일부 사용. 15~25줄."""

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 UX·시장 리서치 분석가입니다. "
                    "웹 스니펫과 설문 데이터만 근거로 씁니다. "
                    "동음이의어(서비스명과 무관한 일반 단어 의미)는 무시합니다."
                ),
            },
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
    }

    req = urllib.request.Request(
        url="https://api.openai.com/v1/chat/completions",
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
        return None


def run_service_research(
    service_name: str,
    service_description: str,
    survey_purpose: str,
    known_problems: str,
    survey_summary: str,
    api_key: str = "",
    model: str = "gpt-4o-mini",
    enabled: bool = True,
) -> dict[str, Any]:
    """
    웹 검색 + AI 종합.
    Returns: {snippets, report, search_ok, ai_ok, meta}
    """
    if not enabled:
        return {
            "snippets": [],
            "report": None,
            "search_ok": False,
            "ai_ok": False,
            "meta": {},
        }

    snippets, meta = gather_search_snippets(service_name, service_description)
    canonical = meta.get("canonical_name", service_name)

    report = None
    if api_key:
        report = synthesize_market_research(
            service_name,
            service_description,
            survey_purpose,
            known_problems,
            snippets,
            survey_summary,
            api_key,
            canonical_name=canonical,
            model=model,
        )

    fallback = _fallback_report(service_name, canonical, snippets, meta) if not report else None

    return {
        "snippets": snippets,
        "report": report or fallback,
        "search_ok": bool(snippets),
        "ai_ok": bool(report),
        "meta": meta,
    }


def _fallback_report(
    service_name: str,
    canonical: str,
    snippets: list[dict],
    meta: dict[str, Any],
) -> str:
    """API 없을 때 검색 스니펫만 나열."""
    target = canonical or service_name
    excluded = meta.get("excluded_count", 0)
    note = ""
    if excluded:
        note = f"\n\n_서비스와 무관한 검색 결과 {excluded}건은 제외했습니다._"
    if not snippets:
        return (
            f"**{target}**: 서비스와 관련된 웹 검색 결과를 찾지 못했습니다. "
            "서비스 설명에 '중고거래 앱', '플랫폼' 등 맥락을 적어 주세요. "
            "`duckduckgo-search` 설치 및 네트워크도 확인하세요."
            + note
        )
    lines = [f"### {target} — 검색 스니펫 (AI 종합 없음)\n"]
    for i, s in enumerate(snippets[:8], 1):
        lines.append(f"{i}. **{s.get('title', '')}** — {s.get('snippet', '')[:200]}")
    lines.append(note)
    return "\n".join(lines)
