"""서비스·경쟁사 웹 리서치 및 AI 종합."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

try:
    from duckduckgo_search import DDGS

    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False


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
) -> list[dict[str, str]]:
    """서비스 기본 정보·리뷰·경쟁사 검색."""
    name = service_name.strip()
    desc = (service_description or "")[:80]
    queries = [
        f"{name} 서비스 소개 특징",
        f"{name} 앱 사용자 리뷰 불편",
        f"{name} 경쟁 서비스 비교",
        f"{name} {desc} UX 개선" if desc else f"{name} UX 문제",
    ]
    seen: set[str] = set()
    all_hits: list[dict[str, str]] = []
    for q in queries:
        for hit in web_search(q, max_results=4):
            key = hit.get("title", "") + hit.get("url", "")
            if key in seen:
                continue
            seen.add(key)
            all_hits.append({**hit, "query": q})
    return all_hits[:16]


def _format_snippets_for_prompt(snippets: list[dict[str, str]]) -> str:
    if not snippets:
        return "(웹 검색 결과 없음 — API Key와 네트워크를 확인하세요.)"
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
    model: str = "gpt-4o-mini",
) -> str | None:
    """검색 스니펫 + 설문 맥락을 바탕으로 서비스·경쟁 리서치 리포트 작성."""
    if not api_key:
        return None

    search_block = _format_snippets_for_prompt(snippets)
    user_msg = f"""다음 서비스에 대해 UX 리서치 보조 리포트를 작성하세요.

## 대상 서비스
- 이름: {service_name}
- 설명: {service_description}
- 설문 목적: {survey_purpose}
- 알고 싶은 문제: {known_problems or '-'}

## 설문 분석 요약 (참고)
{survey_summary[:2500]}

## 웹 검색 스니펫 (출처 후보)
{search_block}

## 작성 지침
1) **서비스 기본 파악**: 무엇을 하는 서비스인지, 주요 사용자, 핵심 가치 (검색 스니펫 기반, 불확실하면 '추가 확인 필요')
2) **경쟁·유사 서비스**: 2~4개 나열, 각각 한 줄 설명
3) **업계 공통 불편·이슈**: 설문 키워드와 검색 내용을 연결
4) **경쟁사 해결 사례**: 비슷한 문제를 어떻게 풀었는지 (있을 때만)
5) **설문 결과와의 연결**: 위 리서치가 설문에서 나온 문제와 어떻게 맞는지
6) 검색에 없는 내용은 추측하지 말 것. 한국어, Fact/Interpretation/Action 구조 일부 사용.

15~25줄 분량."""

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "당신은 UX·시장 리서치 분석가입니다. 웹 스니펫과 설문 데이터만 근거로 씁니다.",
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
    Returns: {snippets, report, search_ok, ai_ok}
    """
    if not enabled:
        return {"snippets": [], "report": None, "search_ok": False, "ai_ok": False}

    snippets = gather_search_snippets(service_name, service_description)
    report = None
    if api_key and snippets:
        report = synthesize_market_research(
            service_name,
            service_description,
            survey_purpose,
            known_problems,
            snippets,
            survey_summary,
            api_key,
            model,
        )
    elif api_key and not snippets:
        report = synthesize_market_research(
            service_name,
            service_description,
            survey_purpose,
            known_problems,
            [],
            survey_summary,
            api_key,
            model,
        )

    fallback = _fallback_report(service_name, snippets) if not report else None

    return {
        "snippets": snippets,
        "report": report or fallback,
        "search_ok": bool(snippets),
        "ai_ok": bool(report),
    }


def _fallback_report(service_name: str, snippets: list[dict]) -> str:
    """API 없을 때 검색 스니펫만 나열."""
    if not snippets:
        return (
            f"**{service_name}**: 웹 검색 결과를 가져오지 못했습니다. "
            "`duckduckgo-search` 설치 및 네트워크를 확인하세요."
        )
    lines = [f"### {service_name} — 검색 스니펫 (AI 종합 없음)\n"]
    for i, s in enumerate(snippets[:8], 1):
        lines.append(f"{i}. **{s.get('title', '')}** — {s.get('snippet', '')[:200]}")
    return "\n".join(lines)
