"""OpenAI 기반 설문 분석 해석 (규칙 기반 결과 보강)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def get_api_key(session_key: str = "") -> str:
    """환경변수 우선, 없으면 세션에 입력한 키."""
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    return (session_key or "").strip()


def load_prompt(name: str, fallback: str = "") -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def call_openai_interpretation(
    section_name: str,
    payload_text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    enabled: bool = True,
) -> str | None:
    """OpenAI Chat Completions 호출. 실패 시 None."""
    if not enabled or not api_key or not payload_text.strip():
        return None

    system_msg = load_prompt(
        "system_prompt.md",
        "당신은 UX Research 보조 분석가입니다. 데이터에 없는 내용을 추측하지 마세요.",
    )

    user_msg = f"""섹션: {section_name}

아래는 Google Forms 설문 CSV에 대한 **규칙 기반 분석 결과**입니다.
이 숫자·키워드·응답만 근거로 Fact / Interpretation / Action 형식으로 짧게 해석하세요.

{payload_text}

요구:
1) Fact: 데이터에서 직접 확인 가능한 내용만
2) Interpretation: UX Research 관점 해석
3) Action: 실행 가능한 개선 제안 2~3개
4) 표본이 적거나 불확실하면 '추가 검증 필요' 명시
5) 한국어, 15줄 이내
"""

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        return obj["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
        return None


def build_payloads(results: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    """규칙 기반 결과를 AI 입력용 텍스트로 압축."""
    payloads: dict[str, str] = {}

    overview_lines = [
        f"서비스: {context.get('service_name', '')}",
        f"서비스 설명: {context.get('service_description', '')}",
        f"설문 목적: {context.get('survey_purpose', '')}",
        f"가설: {context.get('hypotheses', '')}",
        f"타깃: {context.get('target_users', '')}",
    ]
    sr = context.get("service_research") or {}
    if sr.get("report"):
        payloads["서비스·경쟁 리서치"] = sr["report"][:4000]
    basic = results.get("basic", {})
    cleaning = results.get("cleaning_summary", {})
    overview_lines.extend(
        [
            f"응답 수: {basic.get('total_responses', 0)}",
            f"문항 수: {basic.get('total_questions', 0)}",
            f"분석 포함: {cleaning.get('included_count', 0)}",
            f"데이터 신뢰도: {cleaning.get('reliability', '')}",
        ]
    )
    payloads["개요·데이터 품질"] = "\n".join(overview_lines)

    quant_lines = []
    for item in results.get("quant_interp", []):
        quant_lines.append(item.get("text", ""))
    cross = results.get("quant_results", {}).get("_cross_analysis", [])
    for c in cross[:5]:
        quant_lines.append(c.get("interpretation", ""))
    payloads["정량·교차 분석"] = "\n".join(quant_lines) or "(정량 결과 없음)"

    qual_lines = [results.get("qual_results", {}).get("integrated", {}).get("summary", "")]
    for col, data in results.get("qual_results", {}).get("per_question", {}).items():
        qual_lines.append(f"[{col}] {data.get('summary', '')}")
        for rep in data.get("representative_responses", [])[:2]:
            qual_lines.append(f"  ID{rep['response_id']}: {rep['text'][:120]}")
    payloads["정성 분석"] = "\n".join(qual_lines)

    kw_df: pd.DataFrame = results.get("keyword_df")
    if kw_df is not None and not kw_df.empty:
        payloads["키워드·감성"] = kw_df.head(12).to_string(index=False)
    else:
        payloads["키워드·감성"] = "(키워드 없음)"

    hyp_df: pd.DataFrame = results.get("hypothesis_df")
    if hyp_df is not None and not hyp_df.empty:
        payloads["가설 검토"] = hyp_df.to_string(index=False)
    else:
        payloads["가설 검토"] = context.get("hypotheses", "")

    ins_lines = []
    for ins in results.get("insights", [])[:5]:
        ins_lines.append(
            f"- {ins.get('인사이트 제목')}: {ins.get('Fact', '')[:100]}"
        )
    payloads["규칙 기반 인사이트 초안"] = "\n".join(ins_lines) or "(없음)"

    bp = results.get("behavior_summary") or {}
    if bp.get("summary_text"):
        payloads["핵심 사용 행태"] = bp["summary_text"][:3500]
    if bp.get("ai_narrative"):
        payloads["핵심 사용 행태 (AI)"] = bp["ai_narrative"][:3500]

    return payloads


def run_ai_analysis(
    results: dict[str, Any],
    context: dict[str, Any],
    api_key: str,
    model: str = "gpt-4o-mini",
    enabled: bool = True,
) -> dict[str, str | None]:
    """
    섹션별 OpenAI 해석 실행.
    Returns: {섹션명: 해석 텍스트 또는 None}
    """
    if not enabled or not api_key:
        return {}

    payloads = build_payloads(results, context)
    interpretations: dict[str, str | None] = {}

    for section, payload in payloads.items():
        interpretations[section] = call_openai_interpretation(
            section, payload, api_key=api_key, model=model, enabled=True
        )

    # 최종 통합 인사이트
    combined = "\n\n".join(
        f"## {k}\n{v}" for k, v in interpretations.items() if v
    )
    if combined:
        interpretations["최종 AI 인사이트"] = call_openai_interpretation(
            "최종 인사이트 종합",
            combined + "\n\n" + payloads.get("규칙 기반 인사이트 초안", ""),
            api_key=api_key,
            model=model,
            enabled=True,
        )

    return interpretations
