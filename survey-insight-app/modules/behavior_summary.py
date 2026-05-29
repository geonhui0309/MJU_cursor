"""정성·정량 통합 — 전체 응답자 핵심 사용 행태 요약."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

# 행태 유형 분류
BEHAVIOR_TYPES = (
    "사용 경험·빈도",
    "의사결정·선택",
    "만족·평가",
    "불편·이탈",
    "재사용·충성",
    "탐색·정보 이용",
    "기대·요구",
)

BEHAVIOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "사용 경험·빈도": ("사용", "이용", "경험", "써", "해봤"),
    "의사결정·선택": ("선택", "비교", "고르", "결정", "기준"),
    "만족·평가": ("만족", "좋", "괜찮", "평가", "점수"),
    "불편·이탈": ("불편", "어렵", "복잡", "이탈", "안 쓸"),
    "재사용·충성": ("다시", "재사용", "의향", "추천", "계속"),
    "탐색·정보 이용": ("찾", "검색", "정보", "알림", "메뉴"),
    "기대·요구": ("있으면", "원해", "기대", "추가", "개선"),
}


def summarize_usage_behaviors(
    quant_results: dict[str, Any],
    qual_results: dict[str, Any],
    keyword_df: pd.DataFrame | None,
    journey_df: pd.DataFrame | None,
    sentiment_df: pd.DataFrame | None,
    schema_df: pd.DataFrame | None,
    cleaning_summary: dict[str, Any],
    total_responses: int,
) -> dict[str, Any]:
    """
    규칙 기반 핵심 사용 행태 통합.
    Returns: {behaviors: list[dict], summary_text, export_df}
    """
    behaviors: list[dict] = []
    behaviors.extend(_from_quantitative(quant_results, total_responses))
    behaviors.extend(_from_qualitative(qual_results, total_responses))
    behaviors.extend(_from_keywords(keyword_df, total_responses))
    behaviors.extend(_from_journey(journey_df, total_responses))
    behaviors.extend(_from_sentiment(sentiment_df, total_responses))

    # 유형별 병합·중복 제거
    merged = _merge_behaviors(behaviors)
    merged = sorted(merged, key=lambda x: (-x.get("score", 0), x.get("행태 유형", "")))[:10]

    summary_text = _build_summary_narrative(merged, total_responses, cleaning_summary)
    export_df = pd.DataFrame(merged) if merged else pd.DataFrame()

    return {
        "behaviors": merged,
        "summary_text": summary_text,
        "export_df": export_df,
        "total_responses": total_responses,
    }


def _from_quantitative(quant_results: dict, total: int) -> list[dict]:
    items: list[dict] = []
    for col, res in quant_results.items():
        if col.startswith("_") or not isinstance(res, dict):
            continue
        qtype = res.get("type", "")
        col_lower = str(col).lower()

        if "distribution" in res and res["distribution"]:
            top = res["distribution"][0]
            pct = top.get("ratio", 0)
            choice = top.get("choice", "")
            btype = _infer_behavior_type(str(col) + " " + str(choice))
            items.append(
                {
                    "행태 유형": btype,
                    "핵심 행태": f"「{col}」에서 '{choice}' 응답이 가장 많음 ({pct}%)",
                    "정량 근거": f"{pct}% (n≈{total})",
                    "정성 근거": "-",
                    "관련 문항": col,
                    "여정 단계": _journey_from_text(col),
                    "score": pct,
                }
            )

        if qtype == "리커트 척도형":
            mean_v = res.get("mean", 0)
            pos = res.get("positive_pct", 0)
            neg = res.get("negative_pct", 0)
            tone = "긍정적" if mean_v >= 3.5 else ("부정적" if mean_v < 2.5 else "중립적")
            btype = "만족·평가" if "만족" in col or "평가" in col else _infer_behavior_type(col)
            items.append(
                {
                    "행태 유형": btype,
                    "핵심 행태": f"「{col}」 평가는 평균 {mean_v}점으로 전체적으로 {tone} 경향",
                    "정량 근거": f"평균 {mean_v}, 긍정 {pos}%, 부정 {neg}%",
                    "정성 근거": "-",
                    "관련 문항": col,
                    "여정 단계": _journey_from_text(col),
                    "score": abs(mean_v - 3) * 20 + pos,
                }
            )

        # 재사용·경험 문항 키워드 매칭
        if any(k in col for k in ("재사용", "의향", "다시", "경험", "사용해본", "이용")):
            if "distribution" in res and len(res["distribution"]) >= 2:
                d = res["distribution"]
                items.append(
                    {
                        "행태 유형": "재사용·충성" if "재사용" in col or "의향" in col else "사용 경험·빈도",
                        "핵심 행태": f"「{col}」 응답이 {d[0].get('choice')}({d[0].get('ratio')}%) vs {d[1].get('choice')}({d[1].get('ratio')}%)로 분화",
                        "정량 근거": " / ".join(f"{x.get('choice')} {x.get('ratio')}%" for x in d[:3]),
                        "정성 근거": "-",
                        "관련 문항": col,
                        "여정 단계": "재사용" if "재사용" in col else "진입",
                        "score": 50,
                    }
                )
    return items


def _from_qualitative(qual_results: dict, total: int) -> list[dict]:
    items: list[dict] = []
    integrated = qual_results.get("integrated", {})
    for prob in integrated.get("top_problems", [])[:4]:
        kw = prob.get("keyword", "")
        cnt = prob.get("count", 0)
        items.append(
            {
                "행태 유형": _infer_behavior_type(kw),
                "핵심 행태": f"주관식에서 '{kw}' 관련 불편·문제가 반복 언급됨",
                "정량 근거": "-",
                "정성 근거": f"언급 {cnt}회 (전체 {total}명 중)",
                "관련 문항": "주관식 통합",
                "여정 단계": _journey_from_text(kw),
                "score": cnt * 8,
            }
        )
    for imp in integrated.get("top_improvements", [])[:3]:
        kw = imp.get("keyword", "")
        cnt = imp.get("count", 0)
        items.append(
            {
                "행태 유형": "기대·요구",
                "핵심 행태": f"개선·기대 키워드 '{kw}'가 다수 응답에 등장",
                "정량 근거": "-",
                "정성 근거": f"언급 {cnt}회",
                "관련 문항": "주관식 통합",
                "여정 단계": "탐색",
                "score": cnt * 6,
            }
        )
    return items


def _from_keywords(keyword_df: pd.DataFrame | None, total: int) -> list[dict]:
    if keyword_df is None or keyword_df.empty:
        return []
    items = []
    for _, row in keyword_df.head(6).iterrows():
        kw = row.get("키워드", "")
        freq = row.get("빈도", 0)
        items.append(
            {
                "행태 유형": _infer_behavior_type(str(kw) + " " + str(row.get("주요 맥락", ""))),
                "핵심 행태": f"응답 텍스트에서 '{kw}'가 핵심 맥락어로 반복 ({row.get('감성 방향', '')})",
                "정량 근거": f"빈도 {freq}회",
                "정성 근거": str(row.get("대표 응답", ""))[:100],
                "관련 문항": row.get("관련 문항", ""),
                "여정 단계": _journey_from_text(str(kw)),
                "score": int(freq) * 5,
            }
        )
    return items


def _from_journey(journey_df: pd.DataFrame | None, total: int) -> list[dict]:
    if journey_df is None or journey_df.empty:
        return []
    items = []
    for _, row in journey_df.sort_values("응답 수", ascending=False).head(4).iterrows():
        stage = row.get("여정 단계", "")
        cnt = row.get("응답 수", 0)
        pct = round(cnt / max(total, 1) * 100, 1)
        items.append(
            {
                "행태 유형": "탐색·정보 이용" if stage in ("탐색", "선택") else _infer_behavior_type(str(stage)),
                "핵심 행태": f"여정 「{stage}」 단계 경험·언급이 두드러짐 ({pct}% 수준)",
                "정량 근거": f"매핑 응답 {cnt}건",
                "정성 근거": str(row.get("주요 문제", ""))[:80],
                "관련 문항": "주관식·여정",
                "여정 단계": stage,
                "score": cnt * 4,
            }
        )
    return items


def _from_sentiment(sentiment_df: pd.DataFrame | None, total: int) -> list[dict]:
    if sentiment_df is None or sentiment_df.empty:
        return []
    items = []
    col = "감성" if "감성" in sentiment_df.columns else None
    cnt_col = "건수" if "건수" in sentiment_df.columns else None
    if col and cnt_col:
        for _, row in sentiment_df.head(4).iterrows():
            emo = row[col]
            cnt = row[cnt_col]
            items.append(
                {
                    "행태 유형": _emotion_to_behavior(str(emo)),
                    "핵심 행태": f"전체 응답자 감성 분포에서 「{emo}」 성향이 {cnt}건 관찰",
                    "정량 근거": f"{cnt}건",
                    "정성 근거": str(row.get("대표 응답", ""))[:80] if "대표 응답" in sentiment_df.columns else "-",
                    "관련 문항": "주관식",
                    "여정 단계": "-",
                    "score": int(cnt) * 3,
                }
            )
    return items


def _merge_behaviors(behaviors: list[dict]) -> list[dict]:
    """유사 행태 유형 병합."""
    by_type: dict[str, dict] = {}
    for b in behaviors:
        key = b.get("행태 유형", "기타")
        if key not in by_type:
            by_type[key] = {**b, "score": b.get("score", 0)}
        else:
            existing = by_type[key]
            existing["score"] = existing.get("score", 0) + b.get("score", 0)
            if b.get("정성 근거") != "-" and existing.get("정성 근거") == "-":
                existing["정성 근거"] = b["정성 근거"]
            if len(str(b.get("핵심 행태", ""))) > len(str(existing.get("핵심 행태", ""))):
                existing["핵심 행태"] = b["핵심 행태"]
    return list(by_type.values())


def _build_summary_narrative(merged: list[dict], total: int, cleaning: dict) -> str:
    if not merged:
        return f"전체 {total}명 응답 기준으로 통합 행태를 도출하기 어렵습니다. 표본·문항 구성을 확인하세요."
    lines = [
        f"전체 응답자 {total}명(분석 포함 {cleaning.get('included_count', total)}명) 기준, "
        f"정량·정성 데이터를 통합한 핵심 사용 행태입니다.",
        "",
    ]
    for i, b in enumerate(merged[:7], 1):
        lines.append(f"{i}. [{b.get('행태 유형')}] {b.get('핵심 행태')}")
        q = b.get("정량 근거", "-")
        ql = b.get("정성 근거", "-")
        if q != "-" or ql != "-":
            lines.append(f"   - 정량: {q} | 정성: {ql}")
    return "\n".join(lines)


def _infer_behavior_type(text: str) -> str:
    for btype, kws in BEHAVIOR_KEYWORDS.items():
        if any(k in text for k in kws):
            return btype
    return "탐색·정보 이용"


def _journey_from_text(text: str) -> str:
    mapping = {
        "인지": ("인지", "처음", "알게"),
        "진입": ("가입", "시작", "진입"),
        "탐색": ("찾", "검색", "탐색"),
        "선택": ("선택", "비교", "고르"),
        "실행": ("예약", "신청", "결제"),
        "재사용": ("다시", "재사용", "의향"),
        "이탈": ("이탈", "안 쓸"),
    }
    for stage, kws in mapping.items():
        if any(k in text for k in kws):
            return stage
    return "-"


def _emotion_to_behavior(emo: str) -> str:
    if emo in ("불만", "혼란", "피로", "실망", "의심"):
        return "불편·이탈"
    if emo in ("만족", "기대"):
        return "만족·평가" if emo == "만족" else "기대·요구"
    return "만족·평가"


def enhance_behaviors_with_ai(
    behavior_pack: dict[str, Any],
    context: dict[str, Any],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str | None:
    """AI로 통합 행태 서술 보강."""
    if not api_key or not behavior_pack.get("behaviors"):
        return None

    payload = behavior_pack.get("summary_text", "") + "\n\n"
    for b in behavior_pack["behaviors"][:8]:
        payload += json.dumps(b, ensure_ascii=False) + "\n"

    user_msg = f"""다음은 설문 CSV 전체 응답자({behavior_pack.get('total_responses')}명)의 정량·정성 통합 행태 초안입니다.

서비스: {context.get('service_name')}
설문 목적: {context.get('survey_purpose')}

{payload}

## 작성 요청
「전체 응답자의 핵심 사용 행태」를 UX 리서치 리포트 형식으로 12~20줄 작성하세요.
- 4~6개 행태 패턴으로 구조화 (번호 목록)
- 각 패턴: 행태 한 줄 요약 → 정량 근거 → 정성 근거 → 여정 단계
- 데이터에 없는 내용 추측 금지
- 한국어"""

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "UX Researcher. Fact-based only."},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.25,
    }
    req = urllib.request.Request(
        url="https://api.openai.com/v1/chat/completions",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
        return None
