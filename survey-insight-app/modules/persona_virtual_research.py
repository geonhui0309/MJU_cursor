"""HF persona DB 기반 가상 사용자 매칭·Virtual IDI."""

from __future__ import annotations

import json
import re
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROFILE_COLUMN_HINTS = {
    "age": ("연령", "나이", "age"),
    "gender": ("성별", "gender", "sex"),
    "region": ("지역", "거주", "주소", "city", "location"),
    "job": ("직업", "occupation", "job", "소속"),
    "marital": ("결혼", "혼인", "marital"),
}

PROFILE_WEIGHTS = {
    "age": 16.0,
    "gender": 10.0,
    "region": 16.0,
    "job": 18.0,
    "marital": 8.0,
}

PERSONA_TEXT_HINTS = (
    "persona",
    "profile",
    "description",
    "summary",
    "narrative",
    "professional",
    "lifestyle",
    "bio",
)

LABEL_HINTS = ("name", "persona_id", "id", "uid", "nickname")
MISSING_TEXTS = {"", "nan", "none", "null", "nat", "n/a", "na", "-", "--", "미응답", "무응답"}


def materialize_persona_db(
    file_bytes: bytes | None,
    output_dir: Path,
    filename: str = "personas.db",
) -> Path | None:
    """업로드된 DB를 로컬 경로로 저장."""
    if not file_bytes:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename or "personas.db").name
    if not safe_name.lower().endswith((".db", ".sqlite", ".sqlite3")):
        safe_name = "personas.db"
    path = output_dir / safe_name
    path.write_bytes(file_bytes)
    return path


def run_persona_research(
    db_path: str | Path | None,
    cleaned_df: pd.DataFrame,
    insights: list[dict[str, Any]],
    context: dict[str, Any],
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
    enabled: bool = True,
) -> dict[str, Any]:
    """페르소나 매칭과 가상 인터뷰/검증을 수행."""
    result: dict[str, Any] = {
        "enabled": enabled,
        "db_available": False,
        "db_path": str(db_path) if db_path else "",
        "db_meta": {},
        "segment_profile": {},
        "matched_personas": [],
        "virtual_idi": [],
        "validation": [],
        "generation_provider": "openai",
        "summary": "",
        "ai_used": False,
        "error": "",
    }
    if not enabled or not db_path:
        return result

    path = Path(db_path)
    if not path.exists():
        result["error"] = f"persona DB를 찾을 수 없습니다: {path.name}"
        result["summary"] = result["error"]
        return result

    result["db_available"] = True
    segment_profile = infer_segment_profile(cleaned_df, context)
    result["segment_profile"] = segment_profile

    try:
        matches, meta = match_personas(path, segment_profile)
        result["matched_personas"] = matches
        result["db_meta"] = meta
    except sqlite3.Error as exc:
        result["error"] = f"persona DB 조회 실패: {exc}"
        result["summary"] = result["error"]
        return result

    if not matches:
        result["summary"] = "페르소나 DB는 연결되었지만 현재 설문 세그먼트와 유사한 후보를 찾지 못했습니다."
        return result

    if openai_api_key:
        primary = generate_virtual_idi_and_validation(
            matches=matches[:3],
            insights=insights[:3],
            context=context,
            provider_label="openai",
            api_key=openai_api_key,
            model=openai_model,
            endpoint="https://api.openai.com/v1/chat/completions",
        )
        if primary:
            result["virtual_idi"] = primary.get("idi_sessions", [])
            result["validation"] = primary.get("validations", [])
            result["summary"] = primary.get("summary", "")
            result["ai_used"] = True

    if not result["summary"]:
        seg = segment_profile.get("summary", context.get("target_users", "전체"))
        result["summary"] = (
            f"설문 세그먼트 '{seg}' 기준으로 유사 페르소나 {len(matches)}명을 랭킹했습니다. "
            + ("OpenAI로 Virtual IDI/검증까지 생성했습니다." if result["ai_used"] else "DB 추천까지만 완료했습니다.")
        )
    return result


def infer_segment_profile(cleaned_df: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    """설문 응답에서 대표 세그먼트를 추정."""
    attrs: dict[str, str] = {}
    for attr, hints in PROFILE_COLUMN_HINTS.items():
        col = _find_matching_column(cleaned_df.columns, hints)
        if col:
            top_value = _top_value(cleaned_df[col])
            if top_value:
                attrs[attr] = top_value

    target_users = str(context.get("target_users", "") or "").strip()
    known_problems = str(context.get("known_problems", "") or "").strip()
    service_desc = str(context.get("service_description", "") or "").strip()
    tokens = _build_search_tokens(attrs, target_users, known_problems, service_desc)

    parts = [target_users] if target_users else []
    parts.extend(v for v in attrs.values() if v and v not in parts)
    summary = ", ".join(parts) if parts else "전체 응답자"
    query_text = _build_segment_query_text(summary, attrs, known_problems, service_desc)

    return {
        "target_users": target_users or "전체",
        "attributes": attrs,
        "known_problems": known_problems,
        "search_tokens": tokens,
        "summary": summary,
        "query_text": query_text,
    }


def match_personas(
    db_path: Path,
    segment_profile: dict[str, Any],
    top_n: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """속성 점수 + 텍스트 유사도 기반으로 유사 페르소나를 랭킹한다."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        table_meta = _choose_persona_table(conn)
        if not table_meta:
            return [], {}
        table_name = table_meta["table"]
        columns = table_meta["columns"]
        rows = _query_candidate_rows(conn, table_name, columns, segment_profile.get("search_tokens", []))
        if not rows:
            return [], {"table": table_name, "columns": columns, "row_count": table_meta["row_count"]}

        candidates = []
        texts = []
        for row in rows:
            item = dict(row)
            persona_text = _compose_persona_text(item, columns)
            attrs = _extract_row_attributes(item, columns)
            attr_score, attr_reasons = _attribute_match_score(attrs, segment_profile.get("attributes", {}))
            token_score, token_reasons = _token_match_score(persona_text, segment_profile.get("search_tokens", []))
            candidates.append(
                {
                    "row": item,
                    "persona_text": persona_text,
                    "attributes": attrs,
                    "attr_score": attr_score,
                    "token_score": token_score,
                    "attr_reasons": attr_reasons,
                    "token_reasons": token_reasons,
                }
            )
            texts.append(persona_text or _row_as_text(item, columns))

        semantic_scores = _semantic_similarity_scores(segment_profile.get("query_text", ""), texts)
        matched = []
        for idx, cand in enumerate(candidates):
            semantic_score = semantic_scores[idx] if idx < len(semantic_scores) else 0.0
            composite = round(cand["attr_score"] + cand["token_score"] + semantic_score, 2)
            has_signal = (
                cand["attr_score"] > 0
                or cand["token_score"] >= 2.0
                or semantic_score >= 6.0
            )
            if composite <= 0 or not has_signal:
                continue
            row = cand["row"]
            match_reasons = cand["attr_reasons"] + cand["token_reasons"]
            match_reasons.append(f"semantic:{semantic_score:.1f}")
            matched.append(
                {
                    "persona_id": _persona_identifier(row, columns),
                    "label": _persona_label(row, columns),
                    "match_score": composite,
                    "attr_score": round(cand["attr_score"], 2),
                    "token_score": round(cand["token_score"], 2),
                    "semantic_score": round(semantic_score, 2),
                    "match_reasons": match_reasons[:6],
                    "profile_excerpt": cand["persona_text"][:500],
                    "persona_text": cand["persona_text"][:2500],
                    "attributes": cand["attributes"],
                }
            )

        matched.sort(
            key=lambda x: (x["match_score"], x["semantic_score"], x["attr_score"]),
            reverse=True,
        )
        meta = {
            "table": table_name,
            "columns": columns,
            "row_count": table_meta["row_count"],
            "ranking_method": "weighted-attributes + token-match + tfidf-cosine",
        }
        return matched[:top_n], meta
    finally:
        conn.close()


def generate_virtual_idi_and_validation(
    matches: list[dict[str, Any]],
    insights: list[dict[str, Any]],
    context: dict[str, Any],
    provider_label: str,
    api_key: str,
    model: str,
    endpoint: str,
) -> dict[str, Any] | None:
    """선택 페르소나로 Virtual IDI와 인사이트 검증을 수행."""
    if not matches or not insights:
        return None

    persona_block = []
    for idx, persona in enumerate(matches, 1):
        persona_block.append(
            f"[Persona {idx}] id={persona['persona_id']}\n"
            f"label={persona['label']}\n"
            f"match_score={persona['match_score']}\n"
            f"attr_score={persona.get('attr_score', 0)} token_score={persona.get('token_score', 0)} semantic_score={persona.get('semantic_score', 0)}\n"
            f"match_reasons={', '.join(persona.get('match_reasons', []))}\n"
            f"profile={persona['persona_text'][:1400]}"
        )
    insight_block = []
    for idx, ins in enumerate(insights, 1):
        insight_block.append(
            f"[Insight {idx}]\n"
            f"title={ins.get('인사이트 제목', '')}\n"
            f"problem={ins.get('사용자가 실제로 겪는 문제', '')}\n"
            f"fact={ins.get('Fact', '')}\n"
            f"interpretation={ins.get('Interpretation', '')}\n"
            f"action={ins.get('Action', '')}\n"
        )

    user_prompt = f"""당신은 UX Researcher입니다.

설문 요약:
- 서비스: {context.get('service_name', '')}
- 서비스 설명: {context.get('service_description', '')}
- 설문 목적: {context.get('survey_purpose', '')}
- 타깃: {context.get('target_users', '')}
- 알고 싶은 문제: {context.get('known_problems', '') or '-'}

유사 페르소나:
{chr(10).join(persona_block)}

검토할 인사이트:
{chr(10).join(insight_block)}

해야 할 일:
1) 각 persona마다 가장 중요한 insight 1개를 골라 짧은 Virtual IDI를 만드세요.
2) 각 insight를 여러 persona 관점에서 Supported / Partially Supported / Weakly Supported 중 하나로 평가하세요.
3) 점수는 0~100 정수로 주세요.
4) 설문 fact에 없는 내용은 persona 맥락에서 조심스럽게 추론하고, 과장하지 마세요.

반드시 아래 JSON만 반환하세요.
{{
  "summary": "전체 요약 2~4문장",
  "idi_sessions": [
    {{
      "persona_id": "string",
      "persona_label": "string",
      "insight_title": "string",
      "qa": [
        {{"question": "왜 이런 문제가 생기나요?", "answer": "string"}},
        {{"question": "어떤 상황에서 가장 불편한가요?", "answer": "string"}},
        {{"question": "현재 어떻게 우회하나요?", "answer": "string"}},
        {{"question": "무엇이 바뀌면 좋아질까요?", "answer": "string"}}
      ],
      "takeaway": "string"
    }}
  ],
  "validations": [
    {{
      "insight_title": "string",
      "overall_verdict": "Supported|Partially Supported|Weakly Supported",
      "overall_score": 0,
      "rationale": "string",
      "by_persona": [
        {{
          "persona_id": "string",
          "persona_label": "string",
          "verdict": "Supported|Partially Supported|Weakly Supported",
          "score": 0,
          "reason": "string"
        }}
      ]
    }}
  ]
}}"""

    obj = _call_chat_json(
        api_key=api_key,
        model=model,
        system_message=(
            "당신은 근거 중심 UX Research 보조 분석가입니다. "
            "persona 프로필을 일관되게 유지하되, 설문 fact를 넘어서는 단정은 피하세요."
        ),
        user_message=user_prompt,
        endpoint=endpoint,
    )
    if not obj:
        return None
    return {
        "provider": provider_label,
        "model": model,
        "summary": str(obj.get("summary", "")).strip(),
        "idi_sessions": obj.get("idi_sessions", []),
        "validations": obj.get("validations", []),
    }


def build_persona_exports(persona_result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """다운로드용 DataFrame 생성."""
    matches = pd.DataFrame(persona_result.get("matched_personas", []))
    primary_idi_rows = _flatten_idi_rows(persona_result.get("virtual_idi", []), "openai")
    primary_validation_rows = _flatten_validation_rows(persona_result.get("validation", []), "openai")

    return {
        "matches": matches,
        "idi": pd.DataFrame(primary_idi_rows),
        "validation": pd.DataFrame(primary_validation_rows),
    }


def _find_matching_column(columns, hints: tuple[str, ...]) -> str | None:
    lowered = {str(col).lower(): str(col) for col in columns}
    for hint in hints:
        hint_lower = hint.lower()
        for key, original in lowered.items():
            if hint_lower in key:
                return original
    return None


def _top_value(series: pd.Series) -> str:
    valid = series.dropna().map(_clean_text).astype(str)
    valid = valid[valid != ""]
    if valid.empty:
        return ""
    return str(valid.value_counts().index[0]).strip()


def _build_search_tokens(
    attrs: dict[str, str],
    target_users: str,
    known_problems: str,
    service_desc: str,
) -> list[str]:
    tokens = []
    for value in attrs.values():
        tokens.extend(_tokenize(value))
        tokens.append(value)
    tokens.extend(_tokenize(target_users))
    tokens.extend(_tokenize(known_problems))
    tokens.extend(_tokenize(service_desc))

    seen = set()
    output = []
    for token in tokens:
        tok = _clean_text(token).lower()
        if len(tok) < 2 or tok in seen:
            continue
        seen.add(tok)
        output.append(tok)
    return output[:20]


def _build_segment_query_text(
    summary: str,
    attrs: dict[str, str],
    known_problems: str,
    service_desc: str,
) -> str:
    parts = [summary]
    parts.extend(f"{key}:{value}" for key, value in attrs.items())
    if known_problems:
        parts.append(known_problems)
    if service_desc:
        parts.append(service_desc)
    return "\n".join(parts)


def _tokenize(text: str) -> list[str]:
    cleaned = _clean_text(text)
    return re.findall(r"[가-힣A-Za-z0-9]{2,}", cleaned)


def _choose_persona_table(conn: sqlite3.Connection) -> dict[str, Any] | None:
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    best = None
    best_score = -1
    for row in tables:
        table = row[0]
        cols_info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        columns = [info[1] for info in cols_info]
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        score = 0
        score += sum(2 for col in columns if any(h in col.lower() for h in PERSONA_TEXT_HINTS))
        score += sum(1 for col in columns if any(h in col.lower() for h in LABEL_HINTS))
        score += min(row_count, 1000) / 1000
        if score > best_score:
            best_score = score
            best = {"table": table, "columns": columns, "row_count": row_count}
    return best


def _query_candidate_rows(
    conn: sqlite3.Connection,
    table_name: str,
    columns: list[str],
    tokens: list[str],
    limit: int = 250,
) -> list[sqlite3.Row]:
    text_cols = columns[:]
    if not text_cols:
        return []

    predicates = []
    params: list[str] = []
    for token in tokens[:12]:
        token_params = []
        for col in text_cols[:12]:
            token_params.append(f'LOWER(CAST("{col}" AS TEXT)) LIKE ?')
            params.append(f"%{token}%")
        if token_params:
            predicates.append("(" + " OR ".join(token_params) + ")")

    query = f'SELECT * FROM "{table_name}"'
    if predicates:
        query += " WHERE " + " OR ".join(predicates)
    query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    if rows:
        return rows
    return conn.execute(f'SELECT * FROM "{table_name}" LIMIT {int(limit)}').fetchall()


def _attribute_match_score(
    persona_attrs: dict[str, str],
    segment_attrs: dict[str, str],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    for key, target in segment_attrs.items():
        if not target:
            continue
            candidate = str(persona_attrs.get(key, "") or "").strip()
        candidate = _clean_text(candidate)
        if not candidate:
            continue
        if target.lower() == candidate.lower():
            score += PROFILE_WEIGHTS.get(key, 8.0)
            reasons.append(f"{key}=exact")
        elif target.lower() in candidate.lower() or candidate.lower() in target.lower():
            score += PROFILE_WEIGHTS.get(key, 8.0) * 0.65
            reasons.append(f"{key}=partial")
    return score, reasons


def _token_match_score(persona_text: str, tokens: list[str]) -> tuple[float, list[str]]:
    text = _clean_text(persona_text).lower()
    score = 0.0
    reasons = []
    for token in tokens:
        if token in text:
            score += 1.2 if len(token) >= 4 else 0.8
            if len(reasons) < 3:
                reasons.append(f"token:{token}")
    return score, reasons


def _semantic_similarity_scores(query_text: str, texts: list[str]) -> list[float]:
    if not query_text.strip() or not texts:
        return [0.0 for _ in texts]
    corpus = [query_text] + [t if t.strip() else "(empty persona)" for t in texts]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return [round(float(sim) * 35.0, 2) for sim in sims]
    except ValueError:
        return [0.0 for _ in texts]


def _compose_persona_text(row: dict[str, Any], columns: list[str]) -> str:
    persona_fields = [col for col in columns if any(h in col.lower() for h in PERSONA_TEXT_HINTS)]
    if persona_fields:
        values = [_clean_text(row.get(col, "")) for col in persona_fields]
        return "\n".join(v for v in values if v)
    values = []
    for col in columns[:12]:
        val = _clean_text(row.get(col, ""))
        if val:
            values.append(f"{col}: {val}")
    return "\n".join(values)


def _row_as_text(row: dict[str, Any], columns: list[str]) -> str:
    return "\n".join(
        f"{col}: {_clean_text(row.get(col, ''))}"
        for col in columns[:12]
        if _clean_text(row.get(col, ""))
    )


def _persona_identifier(row: dict[str, Any], columns: list[str]) -> str:
    for hint in LABEL_HINTS:
        for col in columns:
            if hint == col.lower() or hint in col.lower():
                value = _clean_text(row.get(col, ""))
                if value:
                    return value
    return f"persona-{abs(hash(tuple(str(row.get(c, '')) for c in columns[:5])))}"


def _persona_label(row: dict[str, Any], columns: list[str]) -> str:
    parts = []
    for attr_cols in PROFILE_COLUMN_HINTS.values():
        col = _find_matching_column(columns, attr_cols)
        if not col:
            continue
        value = _clean_text(row.get(col, ""))
        if value:
            parts.append(value)
    if parts:
        return " / ".join(parts[:4])
    for col in columns:
        value = _clean_text(row.get(col, ""))
        if value:
            return f"{col}: {value[:40]}"
    return "Unnamed persona"


def _extract_row_attributes(row: dict[str, Any], columns: list[str]) -> dict[str, str]:
    attrs = {}
    for key, hints in PROFILE_COLUMN_HINTS.items():
        col = _find_matching_column(columns, hints)
        if not col:
            continue
        value = _clean_text(row.get(col, ""))
        if value:
            attrs[key] = value
    return attrs


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in MISSING_TEXTS:
        return ""
    return text


def _flatten_idi_rows(idi_sessions: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    rows = []
    for session in idi_sessions:
        for qa in session.get("qa", []):
            rows.append(
                {
                    "provider": provider,
                    "persona_id": session.get("persona_id", ""),
                    "persona_label": session.get("persona_label", ""),
                    "insight_title": session.get("insight_title", ""),
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "takeaway": session.get("takeaway", ""),
                }
            )
    return rows


def _flatten_validation_rows(validations: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    rows = []
    for item in validations:
        by_persona = item.get("by_persona", [])
        if not by_persona:
            rows.append(
                {
                    "provider": provider,
                    "insight_title": item.get("insight_title", ""),
                    "overall_verdict": item.get("overall_verdict", ""),
                    "overall_score": item.get("overall_score", ""),
                    "rationale": item.get("rationale", ""),
                }
            )
            continue
        for person in by_persona:
            rows.append(
                {
                    "provider": provider,
                    "insight_title": item.get("insight_title", ""),
                    "overall_verdict": item.get("overall_verdict", ""),
                    "overall_score": item.get("overall_score", ""),
                    "persona_id": person.get("persona_id", ""),
                    "persona_label": person.get("persona_label", ""),
                    "verdict": person.get("verdict", ""),
                    "score": person.get("score", ""),
                    "reason": person.get("reason", ""),
                    "rationale": item.get("rationale", ""),
                }
            )
    return rows


def _call_chat_json(
    api_key: str,
    model: str,
    system_message: str,
    user_message: str,
    endpoint: str,
) -> dict[str, Any] | None:
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url=endpoint,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        content = obj["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return json.loads(content)
        return content
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
        return None
