"""응답 데이터 정제 및 cleaning_log 생성."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

MEANINGLESS = {"", "없음", "없어요", "없습니다", "해당없음", "해당 없음", "모름", "잘 모름", "-", ".", "n/a", "na", "무"}
SHORT_THRESHOLD = 5
PII_PHONE = re.compile(r"01[0-9][-\s]?\d{3,4}[-\s]?\d{4}")
PII_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
SPAM_KEYWORDS = ("광고", "홍보", "제휴", "문의주세요", "카톡", "텔레그램", "수익", "대출")


def clean_data(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
    text_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    데이터 정제 수행.
    Returns: (cleaned_df, cleaning_log_df, summary_dict)
    """
    text_columns = text_columns or _get_text_columns(schema_df)
    logs: list[dict[str, Any]] = []
    cleaned = df.copy()

    # 중복 응답 탐지 (텍스트 컬럼 합친 해시)
    dup_ids = _find_duplicate_response_ids(cleaned, text_columns)
    for rid in dup_ids:
        logs.append(
            _log_row(rid, "(전체)", "", "중복 응답", "동일/유사 응답 패턴", "분석 제외", False)
        )

    for col in text_columns:
        if col not in cleaned.columns:
            continue
        for idx, val in cleaned[col].items():
            rid = int(cleaned.loc[idx, "response_id"]) if "response_id" in cleaned.columns else idx + 1
            if rid in dup_ids:
                continue
            text = str(val).strip() if pd.notna(val) else ""
            if not text:
                continue

            entries = _evaluate_cell(rid, col, text)
            for entry in entries:
                logs.append(entry)
                if entry["action"] == "마스킹 후 포함" and entry["included_in_analysis"]:
                    masked = _mask_pii(text)
                    cleaned.at[idx, col] = masked

    log_df = pd.DataFrame(logs) if logs else pd.DataFrame(
        columns=[
            "response_id", "question", "original_answer",
            "cleaning_label", "reason", "action", "included_in_analysis",
        ]
    )

    summary = _build_summary(df, cleaned, log_df, dup_ids)
    return cleaned, log_df, summary


def _get_text_columns(schema_df: pd.DataFrame) -> list[str]:
    types = ("단답형", "장문형", "개인정보 가능성")
    return schema_df.loc[schema_df["추정 유형"].isin(types), "문항명"].tolist()


def _evaluate_cell(rid: int, col: str, text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    lower = text.lower().strip()

    if lower in MEANINGLESS or lower in ("없음.", "특별히 없음"):
        entries.append(_log_row(rid, col, text, "무의미 응답", "의미 있는 UX 피드백이 없음", "분석 제외", False))
        return entries

    if len(text) < SHORT_THRESHOLD:
        entries.append(
            _log_row(rid, col, text, "짧은 응답", "의미는 있으나 맥락이 부족함", "제한적 분석 포함", True)
        )

    if PII_PHONE.search(text) or PII_EMAIL.search(text):
        entries.append(
            _log_row(rid, col, text, "개인정보 포함", "연락처/이메일 포함", "마스킹 후 포함", True)
        )

    if any(k in text for k in SPAM_KEYWORDS):
        entries.append(_log_row(rid, col, text, "스팸 또는 홍보성 응답", "홍보성 키워드", "분석 제외", False))

    if _is_contradictory(text):
        entries.append(
            _log_row(rid, col, text, "모순 응답", "긍정·부정 표현이 공존", "검토 필요", True)
        )

    if re.match(r"^[\W\d]+$", text):
        entries.append(_log_row(rid, col, text, "해석 불가능 응답", "의미 파악 어려움", "분석 제외", False))

    return entries


def _log_row(
    rid: int, question: str, answer: str, label: str, reason: str, action: str, included: bool
) -> dict[str, Any]:
    return {
        "response_id": rid,
        "question": question,
        "original_answer": answer,
        "cleaning_label": label,
        "reason": reason,
        "action": action,
        "included_in_analysis": included,
    }


def _mask_pii(text: str) -> str:
    text = PII_PHONE.sub("[전화번호 마스킹]", text)
    text = PII_EMAIL.sub("[이메일 마스킹]", text)
    return text


def _is_contradictory(text: str) -> bool:
    pos = ("좋", "만족", "편리", "추천", "괜찮")
    neg = ("안 쓸", "다시 쓰지", "불편", "어렵", "복잡", "실망")
    has_pos = any(p in text for p in pos)
    has_neg = any(n in text for n in neg)
    reuse_neg = "다시" in text and ("않" in text or "안" in text)
    return (has_pos and has_neg) or reuse_neg


def _find_duplicate_response_ids(df: pd.DataFrame, text_cols: list[str]) -> set[int]:
    if not text_cols:
        return set()
    subset = df[text_cols].fillna("").astype(str).agg(" | ".join, axis=1)
    dup_mask = subset.duplicated(keep="first")
    if "response_id" in df.columns:
        return set(df.loc[dup_mask, "response_id"].astype(int).tolist())
    return set(df.index[dup_mask].tolist())


def _build_summary(
    original: pd.DataFrame, cleaned: pd.DataFrame, log_df: pd.DataFrame, dup_ids: set[int]
) -> dict[str, Any]:
    total = len(original)
    excluded = set()
    review = set()
    masked = 0

    if not log_df.empty:
        excluded = set(
            log_df.loc[log_df["included_in_analysis"] == False, "response_id"].astype(int)  # noqa: E712
        )
        review = set(
            log_df.loc[log_df["action"] == "검토 필요", "response_id"].astype(int)
        )
        masked = int((log_df["action"] == "마스킹 후 포함").sum())

    excluded |= dup_ids
    included_count = total - len(excluded)

    label_counts = {}
    if not log_df.empty:
        label_counts = log_df["cleaning_label"].value_counts().to_dict()

    reliability = _assess_reliability(total, included_count, len(review), log_df)

    return {
        "original_count": total,
        "included_count": max(included_count, 0),
        "excluded_count": len(excluded),
        "review_count": len(review),
        "masked_count": masked,
        "label_counts": label_counts,
        "reliability": reliability,
        "excluded_ids": sorted(excluded),
        "review_ids": sorted(review),
    }


def _assess_reliability(total: int, included: int, review: int, log_df: pd.DataFrame) -> str:
    if total == 0:
        return "추가 검토 필요"
    ratio = included / total
    issue_ratio = len(log_df) / max(total * 3, 1) if not log_df.empty else 0
    if ratio >= 0.85 and review <= total * 0.1:
        return "높음"
    if ratio >= 0.7:
        return "보통"
    if ratio >= 0.5:
        return "낮음"
    return "추가 검토 필요"
