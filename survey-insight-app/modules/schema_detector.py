"""문항 유형 자동 분류."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

QUESTION_TYPES = [
    "타임스탬프",
    "단일 선택형",
    "복수 선택형",
    "리커트 척도형",
    "숫자형",
    "단답형",
    "장문형",
    "개인정보 가능성",
    "분석 제외 후보",
]

PII_KEYWORDS = ("이메일", "전화", "연락", "주소", "이름", "휴대", "핸드폰", "email", "phone")
EXCLUDE_KEYWORDS = ("동의", "개인정보", "연락처", "이메일 주소")
LIKERT_KEYWORDS = ("만족", "동의", "평가", "점수", "정도", "얼마나", "1~5", "1-5", "리커트")


def detect_question_types(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
    focus_columns: list[str] | None = None,
) -> pd.DataFrame:
    """각 컬럼의 문항 유형·분석 방식·가설 연결성 추정."""
    exclude_columns = exclude_columns or []
    rows: list[dict[str, Any]] = []

    for col in df.columns:
        if col == "response_id":
            continue
        if focus_columns and col not in focus_columns and col not in exclude_columns:
            # focus 지정 시 focus 외는 스킵하지 않고 분석은 가능하게 유지
            pass

        series = df[col].dropna().astype(str).str.strip()
        series = series[series != ""]
        qtype = _classify_column(col, series, df[col])
        analysis_method = _analysis_method(qtype)
        hypothesis_link = _hypothesis_link(col, qtype)
        bias = _bias_note(qtype, series)
        caution = _caution_note(qtype, series)

        rows.append(
            {
                "문항명": col,
                "추정 유형": qtype,
                "응답 수": int(series.shape[0]),
                "누락 수": int(len(df) - series.shape[0]),
                "고유 응답 수": int(series.nunique()) if len(series) else 0,
                "분석 방식": analysis_method,
                "가설 연결성": hypothesis_link,
                "편향 가능성": bias,
                "주의사항": caution,
                "분석 제외": col in exclude_columns,
            }
        )

    return pd.DataFrame(rows)


def get_columns_by_type(schema_df: pd.DataFrame, qtype: str) -> list[str]:
    """특정 유형 문항 컬럼명 목록."""
    mask = schema_df["추정 유형"] == qtype
    return schema_df.loc[mask, "문항명"].tolist()


def _classify_column(col_name: str, series: pd.Series, raw: pd.Series) -> str:
    name_lower = str(col_name).lower()

    if any(k in name_lower for k in ("timestamp", "타임스탬프", "제출 시각")):
        return "타임스탬프"

    if any(k in col_name for k in EXCLUDE_KEYWORDS):
        return "분석 제외 후보"

    if any(k in col_name for k in PII_KEYWORDS):
        return "개인정보 가능성"

    if len(series) == 0:
        return "단답형"

    # 복수 선택: 쉼표/세미콜론 구분
    multi_sep = series.str.contains(r"[,;]", regex=True, na=False).mean()
    if multi_sep > 0.3:
        return "복수 선택형"

    # 숫자형
    numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
    if numeric_ratio > 0.85:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if len(vals) and vals.min() >= 1 and vals.max() <= 10 and vals.nunique() <= 10:
            if any(k in col_name for k in LIKERT_KEYWORDS) or (vals.max() <= 7 and vals.min() >= 1):
                return "리커트 척도형"
        return "숫자형"

    unique_ratio = series.nunique() / max(len(series), 1)
    avg_len = series.str.len().mean()

    if unique_ratio < 0.35 and series.nunique() <= 15:
        return "단일 선택형"

    if avg_len >= 40 or (avg_len >= 25 and unique_ratio > 0.5):
        return "장문형"

    if avg_len < 25:
        return "단답형"

    return "장문형"


def _analysis_method(qtype: str) -> str:
    mapping = {
        "타임스탬프": "시계열/응답 시점 참고",
        "단일 선택형": "분포 분석",
        "복수 선택형": "다중 응답 빈도 분석",
        "리커트 척도형": "평균/분산·긍정부정 비율",
        "숫자형": "기술 통계",
        "단답형": "키워드·정성 요약",
        "장문형": "정성/키워드/감성 분석",
        "개인정보 가능성": "마스킹 후 제한 분석",
        "분석 제외 후보": "분석 제외",
    }
    return mapping.get(qtype, "탐색적 분석")


def _hypothesis_link(col_name: str, qtype: str) -> str:
    if qtype in ("리커트 척도형", "장문형", "단답형"):
        return "높음"
    if qtype in ("단일 선택형", "복수 선택형"):
        return "중간"
    if qtype == "타임스탬프":
        return "보조 변수"
    return "낮음"


def _bias_note(qtype: str, series: pd.Series) -> str:
    if qtype == "리커트 척도형":
        return "척도 방향·중립 응답 편향 가능"
    if qtype in ("단답형", "장문형") and len(series) < 30:
        return "표본 수 적음"
    if qtype == "단일 선택형":
        return "선택지 편향 가능"
    return "낮음"


def _caution_note(qtype: str, series: pd.Series) -> str:
    if qtype == "장문형":
        short_ratio = (series.str.len() < 10).mean() if len(series) else 0
        if short_ratio > 0.3:
            return "짧은 응답 비율 높음"
        return "맥락 풍부·정제 필요"
    if qtype == "리커트 척도형":
        return "척도 방향 확인 필요"
    if qtype == "개인정보 가능성":
        return "마스킹 필수"
    if qtype == "분석 제외 후보":
        return "분석에서 제외 권장"
    return "-"
