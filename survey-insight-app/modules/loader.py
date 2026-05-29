"""CSV 로드 및 기본 통계 추출."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd


ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1")


def load_csv(file_bytes: bytes | None = None, file_path: str | None = None) -> pd.DataFrame:
    """Google Forms CSV를 인코딩 후보 순으로 로드."""
    if file_bytes is None and file_path is None:
        raise ValueError("file_bytes 또는 file_path 중 하나는 필요합니다.")

    last_error: Exception | None = None
    for enc in ENCODINGS:
        try:
            if file_bytes is not None:
                buffer = io.BytesIO(file_bytes)
                df = pd.read_csv(buffer, encoding=enc)
            else:
                df = pd.read_csv(file_path, encoding=enc)
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(
        f"CSV 파일을 읽을 수 없습니다. 인코딩을 확인해 주세요. ({last_error})"
    )


def get_basic_stats(df: pd.DataFrame) -> dict[str, Any]:
    """전체 응답·문항·컬럼 기본 통계."""
    stats: dict[str, Any] = {
        "total_responses": len(df),
        "total_questions": len(df.columns),
        "columns": list(df.columns),
        "has_timestamp": _detect_timestamp_column(df) is not None,
        "timestamp_column": _detect_timestamp_column(df),
        "per_column": {},
    }
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        non_empty = non_null.astype(str).str.strip()
        non_empty = non_empty[non_empty != ""]
        stats["per_column"][col] = {
            "response_count": int(non_empty.shape[0]),
            "missing_count": int(len(df) - non_empty.shape[0]),
            "unique_count": int(non_empty.nunique()),
        }
    return stats


def _detect_timestamp_column(df: pd.DataFrame) -> str | None:
    """타임스탬프 컬럼 후보 탐지."""
    keywords = ("timestamp", "타임스탬프", "제출", "시간", "date", "날짜")
    for col in df.columns:
        lower = str(col).lower()
        if any(k in lower for k in keywords):
            return col
    return None


def add_response_ids(df: pd.DataFrame) -> pd.DataFrame:
    """분석용 response_id 컬럼 추가 (1-based)."""
    out = df.copy()
    out.insert(0, "response_id", range(1, len(out) + 1))
    return out
