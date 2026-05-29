"""정량 분석 및 교차 분석."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from modules.schema_detector import get_columns_by_type


def run_quantitative_analysis(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
    likert_directions: dict[str, bool] | None = None,
    excluded_ids: set[int] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, str]]]:
    """
    정량 분석 수행.
    Returns: (results_by_question, export_df, interpretations)
    """
    likert_directions = likert_directions or {}
    excluded_ids = excluded_ids or set()
    work = _filter_excluded(df, excluded_ids)

    qtypes = ("단일 선택형", "복수 선택형", "리커트 척도형", "숫자형")
    results: dict[str, Any] = {}
    export_rows: list[dict] = []
    interpretations: list[dict[str, str]] = []

    for qtype in qtypes:
        cols = get_columns_by_type(schema_df, qtype)
        for col in cols:
            if col not in work.columns:
                continue
            if qtype == "복수 선택형":
                res = _analyze_multi_select(work, col)
            elif qtype == "리커트 척도형":
                reverse = likert_directions.get(col, False)
                res = _analyze_likert(work, col, reverse)
            elif qtype == "숫자형":
                res = _analyze_numeric(work, col)
            else:
                res = _analyze_single_select(work, col)

            results[col] = res
            export_rows.extend(_flatten_for_export(col, res))
            if "interpretation" in res:
                interpretations.append({"question": col, "text": res["interpretation"]})

    cross_results, cross_interp = run_cross_analysis(work, schema_df)
    results["_cross_analysis"] = cross_results
    interpretations.extend(cross_interp)

    export_df = pd.DataFrame(export_rows) if export_rows else pd.DataFrame()
    return results, export_df, interpretations


def run_cross_analysis(
    df: pd.DataFrame, schema_df: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """핵심 문항 간 교차 분석."""
    likert_cols = get_columns_by_type(schema_df, "리커트 척도형")
    single_cols = get_columns_by_type(schema_df, "단일 선택형")
    cross_results: list[dict[str, Any]] = []
    interpretations: list[dict[str, str]] = []

    candidates: list[tuple[str, str]] = []
    if len(likert_cols) >= 2:
        candidates.extend(list(combinations(likert_cols[:4], 2)))
    if likert_cols and single_cols:
        for lcol in likert_cols[:2]:
            for scol in single_cols[:3]:
                if lcol != scol:
                    candidates.append((lcol, scol))

    seen = set()
    for col_a, col_b in candidates[:8]:
        key = tuple(sorted([col_a, col_b]))
        if key in seen:
            continue
        seen.add(key)
        cross = _cross_tab(df, col_a, col_b)
        if cross:
            cross_results.append(cross)
            if cross.get("interpretation"):
                interpretations.append(
                    {"question": f"{col_a} × {col_b}", "text": cross["interpretation"]}
                )

    return cross_results, interpretations


def _filter_excluded(df: pd.DataFrame, excluded_ids: set[int]) -> pd.DataFrame:
    if not excluded_ids or "response_id" not in df.columns:
        return df
    return df[~df["response_id"].isin(excluded_ids)].copy()


def _analyze_single_select(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = df[col].dropna().astype(str).str.strip()
    series = series[series != ""]
    counts = series.value_counts()
    total = len(series)
    dist = [
        {"choice": k, "count": int(v), "ratio": round(v / total * 100, 1) if total else 0}
        for k, v in counts.items()
    ]
    top = dist[0]["choice"] if dist else None
    bottom = dist[-1]["choice"] if dist else None
    interp = (
        f"'{col}' 문항에서 가장 많이 선택된 응답은 '{top}'({dist[0]['ratio']}%)입니다."
        if dist
        else f"'{col}' 문항에 유효 응답이 없습니다."
    )
    return {
        "type": "단일 선택형",
        "total": total,
        "distribution": dist,
        "top": top,
        "bottom": bottom,
        "interpretation": interp,
    }


def _analyze_multi_select(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = df[col].dropna().astype(str)
    all_choices: list[str] = []
    for val in series:
        parts = [p.strip() for p in re_split_multi(val) if p.strip()]
        all_choices.extend(parts)
    total_responses = len(series)
    if not all_choices:
        return {"type": "복수 선택형", "total": 0, "distribution": [], "interpretation": ""}
    counts = pd.Series(all_choices).value_counts()
    dist = [
        {
            "choice": k,
            "count": int(v),
            "ratio": round(v / total_responses * 100, 1) if total_responses else 0,
        }
        for k, v in counts.head(15).items()
    ]
    top = dist[0]["choice"] if dist else None
    interp = (
        f"'{col}'에서 '{top}'이(가) 가장 자주 언급되었습니다({dist[0]['ratio']}% 응답)."
        if dist
        else ""
    )
    return {
        "type": "복수 선택형",
        "total": total_responses,
        "distribution": dist,
        "top": top,
        "interpretation": interp,
    }


def _analyze_likert(df: pd.DataFrame, col: str, reverse: bool) -> dict[str, Any]:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return {"type": "리커트 척도형", "total": 0, "interpretation": ""}

    if reverse:
        max_v = series.max()
        series = max_v + 1 - series

    mean_v = float(series.mean())
    median_v = float(series.median())
    mode_v = float(series.mode().iloc[0]) if len(series.mode()) else mean_v
    std_v = float(series.std()) if len(series) > 1 else 0.0

    pos = ((series >= 4) & (series <= 5)).sum() / len(series) * 100
    neu = (series == 3).sum() / len(series) * 100
    neg = (series <= 2).sum() / len(series) * 100

    tone = "긍정" if mean_v >= 3.5 else ("부정" if mean_v < 2.5 else "중립")
    interp = (
        f"'{col}' 평균은 {mean_v:.1f}점으로 {tone}에 가깝습니다. "
        f"긍정 {pos:.0f}%, 중립 {neu:.0f}%, 부정 {neg:.0f}%입니다."
    )
    return {
        "type": "리커트 척도형",
        "total": int(len(series)),
        "mean": round(mean_v, 2),
        "median": median_v,
        "mode": mode_v,
        "std": round(std_v, 2),
        "positive_pct": round(pos, 1),
        "neutral_pct": round(neu, 1),
        "negative_pct": round(neg, 1),
        "interpretation": interp,
    }


def _analyze_numeric(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return {"type": "숫자형", "total": 0}
    return {
        "type": "숫자형",
        "total": int(len(series)),
        "mean": round(float(series.mean()), 2),
        "median": float(series.median()),
        "std": round(float(series.std()), 2) if len(series) > 1 else 0,
        "interpretation": f"'{col}' 평균값은 {series.mean():.2f}입니다.",
    }


def _cross_tab(df: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any] | None:
    a = df[col_a].dropna()
    b = df[col_b].dropna()
    common_idx = a.index.intersection(b.index)
    if len(common_idx) < 5:
        return None

    sub_a = pd.to_numeric(a.loc[common_idx], errors="coerce")
    sub_b = b.loc[common_idx].astype(str)

    if sub_a.notna().mean() > 0.7:
        # likert × category
        groups = sub_b.groupby(sub_b.index)
        group_means = {}
        for cat, idx in sub_b.groupby(sub_b).groups.items():
            vals = sub_a.loc[idx].dropna()
            if len(vals) >= 3:
                group_means[str(cat)] = round(float(vals.mean()), 2)
        if not group_means:
            return None
        best = max(group_means, key=group_means.get)
        worst = min(group_means, key=group_means.get)
        interp = (
            f"'{col_b}' 그룹별 '{col_a}' 평균 차이가 있습니다. "
            f"가장 높은 그룹: {best}({group_means[best]}점), "
            f"가장 낮은 그룹: {worst}({group_means[worst]}점)."
        )
        return {
            "col_a": col_a,
            "col_b": col_b,
            "group_means": group_means,
            "interpretation": interp,
        }
    return None


def re_split_multi(val: str) -> list[str]:
    import re

    return re.split(r"[,;、]", val)


def _flatten_for_export(col: str, res: dict[str, Any]) -> list[dict]:
    rows = []
    if "distribution" in res:
        for d in res["distribution"]:
            rows.append(
                {
                    "question": col,
                    "metric": "distribution",
                    "choice": d.get("choice"),
                    "value": d.get("count"),
                    "ratio_pct": d.get("ratio"),
                }
            )
    for key in ("mean", "median", "positive_pct", "negative_pct"):
        if key in res:
            rows.append({"question": col, "metric": key, "value": res[key]})
    return rows
