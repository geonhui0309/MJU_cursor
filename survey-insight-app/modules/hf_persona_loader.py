"""Hugging Face repo에서 persona DB를 자동 다운로드/변환."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

DB_EXTS = {".db", ".sqlite", ".sqlite3"}
TABULAR_EXTS = {".csv", ".json", ".jsonl", ".ndjson", ".parquet"}


def prepare_persona_db_from_hf(
    repo_id: str,
    output_dir: Path,
    token: str = "",
    filename: str = "",
    repo_type: str = "dataset",
    revision: str = "",
) -> tuple[Path | None, str]:
    """repo id로 HF에서 파일을 받아 personas.db를 준비한다."""
    repo_id = normalize_repo_id(repo_id)
    if not repo_id:
        return None, "HF repo id가 비어 있습니다."

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        return None, "`huggingface_hub`가 설치되지 않아 자동 다운로드를 실행할 수 없습니다."

    output_dir.mkdir(parents=True, exist_ok=True)
    target_db = output_dir / "personas.db"

    kwargs: dict[str, Any] = {"repo_id": repo_id, "repo_type": repo_type}
    if token:
        kwargs["token"] = token
    if revision.strip():
        kwargs["revision"] = revision.strip()

    try:
        if filename.strip():
            downloaded = Path(hf_hub_download(filename=filename.strip(), **kwargs))
            return _materialize_as_db(downloaded, target_db)

        snapshot_path = Path(
            snapshot_download(
                allow_patterns=[
                    "*.db",
                    "*.sqlite",
                    "*.sqlite3",
                    "*.csv",
                    "*.json",
                    "*.jsonl",
                    "*.ndjson",
                    "*.parquet",
                ],
                **kwargs,
            )
        )
        candidates = sorted(
            [p for p in snapshot_path.rglob("*") if p.is_file()],
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if not candidates:
            return None, f"{repo_id}에서 사용할 수 있는 파일을 찾지 못했습니다."

        db_candidate = next((p for p in candidates if p.suffix.lower() in DB_EXTS), None)
        if db_candidate:
            return _materialize_as_db(db_candidate, target_db)

        parquet_candidates = [p for p in candidates if p.suffix.lower() == ".parquet"]
        if parquet_candidates:
            return _materialize_many_as_db(parquet_candidates, target_db)

        tabular_candidate = next((p for p in candidates if p.suffix.lower() in TABULAR_EXTS), None)
        if tabular_candidate:
            return _materialize_as_db(tabular_candidate, target_db)
    except Exception as exc:
        return None, f"HF persona DB 준비 실패: {type(exc).__name__}: {exc}"

    return None, f"{repo_id}에서 DB 또는 변환 가능한 표 형식 파일을 찾지 못했습니다."


def normalize_repo_id(text: str) -> str:
    """URL이 와도 org/name 형태로 정규화."""
    value = str(text or "").strip()
    if not value:
        return ""
    value = value.removeprefix("https://huggingface.co/")
    value = value.removeprefix("hf://")
    value = value.removeprefix("datasets/")
    value = value.removeprefix("models/")
    parts = [p for p in value.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return value


def _materialize_as_db(source_path: Path, target_db: Path) -> tuple[Path | None, str]:
    ext = source_path.suffix.lower()
    if ext in DB_EXTS:
        shutil.copy2(source_path, target_db)
        return target_db, f"HF에서 DB 파일 `{source_path.name}`을 받아 `personas.db`로 준비했습니다."

    table_name = source_path.stem.replace("-", "_").replace(" ", "_") or "personas"
    df = _read_tabular(source_path)
    if df is None or df.empty:
        return None, f"`{source_path.name}`을 읽었지만 비어 있거나 변환할 수 없습니다."

    if target_db.exists():
        target_db.unlink()
    conn = sqlite3.connect(target_db)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=5000)
    finally:
        conn.close()
    return target_db, f"HF 파일 `{source_path.name}`을 SQLite로 변환해 `personas.db`를 생성했습니다."


def _materialize_many_as_db(source_paths: list[Path], target_db: Path) -> tuple[Path | None, str]:
    if target_db.exists():
        target_db.unlink()

    table_name = source_paths[0].stem.replace("-", "_").replace(" ", "_") or "personas"
    conn = sqlite3.connect(target_db)
    total_rows = 0
    try:
        for idx, path in enumerate(sorted(source_paths)):
            df = _read_tabular(path)
            if df is None or df.empty:
                continue
            total_rows += len(df)
            df.to_sql(
                table_name,
                conn,
                if_exists="replace" if idx == 0 else "append",
                index=False,
                chunksize=5000,
            )
    finally:
        conn.close()

    if total_rows == 0:
        return None, "parquet 파일은 찾았지만 읽을 수 있는 행이 없었습니다."
    return target_db, f"HF parquet {len(source_paths)}개 파일을 SQLite로 변환해 `personas.db`를 생성했습니다. (총 {total_rows:,}행)"


def _read_tabular(path: Path) -> pd.DataFrame | None:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if ext == ".json":
        return pd.read_json(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    return None
