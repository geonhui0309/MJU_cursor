"""HTML/PDF 리포트 생성."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    from weasyprint import HTML as WeasyHTML

    WEASY_AVAILABLE = True
except Exception:
    WEASY_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "outputs"


def generate_html_report(
    context: dict[str, Any],
    output_path: Path | None = None,
) -> tuple[str, Path]:
    """Jinja2 템플릿으로 HTML 리포트 생성."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")
    context.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    html = template.render(**context)

    out = output_path or OUTPUT_DIR / "insight_report.html"
    out.write_text(html, encoding="utf-8")
    return html, out


def generate_pdf_report(html_path: Path, pdf_path: Path | None = None) -> Path | None:
    """WeasyPrint로 PDF 생성 (옵션)."""
    if not WEASY_AVAILABLE:
        return None
    pdf_path = pdf_path or OUTPUT_DIR / "insight_report.pdf"
    try:
        WeasyHTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return pdf_path
    except Exception:
        return None


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """outputs 폴더에 CSV 저장."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def dataframes_to_html_tables(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    """리포트용 HTML 테이블 변환."""
    result = {}
    for key, df in data.items():
        if df is not None and not df.empty:
            result[key] = df.to_html(index=False, classes="data-table", border=0)
        else:
            result[key] = "<p>데이터 없음</p>"
    return result
