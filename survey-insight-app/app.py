"""
Google Forms CSV 기반 설문조사 인사이트 도출 Streamlit 앱
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 모듈 경로
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

from modules.cleaner import clean_data
from modules.hypothesis_analysis import run_hypothesis_analysis
from modules.insight_generator import generate_insights
from modules.journey_mapping import run_journey_mapping
from modules.keyword_analysis import run_keyword_analysis
from modules.loader import add_response_ids, get_basic_stats, load_csv
from modules.qualitative import run_qualitative_analysis
from modules.quantitative import run_quantitative_analysis
from modules.report_generator import (
    WEASY_AVAILABLE,
    dataframes_to_html_tables,
    generate_html_report,
    generate_pdf_report,
    save_dataframe,
)
from modules.schema_detector import detect_question_types, get_columns_by_type
from modules.sentiment_analysis import run_sentiment_analysis
from modules.text_structure import run_text_structure_analysis

st.set_page_config(
    page_title="Survey Insight — UX Research",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header { padding: 1rem 0; }
    .stMetric { background: #f8fafc; padding: 0.5rem; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Survey Insight")
st.caption("Google Forms CSV 설문 데이터 → UX Research 인사이트 도출")

# ─── 사이드바 입력 ─────────────────────────────────────────
with st.sidebar:
    st.header("입력 설정")
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
    service_name = st.text_input("서비스 이름 *", placeholder="예: 정비소 예약 앱")
    service_description = st.text_area("서비스 설명 *", height=80)
    survey_purpose = st.text_area("설문 목적 *", height=80)
    hypotheses = st.text_area(
        "분석 가설 * (줄바꿈으로 구분)",
        height=120,
        placeholder="사용자는 기능을 찾는 과정에서 어려움을 느끼고 있을 것이다.",
    )
    target_users = st.text_input("주요 타깃 사용자", placeholder="예: 20~30대 신규 사용자")
    known_problems = st.text_input("알고 싶은 문제", placeholder="예: 예약 이탈, 선택 피로")
    journey_input = st.text_input(
        "사용자 여정 단계 (쉼표 구분, 비우면 기본값)",
        placeholder="인지, 진입, 탐색, 선택, 실행, 확인, 재사용, 이탈",
    )
    focus_questions = st.text_input("중점 문항 (쉼표 구분)", placeholder="선택")
    exclude_questions = st.text_input("제외 문항 (쉼표 구분)", placeholder="선택")
    run_btn = st.button("🔍 분석 실행", type="primary", use_container_width=True)

DEFAULT_JOURNEY = ["인지", "진입", "탐색", "선택", "실행", "확인", "재사용", "이탈"]


def _parse_list(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _validate_inputs() -> str | None:
    if uploaded is None:
        return "CSV 파일을 업로드해 주세요."
    if not service_name.strip():
        return "서비스 이름을 입력해 주세요."
    if not service_description.strip():
        return "서비스 설명을 입력해 주세요."
    if not survey_purpose.strip():
        return "설문 목적을 입력해 주세요."
    if not hypotheses.strip():
        return "분석 가설을 입력해 주세요."
    return None


def run_pipeline() -> dict | None:
    """전체 분석 파이프라인."""
    err = _validate_inputs()
    if err:
        st.error(err)
        return None

    try:
        raw_df = load_csv(file_bytes=uploaded.getvalue())
    except ValueError as e:
        st.error(str(e))
        return None

    if raw_df.empty:
        st.error("CSV에 데이터가 없습니다.")
        return None

    if len(raw_df) < 3:
        st.warning("응답 수가 매우 적습니다(3건 미만). 해석에 주의하세요.")

    df = add_response_ids(raw_df)
    basic = get_basic_stats(df)

    exclude_cols = _parse_list(exclude_questions)
    focus_cols = _parse_list(focus_questions)
    journey_stages = _parse_list(journey_input) or DEFAULT_JOURNEY

    schema_df = detect_question_types(df, exclude_columns=exclude_cols, focus_columns=focus_cols)

    cleaned_df, cleaning_log, cleaning_summary = clean_data(df, schema_df)
    save_dataframe(cleaned_df, "cleaned_data.csv")
    save_dataframe(cleaning_log, "cleaning_log.csv")

    excluded_ids = set(cleaning_summary.get("excluded_ids", []))

    # 역채점 문항은 사이드바 '제외 문항' 옆 확장 시 session_state로 전달 가능 (기본: 정채점)
    likert_directions: dict[str, bool] = st.session_state.get("likert_directions", {})

    quant_results, quant_export, quant_interp = run_quantitative_analysis(
        cleaned_df, schema_df, likert_directions, excluded_ids
    )
    save_dataframe(quant_export, "quantitative_analysis.csv")

    qual_results, qual_export = run_qualitative_analysis(
        cleaned_df, schema_df, excluded_ids, cleaning_log
    )
    save_dataframe(qual_export, "qualitative_analysis.csv")

    text_cols = get_columns_by_type(schema_df, "장문형") + get_columns_by_type(schema_df, "단답형")

    text_structure_df = run_text_structure_analysis(cleaned_df, text_cols, excluded_ids)
    keyword_df = run_keyword_analysis(cleaned_df, text_cols, excluded_ids)
    save_dataframe(keyword_df, "keyword_analysis.csv")

    sentiment_df = run_sentiment_analysis(cleaned_df, text_cols, excluded_ids)
    save_dataframe(sentiment_df, "sentiment_analysis.csv")

    journey_df = run_journey_mapping(
        cleaned_df, text_cols, journey_stages, keyword_df, sentiment_df, excluded_ids
    )
    save_dataframe(journey_df, "journey_mapping.csv")

    hypothesis_df = run_hypothesis_analysis(
        hypotheses, schema_df, quant_results, qual_results, keyword_df
    )
    save_dataframe(hypothesis_df, "hypothesis_analysis.csv")

    context = {
        "service_name": service_name,
        "service_description": service_description,
        "survey_purpose": survey_purpose,
        "hypotheses": hypotheses,
        "target_users": target_users or "전체",
        "known_problems": known_problems,
        "journey_stages": journey_stages,
    }

    insights, actions, follow_up = generate_insights(
        context,
        quant_results,
        qual_results,
        keyword_df,
        journey_df,
        hypothesis_df,
        cleaning_summary,
    )

    tables = dataframes_to_html_tables(
        {
            "schema": schema_df,
            "cleaning_log": cleaning_log,
            "keyword": keyword_df,
            "journey": journey_df,
            "hypothesis": hypothesis_df,
            "quant": quant_export,
        }
    )

    cleaning_summary_html = f"""
    <ul>
      <li>원본 응답: {cleaning_summary['original_count']}</li>
      <li>분석 포함: {cleaning_summary['included_count']}</li>
      <li>분석 제외: {cleaning_summary['excluded_count']}</li>
      <li>검토 필요: {cleaning_summary['review_count']}</li>
      <li>데이터 신뢰도: {cleaning_summary['reliability']}</li>
    </ul>
    """

    actions_df = pd.DataFrame(actions)
    save_dataframe(actions_df, "action_items.csv")

    report_ctx = {
        "service_name": service_name,
        "service_description": service_description,
        "survey_purpose": survey_purpose,
        "hypotheses": hypotheses,
        "response_count": basic["total_responses"],
        "question_count": basic["total_questions"],
        "reliability": cleaning_summary["reliability"],
        "cleaning_summary_html": cleaning_summary_html,
        "schema_html": tables.get("schema", ""),
        "quant_html": tables.get("quant", ""),
        "keyword_html": tables.get("keyword", ""),
        "sentiment_html": sentiment_df.to_html(index=False) if not sentiment_df.empty else "<p>없음</p>",
        "journey_html": tables.get("journey", ""),
        "hypothesis_html": tables.get("hypothesis", ""),
        "quant_interpretations": [i["text"] for i in quant_interp],
        "insights": insights,
        "actions_html": actions_df.to_html(index=False) if not actions_df.empty else "<p>없음</p>",
        "follow_up": follow_up,
    }

    _, html_path = generate_html_report(report_ctx)
    pdf_path = generate_pdf_report(html_path)

    return {
        "df": df,
        "cleaned_df": cleaned_df,
        "basic": basic,
        "schema_df": schema_df,
        "cleaning_log": cleaning_log,
        "cleaning_summary": cleaning_summary,
        "quant_results": quant_results,
        "quant_export": quant_export,
        "quant_interp": quant_interp,
        "qual_results": qual_results,
        "qual_export": qual_export,
        "text_structure_df": text_structure_df,
        "keyword_df": keyword_df,
        "sentiment_df": sentiment_df,
        "journey_df": journey_df,
        "hypothesis_df": hypothesis_df,
        "insights": insights,
        "actions": actions,
        "follow_up": follow_up,
        "html_path": html_path,
        "pdf_path": pdf_path,
        "context": context,
    }


def render_results(results: dict) -> None:
    """메인 화면 결과 렌더링."""
    basic = results["basic"]
    cleaning_summary = results["cleaning_summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 응답", basic["total_responses"])
    c2.metric("문항 수", basic["total_questions"])
    c3.metric("분석 포함", cleaning_summary["included_count"])
    c4.metric("데이터 신뢰도", cleaning_summary["reliability"])

    st.subheader("1. 데이터 미리보기")
    st.dataframe(results["df"].head(20), use_container_width=True)

    st.subheader("2. 문항 자동 분류")
    st.dataframe(results["schema_df"], use_container_width=True)

    st.subheader("3. 데이터 정제 요약")
    cols = st.columns(3)
    cols[0].write(f"**제외:** {cleaning_summary['excluded_count']}건")
    cols[1].write(f"**검토 필요:** {cleaning_summary['review_count']}건")
    cols[2].write(f"**마스킹:** {cleaning_summary['masked_count']}건")
    if cleaning_summary.get("label_counts"):
        st.bar_chart(pd.Series(cleaning_summary["label_counts"]))

    st.subheader("4. 정제 필요 응답")
    if not results["cleaning_log"].empty:
        st.dataframe(results["cleaning_log"], use_container_width=True)
    else:
        st.info("정제 대상 응답이 없습니다.")

    st.subheader("5. 정량 분석")
    for item in results["quant_interp"]:
        st.write(f"• {item['text']}")
    for col, res in results["quant_results"].items():
        if col.startswith("_"):
            continue
        with st.expander(col):
            if "distribution" in res:
                st.dataframe(pd.DataFrame(res["distribution"]))
            else:
                st.json({k: v for k, v in res.items() if k != "interpretation"})

    cross = results["quant_results"].get("_cross_analysis", [])
    if cross:
        st.markdown("**교차 분석**")
        for c in cross:
            st.write(c.get("interpretation", ""))

    st.subheader("6. 정성 분석")
    integrated = results["qual_results"].get("integrated", {})
    st.write(integrated.get("summary", ""))
    for col, data in results["qual_results"].get("per_question", {}).items():
        with st.expander(f"문항: {col}"):
            st.write(data.get("summary", ""))
            if data.get("representative_responses"):
                for rep in data["representative_responses"]:
                    st.caption(f"ID {rep['response_id']}: {rep['text']}")

    st.subheader("7. 텍스트 구조 분석")
    if not results["text_structure_df"].empty:
        st.dataframe(results["text_structure_df"], use_container_width=True)

    st.subheader("8. 키워드 분석")
    if not results["keyword_df"].empty:
        st.dataframe(results["keyword_df"], use_container_width=True)

    st.subheader("9. 감성 분석")
    if not results["sentiment_df"].empty:
        st.dataframe(results["sentiment_df"], use_container_width=True)

    st.subheader("10. 사용자 여정 매핑")
    if not results["journey_df"].empty:
        st.dataframe(results["journey_df"], use_container_width=True)

    st.subheader("11. 가설 검토")
    if not results["hypothesis_df"].empty:
        st.dataframe(results["hypothesis_df"], use_container_width=True)

    st.subheader("12. 핵심 인사이트")
    for ins in results["insights"]:
        with st.container(border=True):
            st.markdown(f"### {ins['인사이트 제목']}")
            st.markdown(f"**Fact:** {ins['Fact']}")
            st.markdown(f"**Interpretation:** {ins['Interpretation']}")
            st.markdown(f"**Action:** {ins['Action']}")
            st.caption(
                f"우선순위: {ins['우선순위']} | 신뢰도: {ins['판단 신뢰도']} | "
                f"여정: {ins['관련 여정 단계']}"
            )

    st.subheader("13. 개선 액션 아이템")
    st.dataframe(pd.DataFrame(results["actions"]), use_container_width=True)

    st.subheader("14. 후속 리서치 제안")
    for item in results["follow_up"]:
        st.write(f"• {item}")

    st.subheader("15. 리포트 다운로드")
    col_d1, col_d2, col_d3 = st.columns(3)

    def _file_bytes(path: Path) -> bytes:
        return path.read_bytes() if path.exists() else b""

    with col_d1:
        p = OUTPUT_DIR / "cleaned_data.csv"
        if p.exists():
            st.download_button("cleaned_data.csv", _file_bytes(p), "cleaned_data.csv")
        p2 = OUTPUT_DIR / "cleaning_log.csv"
        if p2.exists():
            st.download_button("cleaning_log.csv", _file_bytes(p2), "cleaning_log.csv")

    with col_d2:
        hp = results.get("html_path")
        if hp and Path(hp).exists():
            st.download_button(
                "insight_report.html",
                _file_bytes(Path(hp)),
                "insight_report.html",
                mime="text/html",
            )

    with col_d3:
        pp = results.get("pdf_path")
        if pp and Path(pp).exists():
            st.download_button(
                "insight_report.pdf",
                _file_bytes(Path(pp)),
                "insight_report.pdf",
                mime="application/pdf",
            )
        elif not WEASY_AVAILABLE:
            st.caption("PDF: WeasyPrint 미설치 — HTML 리포트를 사용하세요.")


# ─── 메인 실행 ─────────────────────────────────────────────
if run_btn:
    with st.spinner("분석 중..."):
        results = run_pipeline()
    if results:
        st.session_state["analysis_results"] = results
        st.success("분석이 완료되었습니다.")

if "analysis_results" in st.session_state:
    render_results(st.session_state["analysis_results"])
else:
    st.info("사이드바에서 CSV와 필수 정보를 입력한 뒤 **분석 실행**을 눌러 주세요.")
    sample = BASE_DIR / "data" / "sample_survey.csv"
    if sample.exists():
        st.markdown("---")
        st.markdown("**샘플 데이터 미리보기** (`data/sample_survey.csv`)")
        st.dataframe(pd.read_csv(sample), use_container_width=True)
