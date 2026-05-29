"""
Google Forms CSV 기반 설문조사 인사이트 도출 Streamlit 앱
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "outputs"
STORAGE_DIR = BASE_DIR.parent / "Storage"
OUTPUT_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)

from modules.ai_analyzer import get_api_key, run_ai_analysis
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
    initial_sidebar_state="collapsed",
)

DEFAULT_JOURNEY = ["인지", "진입", "탐색", "선택", "실행", "확인", "재사용", "이탈"]
NAV_LABELS = ["탭1. 데이터 입력", "탭2. 분석 결과", "탭3. 인사이트·저장"]
SESSION_API_KEY = "si_openai_api_key"
SESSION_AI_ENABLED = "si_ai_enabled"
SESSION_MODEL = "si_openai_model"

# ─── 스타일 (리뷰 분석기와 유사한 다크 테마 + 상단 탭) ─────────
st.markdown(
    """
<style>
  :root {
    --si-primary: #2563EB;
    --si-bg: #020617;
    --si-surface: rgba(15, 23, 42, 0.55);
    --si-border: #1E293B;
    --si-text: #f1f5f9;
    --si-text-muted: #94a3b8;
  }
  html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--si-bg) !important;
    color: var(--si-text) !important;
  }
  header[data-testid="stHeader"] {
    background: rgba(2, 6, 23, 0.85) !important;
    border-bottom: 1px solid var(--si-border) !important;
  }
  [data-testid="stSidebar"] { display: none; }
  section[data-testid="stSidebar"] { display: none; }
  .main h1, .main h2, .main h3, .main p, .main li, .main span { color: var(--si-text) !important; }
  .stCaption, [data-testid="stCaption"] { color: var(--si-text-muted) !important; }
  .stTextInput input, .stTextArea textarea {
    background-color: #020617 !important;
    border: 1px solid var(--si-border) !important;
    border-radius: 0.75rem !important;
    color: var(--si-text) !important;
  }
  button[kind="primary"] {
    background: var(--si-primary) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 0.75rem !important;
  }
  button[kind="secondary"] {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid var(--si-border) !important;
    color: var(--si-text-muted) !important;
    border-radius: 0.75rem !important;
  }
  [data-testid="stMetric"] {
    background: var(--si-surface) !important;
    border: 1px solid var(--si-border) !important;
    border-radius: 0.75rem !important;
    padding: 0.5rem !important;
  }
  .hero {
    padding: 1.35rem 1.5rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(15,23,42,0.9));
    border: 1px solid var(--si-border);
    margin-bottom: 1rem;
  }
  .hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f8fafc, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .hero p { margin: 0; color: var(--si-text-muted); font-size: 0.95rem; }
  .si-pill {
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
    color: #60a5fa; text-transform: uppercase;
  }
  .file-badge {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; margin-right: 0.5rem;
  }
  .badge-csv { color: #bfdbfe; background: rgba(37,99,235,0.22); border: 1px solid rgba(96,165,250,0.5); }
  .badge-html { color: #bbf7d0; background: rgba(22,163,74,0.2); border: 1px solid rgba(74,222,128,0.5); }
  .badge-pdf { color: #fecaca; background: rgba(220,38,38,0.2); border: 1px solid rgba(248,113,113,0.5); }
</style>
""",
    unsafe_allow_html=True,
)

if "nav_tab_index" not in st.session_state:
    st.session_state["nav_tab_index"] = 0


def _parse_list(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def render_nav() -> int:
    """상단 3탭 네비게이션."""
    nav_idx = int(st.session_state.get("nav_tab_index", 0))
    nav_idx = max(0, min(2, nav_idx))

    st.markdown(
        """
<div class="hero">
  <div class="si-pill">● UX RESEARCH</div>
  <h1>설문조사 인사이트 분석 시스템</h1>
  <p>Google Forms CSV 업로드 → 정제·분석 → 인사이트·리포트까지 한 화면에서 실행합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(f"현재 화면: **{NAV_LABELS[nav_idx]}**")
    with st.container(border=True):
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            if st.button(
                NAV_LABELS[0],
                type="primary" if nav_idx == 0 else "secondary",
                use_container_width=True,
                key="nav_tab_0",
            ):
                if nav_idx != 0:
                    st.session_state["nav_tab_index"] = 0
                    st.rerun()
        with nc2:
            if st.button(
                NAV_LABELS[1],
                type="primary" if nav_idx == 1 else "secondary",
                use_container_width=True,
                key="nav_tab_1",
            ):
                if nav_idx != 1:
                    st.session_state["nav_tab_index"] = 1
                    st.rerun()
        with nc3:
            if st.button(
                NAV_LABELS[2],
                type="primary" if nav_idx == 2 else "secondary",
                use_container_width=True,
                key="nav_tab_2",
            ):
                if nav_idx != 2:
                    st.session_state["nav_tab_index"] = 2
                    st.rerun()

    st.divider()
    return nav_idx


def _validate_inputs(uploaded, service_name, service_description, survey_purpose, hypotheses) -> str | None:
    if uploaded is None:
        return "CSV 파일을 업로드해 주세요."
    if not str(service_name).strip():
        return "서비스 이름을 입력해 주세요."
    if not str(service_description).strip():
        return "서비스 설명을 입력해 주세요."
    if not str(survey_purpose).strip():
        return "설문 목적을 입력해 주세요."
    if not str(hypotheses).strip():
        return "분석 가설을 입력해 주세요."
    return None


def render_ai_settings() -> tuple[bool, str, str]:
    """탭1 OpenAI 설정 UI. Returns (enabled, api_key, model)."""
    import os

    env_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    with st.expander("OpenAI AI 해석 (선택)", expanded=not env_key):
        if env_key:
            st.caption("환경변수 `OPENAI_API_KEY`가 설정되어 있습니다.")
        else:
            st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                key=SESSION_API_KEY,
                help="세션에만 저장됩니다. Streamlit Cloud는 Secrets에 OPENAI_API_KEY를 권장합니다.",
            )
        if SESSION_MODEL not in st.session_state:
            st.session_state[SESSION_MODEL] = os.environ.get("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
        model = st.text_input("모델", key=SESSION_MODEL, disabled=bool(os.environ.get("OPENAI_MODEL", "").strip()))

        api_key = get_api_key(st.session_state.get(SESSION_API_KEY, ""))
        can_ai = bool(api_key)
        if can_ai:
            enabled = st.checkbox("분석 실행 시 OpenAI 해석 포함", value=True, key=SESSION_AI_ENABLED)
        else:
            st.info("API Key가 없으면 **규칙 기반 분석만** 실행됩니다.")
            enabled = False
    api_key = get_api_key(st.session_state.get(SESSION_API_KEY, ""))
    model = st.session_state.get(SESSION_MODEL, "gpt-4o-mini")
    if api_key and SESSION_AI_ENABLED not in st.session_state:
        st.session_state[SESSION_AI_ENABLED] = True
    enabled = bool(st.session_state.get(SESSION_AI_ENABLED, False)) if api_key else False
    return enabled, api_key, model


def _render_ai_block(results: dict, section_key: str, title: str | None = None) -> None:
    """해당 섹션의 OpenAI 해석이 있으면 표시."""
    ai = results.get("ai_interpretations") or {}
    text = ai.get(section_key)
    if text:
        with st.container(border=True):
            st.markdown(f"**🤖 OpenAI 해석** — {title or section_key}")
            st.markdown(text)


def run_pipeline(
    uploaded,
    service_name,
    service_description,
    survey_purpose,
    hypotheses,
    target_users,
    known_problems,
    journey_input,
    focus_questions,
    exclude_questions,
    ai_enabled: bool = False,
    api_key: str = "",
    ai_model: str = "gpt-4o-mini",
) -> dict | None:
    err = _validate_inputs(
        uploaded, service_name, service_description, survey_purpose, hypotheses
    )
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
        context, quant_results, qual_results, keyword_df, journey_df, hypothesis_df, cleaning_summary
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

    # 리포트를 Storage에도 복사
    if html_path and Path(html_path).exists():
        (STORAGE_DIR / "insight_report.html").write_bytes(Path(html_path).read_bytes())
    if pdf_path and Path(pdf_path).exists():
        (STORAGE_DIR / "insight_report.pdf").write_bytes(Path(pdf_path).read_bytes())

    result_bundle = {
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
        "ai_interpretations": {},
        "ai_used": False,
    }

    if ai_enabled and api_key:
        ai_interp = run_ai_analysis(result_bundle, context, api_key, ai_model, enabled=True)
        result_bundle["ai_interpretations"] = ai_interp
        result_bundle["ai_used"] = any(v for v in ai_interp.values() if v)
        if not result_bundle["ai_used"]:
            st.warning("OpenAI 해석 생성에 실패했습니다. 규칙 기반 결과만 표시합니다.")

    return result_bundle


def render_tab_input() -> None:
    """탭1: CSV 업로드·설정·정제."""
    st.subheader("설문 데이터 입력")
    col_l, col_r = st.columns([1, 1])

    with col_l:
        uploaded = st.file_uploader("CSV 파일 업로드 *", type=["csv"], key="si_csv_upload")
        service_name = st.text_input("서비스 이름 *", key="si_service_name", placeholder="예: 정비소 예약 앱")
        service_description = st.text_area("서비스 설명 *", height=100, key="si_service_desc")
        survey_purpose = st.text_area("설문 목적 *", height=100, key="si_survey_purpose")

    with col_r:
        hypotheses = st.text_area(
            "분석 가설 * (줄바꿈 구분)",
            height=120,
            key="si_hypotheses",
            placeholder="사용자는 기능을 찾는 과정에서 어려움을 느끼고 있을 것이다.",
        )
        target_users = st.text_input("주요 타깃 사용자", key="si_target_users")
        known_problems = st.text_input("알고 싶은 문제", key="si_known_problems")
        journey_input = st.text_input(
            "사용자 여정 단계 (쉼표 구분)",
            key="si_journey",
            placeholder="인지, 진입, 탐색, 선택, 실행, 확인, 재사용, 이탈",
        )
        c1, c2 = st.columns(2)
        with c1:
            focus_questions = st.text_input("중점 문항", key="si_focus_q")
        with c2:
            exclude_questions = st.text_input("제외 문항", key="si_exclude_q")

    ai_enabled, api_key, ai_model = render_ai_settings()

    st.markdown("")
    run_btn = st.button("🔍 분석 실행 (규칙 기반 + 선택 AI)", type="primary", use_container_width=True)

    if run_btn:
        msg = "규칙 기반 분석 및 OpenAI 해석 생성 중..." if (ai_enabled and api_key) else "규칙 기반 분석 중..."
        with st.spinner(msg):
            results = run_pipeline(
                uploaded,
                service_name,
                service_description,
                survey_purpose,
                hypotheses,
                target_users,
                known_problems,
                journey_input,
                focus_questions,
                exclude_questions,
                ai_enabled=ai_enabled,
                api_key=api_key,
                ai_model=ai_model,
            )
        if results:
            st.session_state["analysis_results"] = results
            st.session_state["nav_tab_index"] = 1
            ai_note = " (AI 해석 포함)" if results.get("ai_used") else ""
            st.success(f"분석이 완료되었습니다{ai_note}. **탭2**에서 결과, **탭3**에서 인사이트를 확인하세요.")
            st.rerun()

    results = st.session_state.get("analysis_results")
    if results:
        basic = results["basic"]
        cleaning_summary = results["cleaning_summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전체 응답", basic["total_responses"])
        m2.metric("문항 수", basic["total_questions"])
        m3.metric("분석 포함", cleaning_summary["included_count"])
        m4.metric("데이터 신뢰도", cleaning_summary["reliability"])

        st.subheader("데이터 미리보기")
        st.dataframe(results["df"].head(20), use_container_width=True)

        st.subheader("문항 자동 분류")
        st.dataframe(results["schema_df"], use_container_width=True)

        st.subheader("데이터 정제")
        cc1, cc2, cc3 = st.columns(3)
        cc1.write(f"**제외:** {cleaning_summary['excluded_count']}건")
        cc2.write(f"**검토 필요:** {cleaning_summary['review_count']}건")
        cc3.write(f"**마스킹:** {cleaning_summary['masked_count']}건")
        if cleaning_summary.get("label_counts"):
            st.bar_chart(pd.Series(cleaning_summary["label_counts"]))
        if not results["cleaning_log"].empty:
            st.dataframe(results["cleaning_log"], use_container_width=True)
    else:
        st.info("CSV와 필수 정보를 입력한 뒤 **분석 실행**을 눌러 주세요.")
        sample = BASE_DIR / "data" / "sample_survey.csv"
        if sample.exists():
            st.markdown("**샘플 데이터** (`data/sample_survey.csv`)")
            st.dataframe(pd.read_csv(sample), use_container_width=True)


def render_tab_analysis(results: dict) -> None:
    """탭2: 정량·정성·텍스트 분석 + OpenAI 해석."""
    if results.get("ai_used"):
        st.success("OpenAI 해석이 포함되어 있습니다. 각 섹션 아래 🤖 블록을 확인하세요.")
    else:
        st.caption(
            "현재 **규칙 기반 분석**만 표시됩니다. "
            "탭1에서 OpenAI API Key를 입력하고 「OpenAI 해석 포함」을 켠 뒤 다시 분석 실행하세요."
        )

    _render_ai_block(results, "개요·데이터 품질", "개요·데이터 품질")

    st.subheader("정량 분석")
    for item in results["quant_interp"]:
        st.write(f"• {item['text']}")
    for col, res in results["quant_results"].items():
        if col.startswith("_"):
            continue
        with st.expander(col):
            if "distribution" in res:
                st.dataframe(pd.DataFrame(res["distribution"]), use_container_width=True)
            else:
                st.json({k: v for k, v in res.items() if k != "interpretation"})

    cross = results["quant_results"].get("_cross_analysis", [])
    if cross:
        st.markdown("**교차 분석**")
        for c in cross:
            st.write(c.get("interpretation", ""))
    _render_ai_block(results, "정량·교차 분석")

    st.subheader("정성 분석")
    integrated = results["qual_results"].get("integrated", {})
    st.write(integrated.get("summary", ""))
    for col, data in results["qual_results"].get("per_question", {}).items():
        with st.expander(f"문항: {col}"):
            st.write(data.get("summary", ""))
            for rep in data.get("representative_responses", []):
                st.caption(f"ID {rep['response_id']}: {rep['text']}")
    _render_ai_block(results, "정성 분석")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("텍스트 구조")
        if not results["text_structure_df"].empty:
            st.dataframe(results["text_structure_df"], use_container_width=True)
    with c2:
        st.subheader("키워드")
        if not results["keyword_df"].empty:
            st.dataframe(results["keyword_df"], use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("감성 분석")
        if not results["sentiment_df"].empty:
            st.dataframe(results["sentiment_df"], use_container_width=True)
    with c4:
        st.subheader("사용자 여정")
        if not results["journey_df"].empty:
            st.dataframe(results["journey_df"], use_container_width=True)
    _render_ai_block(results, "키워드·감성")

    st.subheader("가설 검토")
    if not results["hypothesis_df"].empty:
        st.dataframe(results["hypothesis_df"], use_container_width=True)
    _render_ai_block(results, "가설 검토")


def render_tab_insights_storage(results: dict) -> None:
    """탭3: 인사이트·액션·다운로드·저장공간."""
    _render_ai_block(results, "최종 AI 인사이트", "최종 AI 인사이트")

    st.subheader("핵심 인사이트 (규칙 기반)")
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

    st.subheader("개선 액션 아이템")
    st.dataframe(pd.DataFrame(results["actions"]), use_container_width=True)

    st.subheader("후속 리서치 제안")
    for item in results["follow_up"]:
        st.write(f"• {item}")

    st.divider()
    st.subheader("분석 결과 다운로드")

    def _bytes(path: Path) -> bytes:
        return path.read_bytes() if path.exists() else b""

    d1, d2, d3, d4 = st.columns(4)
    outputs = [
        ("cleaned_data.csv", "text/csv"),
        ("cleaning_log.csv", "text/csv"),
        ("keyword_analysis.csv", "text/csv"),
        ("hypothesis_analysis.csv", "text/csv"),
    ]
    for col, (fname, mime) in zip([d1, d2, d3, d4], outputs):
        p = OUTPUT_DIR / fname
        with col:
            if p.exists():
                st.download_button(fname, _bytes(p), fname, mime=mime, use_container_width=True)

    d5, d6 = st.columns(2)
    hp = results.get("html_path")
    if hp and Path(hp).exists():
        with d5:
            st.download_button(
                "insight_report.html",
                _bytes(Path(hp)),
                "insight_report.html",
                mime="text/html",
                use_container_width=True,
            )
    pp = results.get("pdf_path")
    if pp and Path(pp).exists():
        with d6:
            st.download_button(
                "insight_report.pdf",
                _bytes(Path(pp)),
                "insight_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    elif not WEASY_AVAILABLE:
        st.caption("PDF: WeasyPrint 미설치 시 HTML 리포트를 사용하세요.")

    st.divider()
    st.subheader("저장공간")
    st.caption("CSV·HTML·PDF를 업로드해 두면 이 탭에서 다시 다운로드할 수 있습니다.")

    upload = st.file_uploader(
        "파일 업로드",
        type=["csv", "pdf", "html"],
        accept_multiple_files=True,
        key="si_storage_upload",
    )
    if upload:
        for f in upload:
            filename = f.name
            target = STORAGE_DIR / filename
            if target.exists():
                stem, suffix = target.stem, target.suffix
                filename = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
                target = STORAGE_DIR / filename
            target.write_bytes(f.getbuffer())
        st.success(f"{len(upload)}개 파일을 저장했습니다.")

    files = sorted(
        [
            p
            for p in STORAGE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv", ".pdf", ".html"}
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        st.info("저장된 파일이 없습니다. 분석 실행 후 리포트가 자동 저장됩니다.")
    else:
        for p in files:
            fc1, fc2, fc3 = st.columns([3.5, 1.5, 1])
            ext = p.suffix.lower()
            badge = "badge-pdf" if ext == ".pdf" else ("badge-html" if ext == ".html" else "badge-csv")
            label = ext.upper().replace(".", "")
            with fc1:
                st.markdown(
                    f'<span class="file-badge {badge}">{label}</span> `{p.name}`',
                    unsafe_allow_html=True,
                )
            with fc2:
                st.caption(datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
            with fc3:
                mime = (
                    "application/pdf"
                    if ext == ".pdf"
                    else ("text/html" if ext == ".html" else "text/csv")
                )
                st.download_button(
                    "다운로드",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime=mime,
                    key=f"dl_{p.name}",
                    use_container_width=True,
                )


# ─── 메인 ─────────────────────────────────────────────────
nav_idx = render_nav()
results = st.session_state.get("analysis_results")

if nav_idx == 0:
    render_tab_input()
elif nav_idx == 1:
    if results:
        render_tab_analysis(results)
    else:
        st.warning("먼저 **탭1. 데이터 입력**에서 분석을 실행해 주세요.")
        if st.button("탭1으로 이동", type="primary"):
            st.session_state["nav_tab_index"] = 0
            st.rerun()
else:
    if results:
        render_tab_insights_storage(results)
    else:
        st.warning("먼저 **탭1. 데이터 입력**에서 분석을 실행해 주세요.")
        if st.button("탭1으로 이동", type="primary"):
            st.session_state["nav_tab_index"] = 0
            st.rerun()
