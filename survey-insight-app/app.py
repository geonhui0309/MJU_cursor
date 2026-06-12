"""
Google Forms CSV 기반 설문조사 인사이트 도출 Streamlit 앱
"""

from __future__ import annotations

import html as html_module
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "outputs"
STORAGE_DIR = BASE_DIR.parent / "Storage"
OUTPUT_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)

from modules.ai_analyzer import get_api_key, run_ai_analysis
from modules.behavior_summary import enhance_behaviors_with_ai, summarize_usage_behaviors
from modules.service_research import run_service_research
from modules.cleaner import clean_data
from modules.hf_persona_loader import prepare_persona_db_from_hf
from modules.hypothesis_analysis import run_hypothesis_analysis
from modules.insight_generator import generate_insights
from modules.journey_mapping import run_journey_mapping
from modules.keyword_analysis import run_keyword_analysis
from modules.loader import add_response_ids, get_basic_stats, load_csv
from modules.persona_virtual_research import (
    build_persona_exports,
    run_persona_research,
)
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
from modules import visualizations as viz

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
SESSION_RESEARCH_ENABLED = "si_research_enabled"
SESSION_MODEL = "si_openai_model"
SESSION_HF_API_KEY = "si_hf_api_key"
SESSION_HF_SOURCE_PRESET = "si_hf_source_preset"
SESSION_HF_DATASET_REPO = "si_hf_dataset_repo"
SESSION_HF_DATASET_FILE = "si_hf_dataset_file"
SESSION_HF_DATASET_REVISION = "si_hf_dataset_revision"
PERSONA_DB_CACHE = BASE_DIR / "data" / "personas.db"
KOREA_NEMOTRON_REPO_ID = os.environ.get("HF_KOREA_NEMOTRON_REPO_ID", "nvidia/Nemotron-Personas-Korea").strip()
KOREA_NEMOTRON_FILENAME = os.environ.get("HF_KOREA_NEMOTRON_FILENAME", "").strip()
KOREA_NEMOTRON_REVISION = os.environ.get("HF_KOREA_NEMOTRON_REVISION", "").strip()
HF_PERSONA_PRESETS = {
    "korea_nemotron": {
        "label": "Korea Nemotron Persona DB",
        "repo_id": KOREA_NEMOTRON_REPO_ID,
        "filename": KOREA_NEMOTRON_FILENAME,
        "revision": KOREA_NEMOTRON_REVISION,
    },
    "custom": {
        "label": "직접 입력",
        "repo_id": "",
        "filename": "",
        "revision": "",
    },
}

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
  [data-testid="stVerticalBlock"] > div:has(> [data-testid="stTabs"]) { gap: 0.25rem; }
  h5 { margin-top: 0.5rem !important; margin-bottom: 0.35rem !important; color: #93c5fd !important; }
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


def render_ai_settings() -> tuple[bool, str, str, bool, str, str, str, str]:
    """탭1 OpenAI + HF DB 설정 UI."""
    env_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    with st.expander("LLM 설정", expanded=not env_key):
        st.markdown("**OpenAI**")
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

        if SESSION_RESEARCH_ENABLED not in st.session_state:
            st.session_state[SESSION_RESEARCH_ENABLED] = True
        st.checkbox(
            "서비스·경쟁사 웹 리서치 포함 (검색+AI 리포트)",
            value=st.session_state.get(SESSION_RESEARCH_ENABLED, True),
            key=SESSION_RESEARCH_ENABLED,
            help="서비스명으로 웹 검색합니다. AI 종합 리포트는 API Key가 있을 때 생성됩니다.",
        )

        st.markdown("**Hugging Face Persona DB**")
        if SESSION_HF_SOURCE_PRESET not in st.session_state:
            st.session_state[SESSION_HF_SOURCE_PRESET] = "korea_nemotron"
        if SESSION_HF_DATASET_REPO not in st.session_state:
            st.session_state[SESSION_HF_DATASET_REPO] = ""
        if SESSION_HF_DATASET_FILE not in st.session_state:
            st.session_state[SESSION_HF_DATASET_FILE] = ""
        if SESSION_HF_DATASET_REVISION not in st.session_state:
            st.session_state[SESSION_HF_DATASET_REVISION] = ""
        selected_preset = st.selectbox(
            "Persona DB 소스",
            options=list(HF_PERSONA_PRESETS.keys()),
            index=list(HF_PERSONA_PRESETS.keys()).index(st.session_state.get(SESSION_HF_SOURCE_PRESET, "korea_nemotron")) if st.session_state.get(SESSION_HF_SOURCE_PRESET, "korea_nemotron") in HF_PERSONA_PRESETS else 0,
            format_func=lambda key: HF_PERSONA_PRESETS[key]["label"],
            key=SESSION_HF_SOURCE_PRESET,
        )
        preset = HF_PERSONA_PRESETS[selected_preset]
        if selected_preset == "custom":
            hf_repo_input = st.text_input(
                "HF repo id",
                key=SESSION_HF_DATASET_REPO,
                help="예: org_or_user/repo_name. 데이터셋 repo 기준입니다.",
                placeholder="예: my-org/korean-persona-db",
            )
            hfc1, hfc2 = st.columns(2)
            with hfc1:
                hf_file_input = st.text_input(
                    "HF 파일명 (선택)",
                    key=SESSION_HF_DATASET_FILE,
                    placeholder="예: personas.db 또는 personas.csv",
                )
            with hfc2:
                hf_revision_input = st.text_input(
                    "HF revision (선택)",
                    key=SESSION_HF_DATASET_REVISION,
                    placeholder="main",
                )
        else:
            hf_repo_input = preset.get("repo_id", "")
            hf_file_input = preset.get("filename", "")
            hf_revision_input = preset.get("revision", "")
            st.caption(f"선택된 preset: `{hf_repo_input}`")

        with st.expander("고급 옵션: 비공개 HF repo", expanded=False):
            hf_token_input = st.text_input(
                "HF Token (선택)",
                type="password",
                placeholder="hf_...",
                key=SESSION_HF_API_KEY,
                help="공개 repo는 비워도 됩니다. 비공개/gated repo 다운로드 시에만 필요합니다.",
            )
        st.caption("HF는 persona DB 다운로드/준비에만 사용합니다. Virtual IDI / Validation acting은 OpenAI로 생성됩니다.")

    api_key = get_api_key(st.session_state.get(SESSION_API_KEY, ""))
    model = st.session_state.get(SESSION_MODEL, "gpt-4o-mini")
    hf_api_key = str(hf_token_input or st.session_state.get(SESSION_HF_API_KEY, "") or "").strip()
    hf_repo_id = str(hf_repo_input or "").strip()
    hf_filename = str(hf_file_input or "").strip()
    hf_revision = str(hf_revision_input or "").strip()
    if api_key and SESSION_AI_ENABLED not in st.session_state:
        st.session_state[SESSION_AI_ENABLED] = True
    enabled = bool(st.session_state.get(SESSION_AI_ENABLED, False)) if api_key else False
    research = bool(st.session_state.get(SESSION_RESEARCH_ENABLED, False))
    return enabled, api_key, model, research, hf_api_key, hf_repo_id, hf_filename, hf_revision


def ensure_persona_db(hf_repo_id: str, hf_token: str, hf_filename: str, hf_revision: str) -> tuple[Path | None, str]:
    """로컬 캐시 또는 HF 자동 다운로드로 persona DB를 준비."""
    if hf_repo_id.strip():
        return prepare_persona_db_from_hf(
            repo_id=hf_repo_id,
            output_dir=PERSONA_DB_CACHE.parent,
            token=hf_token,
            filename=hf_filename,
            repo_type="dataset",
            revision=hf_revision,
        )
    if PERSONA_DB_CACHE.exists():
        return PERSONA_DB_CACHE, ""
    return None, ""


def _render_legend(legend: list[str]) -> None:
    """차트 아래 번호·항목 목록 (한글은 브라우저 폰트로 표시)."""
    if not legend:
        return
    st.markdown("**항목**")
    for line in legend:
        st.markdown(f"- {line}")


def _show_fig(fig, legend: list[str] | None = None) -> None:
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    _render_legend(legend or [])


def _show_pie(data: pd.Series | pd.DataFrame, title: str = "") -> None:
    if isinstance(data, pd.DataFrame):
        pie_vals = data["건수"] if "건수" in data.columns else data.iloc[:, 0]
    else:
        pie_vals = data
    fig, legend = viz.make_pie_figure(pie_vals, title)
    _show_fig(fig, legend)


def _show_bar_numbered(df: pd.DataFrame, height: int = 160) -> None:
    """Streamlit 막대 차트 + 번호 범례."""
    if df is None or df.empty:
        return
    numbered, legend = viz.numbered_index_df(df)
    st.bar_chart(numbered, height=height)
    _render_legend(legend)


def render_analysis_dashboard(results: dict) -> None:
    """탭2 — 한 화면 요약 대시보드 (차트·워드클라우드 밀집 배치)."""
    basic = results["basic"]
    cleaning = results["cleaning_summary"]
    kw_df = results.get("keyword_df")
    sent_df = results.get("sentiment_df")
    journey_df = results.get("journey_df")
    text_df = results.get("text_structure_df")
    hyp_df = results.get("hypothesis_df")
    quant = results.get("quant_results", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("응답", basic["total_responses"])
    m2.metric("문항", basic["total_questions"])
    m3.metric("분석 포함", cleaning["included_count"])
    m4.metric("신뢰도", cleaning["reliability"])
    m5.metric("핵심 키워드", len(kw_df) if kw_df is not None and not kw_df.empty else 0)

    _render_behavior_summary_block(results)

    st.markdown("##### 핵심 지표")
    r1a, r1b, r1c = st.columns(3)
    likert_df = viz.likert_summary_df(quant)
    with r1a:
        st.caption("리커트 평균")
        if not likert_df.empty:
            fig, leg = viz.make_likert_horizontal_figure(likert_df)
            _show_fig(fig, leg)
        else:
            st.info("리커트 문항 없음")
    with r1b:
        st.caption("감성 분포")
        sent_series = viz.sentiment_counts_df(sent_df)
        if not sent_series.empty:
            _show_pie(sent_series, "감성")
        else:
            st.info("감성 데이터 없음")
    with r1c:
        st.caption("데이터 정제")
        clean_s = viz.cleaning_counts_series(cleaning)
        if not clean_s.empty:
            _show_pie(clean_s, "응답 처리")
        else:
            st.info("정제 데이터 없음")

    st.markdown("##### 키워드")
    r2a, r2b = st.columns([1, 1])
    kw_chart = viz.keyword_chart_df(kw_df, top_n=12)
    with r2a:
        st.caption("상위 키워드 빈도")
        if not kw_chart.empty:
            _show_bar_numbered(kw_chart, height=150)
        else:
            st.info("키워드 없음")
    with r2b:
        st.caption("워드 클라우드")
        wc_fig = viz.make_wordcloud_figure(kw_df)
        if wc_fig:
            _show_fig(wc_fig)
        elif not viz.WORDCLOUD_AVAILABLE:
            st.caption("`pip install wordcloud` 설치 시 표시됩니다. 좌측 막대 차트를 참고하세요.")
        else:
            st.info("워드클라우드 생성 불가")

    st.markdown("##### 여정·텍스트·선택지")
    r3a, r3b, r3c = st.columns(3)
    journey_chart = viz.journey_chart_df(journey_df)
    with r3a:
        st.caption("여정 단계별 응답")
        if not journey_chart.empty:
            _show_bar_numbered(journey_chart, height=140)
        else:
            st.info("여정 매핑 없음")
    with r3b:
        st.caption("텍스트 구조 유형")
        ts = viz.text_structure_counts_df(text_df)
        if not ts.empty:
            _show_pie(ts["건수"] if "건수" in ts.columns else ts.iloc[:, 0], "구조 유형")
        else:
            st.info("텍스트 구조 없음")
    with r3c:
        st.caption("선택형 문항 Top")
        choice_df = viz.choice_distribution_df(quant, max_questions=2)
        if choice_df is not None and not choice_df.empty:
            fig, leg = viz.make_grouped_choice_figure(choice_df)
            _show_fig(fig, leg)
        else:
            st.info("선택형 문항 없음")

    hyp_chart = viz.hypothesis_verdict_df(hyp_df)
    if not hyp_chart.empty:
        st.markdown("##### 가설 검증 요약")
        c1, c2 = st.columns([1, 2])
        with c1:
            _show_bar_numbered(hyp_chart, height=130)
        with c2:
            hyp_cols = [c for c in ("가설", "지지 여부", "판단 근거") if c in hyp_df.columns]
            st.dataframe(hyp_df[hyp_cols].head(5), use_container_width=True, hide_index=True)

    if results.get("quant_interp"):
        st.markdown("##### 정량 해석 한줄 요약")
        for item in results["quant_interp"][:4]:
            st.markdown(f"- {item['text']}")


def render_quant_detail(results: dict) -> None:
    """정량 문항별 차트 + 표 (요약 + 선택형 상세)."""
    quant = results["quant_results"]
    valid_items = [
        (col, res)
        for col, res in quant.items()
        if not col.startswith("_") and isinstance(res, dict)
    ]
    if not valid_items:
        st.info("정량 분석 가능한 문항이 없습니다.")
        return

    summary_rows = []
    for col, res in valid_items:
        summary_rows.append(
            {
                "문항": col,
                "유형": res.get("type", "-"),
                "응답수": res.get("total", "-"),
                "주요 지표": (
                    f"평균 {res.get('mean', '-')}"
                    if res.get("type") == "리커트 척도형"
                    else (f"Top {res.get('top', '-')}" if "top" in res else "-")
                ),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    selected_col = st.selectbox(
        "상세 분석 문항 선택",
        options=[col for col, _ in valid_items],
        key="quant_detail_select",
    )
    selected_res = dict(valid_items)[selected_col]

    left, right = st.columns([1.1, 1])
    with left:
        if "distribution" in selected_res:
            dist_df = pd.DataFrame(selected_res["distribution"])
            if not dist_df.empty and "choice" in dist_df.columns:
                chart_df = dist_df.set_index("choice")[["ratio"]]
                chart_df.columns = ["비율(%)"]
                _show_bar_numbered(chart_df, height=140)
        elif selected_res.get("type") == "리커트 척도형":
            likert_one = viz.likert_summary_df({selected_col: selected_res})
            fig, leg = viz.make_likert_horizontal_figure(likert_one)
            _show_fig(fig, leg)
            st.caption(
                f"평균 {selected_res.get('mean')} · 긍정 {selected_res.get('positive_pct')}% · "
                f"부정 {selected_res.get('negative_pct')}%"
            )
    with right:
        if "distribution" in selected_res:
            st.dataframe(pd.DataFrame(selected_res["distribution"]), use_container_width=True, hide_index=True)
        else:
            st.json({k: v for k, v in selected_res.items() if k != "interpretation"})
    if selected_res.get("interpretation"):
        st.info(selected_res["interpretation"])

    cross = quant.get("_cross_analysis", [])
    if cross:
        st.markdown("**교차 분석**")
        for c in cross:
            st.write(c.get("interpretation", ""))


def render_qual_keyword_detail(results: dict) -> None:
    """정성·키워드·감성·여정 — 시각화 우선."""
    integrated = results["qual_results"].get("integrated", {})
    st.write(integrated.get("summary", ""))

    st.markdown("##### 키워드 · 감성")
    a, b = st.columns(2)
    kw_df = results.get("keyword_df")
    sent_df = results.get("sentiment_df")
    with a:
        kw_chart = viz.keyword_chart_df(kw_df, top_n=15)
        if not kw_chart.empty:
            _show_bar_numbered(kw_chart, height=150)
        wc = viz.make_wordcloud_figure(kw_df)
        if wc:
            _show_fig(wc)
    with b:
        sent_series = viz.sentiment_counts_df(sent_df)
        if not sent_series.empty:
            _show_pie(sent_series, "감성")
        if sent_df is not None and not sent_df.empty:
            with st.expander("감성 상세 표"):
                show_cols = [c for c in ["감성", "의미", "건수", "대표 응답"] if c in sent_df.columns]
                st.dataframe(sent_df[show_cols].head(12), use_container_width=True, hide_index=True)

    st.markdown("##### 여정 · 텍스트 구조")
    c1, c2 = st.columns(2)
    with c1:
        jc = viz.journey_chart_df(results.get("journey_df"))
        if not jc.empty:
            _show_bar_numbered(jc, height=140)
        jdf = results.get("journey_df")
        if jdf is not None and not jdf.empty:
            st.dataframe(jdf, use_container_width=True, hide_index=True)
    with c2:
        ts = viz.text_structure_counts_df(results.get("text_structure_df"))
        if not ts.empty:
            _show_pie(ts["건수"] if "건수" in ts.columns else ts.iloc[:, 0], "텍스트 구조")
        tdf = results.get("text_structure_df")
        if tdf is not None and not tdf.empty:
            with st.expander("텍스트 구조 상세"):
                st.dataframe(tdf.head(20), use_container_width=True, hide_index=True)

    st.markdown("##### 정성 코멘트")
    per_q = results["qual_results"].get("per_question", {})
    if not per_q:
        st.info("정성 코멘트가 없습니다.")
        return

    qual_rows = []
    for col, data in per_q.items():
        qual_rows.append(
            {
                "문항": col,
                "응답수": data.get("response_count", 0),
                "구체성": data.get("specificity", "-"),
                "요약": (data.get("summary", "") or "")[:120],
            }
        )
    st.dataframe(pd.DataFrame(qual_rows), use_container_width=True, hide_index=True)

    selected_q = st.selectbox(
        "정성 코멘트 상세 문항 선택",
        options=list(per_q.keys()),
        key="qual_detail_select",
    )
    selected_data = per_q[selected_q]
    st.write(selected_data.get("summary", ""))
    reps = selected_data.get("representative_responses", [])
    if reps:
        st.markdown("**대표 응답**")
        for rep in reps[:5]:
            st.caption(f"ID {rep['response_id']}: {rep['text']}")


def _render_behavior_summary_block(results: dict) -> None:
    """전체 응답자 핵심 사용 행태 (정성+정량 통합)."""
    bp = results.get("behavior_summary") or {}
    if not bp.get("behaviors") and not bp.get("summary_text"):
        return
    with st.container(border=True):
        st.markdown("### 👥 전체 응답자 핵심 사용 행태")
        st.caption(
            f"정량·정성 데이터 통합 · 전체 {bp.get('total_responses', '-')}명 기준"
        )
        if bp.get("ai_narrative"):
            st.markdown(bp["ai_narrative"])
        else:
            st.markdown(bp.get("summary_text", "").replace("\n", "\n\n"))
        export_df = bp.get("export_df")
        if export_df is not None and not export_df.empty:
            st.dataframe(
                export_df[
                    [
                        c
                        for c in (
                            "행태 유형",
                            "핵심 행태",
                            "정량 근거",
                            "정성 근거",
                            "여정 단계",
                            "관련 문항",
                        )
                        if c in export_df.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


def _render_service_research_block(results: dict) -> None:
    """서비스·경쟁사 웹 리서치 결과."""
    sr = results.get("service_research") or {}
    report = sr.get("report")
    if not report:
        return
    with st.container(border=True):
        st.markdown("### 🌐 서비스·경쟁사 리서치")
        meta = sr.get("meta") or {}
        if sr.get("search_ok"):
            cap = f"웹 검색 스니펫 {len(sr.get('snippets', []))}건 반영"
            excluded = meta.get("excluded_count", 0)
            if excluded:
                cap += f" (무관 결과 {excluded}건 제외)"
            if meta.get("ambiguous") and meta.get("canonical_name"):
                cap += f" · 분석 대상: {meta['canonical_name']}"
            st.caption(cap)
        st.markdown(report)
        if sr.get("snippets"):
            with st.expander("검색 출처 스니펫"):
                for i, s in enumerate(sr["snippets"][:8], 1):
                    st.markdown(f"**{i}. {s.get('title', '')}**  \n{s.get('snippet', '')[:300]}")


def _render_ai_block(results: dict, section_key: str, title: str | None = None) -> None:
    """해당 섹션의 OpenAI 해석이 있으면 표시."""
    ai = results.get("ai_interpretations") or {}
    text = ai.get(section_key)
    if text:
        with st.container(border=True):
            st.markdown(f"**🤖 OpenAI 해석** — {title or section_key}")
            st.markdown(text)


def _render_persona_research_block(results: dict) -> None:
    """HF persona DB 기반 가상 사용자 검증."""
    persona = results.get("persona_research") or {}
    if not persona.get("db_available") and not persona.get("summary"):
        return

    with st.container(border=True):
        st.markdown("### 🧑 HF Persona DB · Virtual IDI")
        if persona.get("summary"):
            st.markdown(persona["summary"])
        if persona.get("error"):
            st.warning(persona["error"])
            return

        segment = persona.get("segment_profile") or {}
        attrs = segment.get("attributes") or {}
        caps = []
        if segment.get("summary"):
            caps.append(f"세그먼트: {segment['summary']}")
        if persona.get("db_meta", {}).get("table"):
            caps.append(f"DB 테이블: {persona['db_meta']['table']}")
        if persona.get("db_meta", {}).get("ranking_method"):
            caps.append(f"랭킹: {persona['db_meta']['ranking_method']}")
        if persona.get("matched_personas"):
            caps.append(f"매칭: {len(persona['matched_personas'])}명")
        if caps:
            st.caption(" · ".join(caps))

        if attrs:
            st.dataframe(
                pd.DataFrame([{"속성": k, "대표 값": v} for k, v in attrs.items()]),
                use_container_width=True,
                hide_index=True,
            )

        matches = persona.get("matched_personas") or []
        if matches:
            match_df = pd.DataFrame(
                [
                    {
                        "persona_id": p.get("persona_id", ""),
                        "label": p.get("label", ""),
                        "match_score": p.get("match_score", 0),
                        "attr_score": p.get("attr_score", 0),
                        "token_score": p.get("token_score", 0),
                        "semantic_score": p.get("semantic_score", 0),
                        "match_reasons": ", ".join(p.get("match_reasons", [])),
                        "profile_excerpt": p.get("profile_excerpt", ""),
                    }
                    for p in matches
                ]
            )
            st.markdown("**유사 페르소나 매칭**")
            st.dataframe(match_df, use_container_width=True, hide_index=True)

        idi_sessions = persona.get("virtual_idi") or []
        if idi_sessions:
            st.markdown("**Virtual IDI** (OpenAI acting)")
            for session in idi_sessions:
                with st.expander(
                    f"{session.get('persona_label', session.get('persona_id', 'persona'))} · {session.get('insight_title', '')}",
                    expanded=False,
                ):
                    for qa in session.get("qa", []):
                        st.markdown(f"**Q. {qa.get('question', '')}**")
                        st.write(qa.get("answer", ""))
                    if session.get("takeaway"):
                        st.caption(f"Takeaway: {session['takeaway']}")
        elif persona.get("db_available"):
            st.caption("API Key가 없거나 생성에 실패해 Virtual IDI는 아직 비어 있습니다.")

        validations = persona.get("validation") or []
        if validations:
            st.markdown("**Insight Validation** (OpenAI acting)")
            val_rows = []
            for item in validations:
                val_rows.append(
                    {
                        "insight_title": item.get("insight_title", ""),
                        "overall_verdict": item.get("overall_verdict", ""),
                        "overall_score": item.get("overall_score", ""),
                        "rationale": item.get("rationale", ""),
                    }
                )
            st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)

            with st.expander("페르소나별 검증 상세"):
                for item in validations:
                    st.markdown(f"**{item.get('insight_title', '')}**")
                    st.caption(
                        f"{item.get('overall_verdict', '')} · {item.get('overall_score', '')}"
                    )
                    for person in item.get("by_persona", []):
                        st.write(
                            f"- {person.get('persona_label', person.get('persona_id', ''))}: "
                            f"{person.get('verdict', '')} ({person.get('score', '')})"
                        )
                        if person.get("reason"):
                            st.caption(person["reason"])



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
    research_enabled: bool = True,
    persona_db_path: str | Path | None = None,
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

    behavior_summary = summarize_usage_behaviors(
        quant_results,
        qual_results,
        keyword_df,
        journey_df,
        sentiment_df,
        schema_df,
        cleaning_summary,
        basic["total_responses"],
    )
    bdf_out = behavior_summary.get("export_df")
    if bdf_out is not None and not bdf_out.empty:
        save_dataframe(bdf_out, "behavior_summary.csv")

    survey_summary_parts = [item.get("text", "") for item in quant_interp[:6]]
    if keyword_df is not None and not keyword_df.empty:
        survey_summary_parts.append(
            "키워드: " + ", ".join(keyword_df["키워드"].head(8).astype(str).tolist())
        )
    for ins in insights[:4]:
        survey_summary_parts.append(f"{ins.get('인사이트 제목')}: {ins.get('Fact', '')[:120]}")
    survey_summary = "\n".join(survey_summary_parts)

    service_research = run_service_research(
        service_name=str(service_name),
        service_description=str(service_description),
        survey_purpose=str(survey_purpose),
        known_problems=str(known_problems or ""),
        survey_summary=survey_summary,
        api_key=api_key,
        model=ai_model,
        enabled=research_enabled,
    )
    context["service_research"] = service_research

    persona_research = run_persona_research(
        db_path=persona_db_path,
        cleaned_df=cleaned_df,
        insights=insights,
        context=context,
        openai_api_key=api_key if ai_enabled else "",
        openai_model=ai_model,
        enabled=True,
    )
    persona_exports = build_persona_exports(persona_research)
    for fname, export_df in (
        ("persona_matches.csv", persona_exports["matches"]),
        ("virtual_idi.csv", persona_exports["idi"]),
        ("insight_validation.csv", persona_exports["validation"]),
    ):
        if export_df is not None and not export_df.empty:
            save_dataframe(export_df, fname)

    if ai_enabled and api_key:
        ai_behavior = enhance_behaviors_with_ai(behavior_summary, context, api_key, ai_model)
        if ai_behavior:
            behavior_summary["ai_narrative"] = ai_behavior

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

    sr_report = service_research.get("report") or ""
    esc_sr = html_module.escape(sr_report)
    service_research_html = (
        f"<div class='insight-card'><pre style='white-space:pre-wrap'>{esc_sr}</pre></div>"
        if sr_report
        else "<p>리서치 없음</p>"
    )

    # 리포트 생성 이전에 result_bundle에서 참조될 수 있으므로 초기화합니다.
    html_path = None
    pdf_path = None

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
        "service_research": service_research,
        "behavior_summary": behavior_summary,
        "persona_research": persona_research,
        "persona_ai_used": persona_research.get("ai_used", False),
        "ai_interpretations": {},
        "ai_used": False,
    }

    if ai_enabled and api_key:
        ai_interp = run_ai_analysis(result_bundle, context, api_key, ai_model, enabled=True)
        result_bundle["ai_interpretations"] = ai_interp
        result_bundle["ai_used"] = any(v for v in ai_interp.values() if v)
        if not result_bundle["ai_used"]:
            st.warning("OpenAI 해석 생성에 실패했습니다. 규칙 기반 결과만 표시합니다.")

    beh_text = behavior_summary.get("ai_narrative") or behavior_summary.get("summary_text") or ""
    behavior_summary_html = (
        f"<div class='insight-card'><pre style='white-space:pre-wrap'>{html_module.escape(beh_text)}</pre></div>"
        if beh_text
        else "<p>없음</p>"
    )
    behavior_table_html = ""
    bdf = behavior_summary.get("export_df")
    if bdf is not None and not bdf.empty:
        behavior_table_html = bdf.to_html(index=False, classes="data-table", border=0)

    report_ctx = {
        "service_name": service_name,
        "service_description": service_description,
        "survey_purpose": survey_purpose,
        "hypotheses": hypotheses,
        "response_count": basic["total_responses"],
        "question_count": basic["total_questions"],
        "reliability": cleaning_summary["reliability"],
        "behavior_summary_html": behavior_summary_html,
        "behavior_table_html": behavior_table_html,
        "service_research_html": service_research_html,
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
        "ai_report_html": "",
    }
    ai_final = result_bundle.get("ai_interpretations", {}).get("최종 AI 인사이트")
    if ai_final:
        report_ctx["ai_report_html"] = (
            f"<div class='insight-card'><pre style='white-space:pre-wrap'>"
            f"{html_module.escape(ai_final)}</pre></div>"
        )

    _, html_path = generate_html_report(report_ctx)
    pdf_path = generate_pdf_report(html_path)
    result_bundle["html_path"] = html_path
    result_bundle["pdf_path"] = pdf_path

    if html_path and Path(html_path).exists():
        (STORAGE_DIR / "insight_report.html").write_bytes(Path(html_path).read_bytes())
    if pdf_path and Path(pdf_path).exists():
        (STORAGE_DIR / "insight_report.pdf").write_bytes(Path(pdf_path).read_bytes())

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

    (
        ai_enabled,
        api_key,
        ai_model,
        research_enabled,
        hf_api_key,
        hf_repo_id,
        hf_filename,
        hf_revision,
    ) = render_ai_settings()

    st.markdown("")
    run_btn = st.button("🔍 분석 실행 (규칙 기반 + 선택 AI)", type="primary", use_container_width=True)

    if run_btn:
        persona_db_path, persona_db_note = ensure_persona_db(
            hf_repo_id,
            hf_api_key,
            hf_filename,
            hf_revision,
        )
        persona_llm_enabled = bool(api_key) and ai_enabled
        msg = "분석·웹 리서치·가상 인터뷰 생성 중..." if (ai_enabled and api_key) or persona_llm_enabled else "규칙 기반 분석 중..."
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
                research_enabled=research_enabled,
                persona_db_path=persona_db_path,
            )
        if results:
            st.session_state["analysis_results"] = results
            st.session_state["nav_tab_index"] = 1
            notes = []
            if results.get("ai_used"):
                notes.append("OpenAI 해석 포함")
            if results.get("persona_ai_used"):
                notes.append("가상 인터뷰/검증 포함")
            note_text = f" ({', '.join(notes)})" if notes else ""
            st.success(f"분석이 완료되었습니다{note_text}. **탭2**에서 결과, **탭3**에서 인사이트를 확인하세요.")
            if persona_db_note:
                st.info(persona_db_note)
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
        cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1.2])
        cc1.metric("제외", cleaning_summary["excluded_count"])
        cc2.metric("검토 필요", cleaning_summary["review_count"])
        cc3.metric("마스킹", cleaning_summary["masked_count"])
        with cc4:
            clean_s = viz.cleaning_counts_series(cleaning_summary)
            if not clean_s.empty:
                _show_pie(clean_s, "정제 유형")
        if not results["cleaning_log"].empty:
            with st.expander("정제 로그 상세"):
                st.dataframe(results["cleaning_log"], use_container_width=True, hide_index=True)
        _render_service_research_block(results)
        _render_behavior_summary_block(results)
        _render_persona_research_block(results)
    else:
        st.info("CSV와 필수 정보를 입력한 뒤 **분석 실행**을 눌러 주세요.")
        sample = BASE_DIR / "data" / "sample_survey.csv"
        if sample.exists():
            st.markdown("**샘플 데이터** (`data/sample_survey.csv`)")
            st.dataframe(pd.read_csv(sample), use_container_width=True)


def render_tab_analysis(results: dict) -> None:
    """탭2: 대시보드 중심 시각화 + 상세 탭."""
    if results.get("ai_used"):
        st.success("OpenAI 해석 포함 — **AI 해석** 탭에서 모아볼 수 있습니다.")
    else:
        st.caption("탭1에서 OpenAI를 켜면 해석 탭이 활성화됩니다.")

    sub_tabs = ["📊 한눈에 보기", "👥 핵심 사용 행태", "🧑 가상 사용자", "정량 상세", "정성·키워드", "가설", "🤖 AI 해석"]
    t1, t1b, t1c, t2, t3, t4, t5 = st.tabs(sub_tabs)

    with t1:
        render_analysis_dashboard(results)

    with t1b:
        _render_behavior_summary_block(results)
        _render_ai_block(results, "핵심 사용 행태")
        _render_ai_block(results, "핵심 사용 행태 (AI)")

    with t1c:
        _render_persona_research_block(results)

    with t2:
        st.subheader("정량 분석 상세")
        render_quant_detail(results)
        _render_ai_block(results, "정량·교차 분석")

    with t3:
        st.subheader("정성 · 키워드 · 감성")
        render_qual_keyword_detail(results)
        _render_ai_block(results, "정성 분석")
        _render_ai_block(results, "키워드·감성")

    with t4:
        st.subheader("가설 검토")
        hyp_df = results.get("hypothesis_df")
        if hyp_df is not None and not hyp_df.empty:
            h1, h2 = st.columns([1, 2])
            with h1:
                hc = viz.hypothesis_verdict_df(hyp_df)
                if not hc.empty:
                    _show_bar_numbered(hc, height=130)
            with h2:
                st.dataframe(hyp_df, use_container_width=True, hide_index=True)
        else:
            st.info("가설 분석 결과 없음")
        _render_ai_block(results, "가설 검토")

    with t5:
        st.subheader("OpenAI 해석 모음")
        _render_service_research_block(results)
        _render_ai_block(results, "서비스·경쟁 리서치")
        _render_ai_block(results, "개요·데이터 품질", "개요·데이터 품질")
        _render_ai_block(results, "정량·교차 분석")
        _render_ai_block(results, "정성 분석")
        _render_ai_block(results, "키워드·감성")
        _render_ai_block(results, "가설 검토")
        if not results.get("ai_interpretations"):
            st.info("AI 해석이 없습니다. 탭1에서 API Key와 「OpenAI 해석 포함」을 설정 후 다시 분석하세요.")


def render_tab_insights_storage(results: dict) -> None:
    """탭3: 인사이트·액션·다운로드·저장공간."""
    _render_behavior_summary_block(results)
    _render_service_research_block(results)
    _render_persona_research_block(results)
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
    actions = results.get("actions", [])
    if actions:
        act_df = pd.DataFrame(actions)
        ac1, ac2 = st.columns([1, 1.2])
        with ac1:
            pri = act_df["우선순위"].value_counts() if "우선순위" in act_df.columns else None
            if pri is not None and not pri.empty:
                _show_bar_numbered(pri.to_frame("건수"), height=120)
        with ac2:
            st.dataframe(act_df, use_container_width=True, hide_index=True)
    else:
        st.info("액션 아이템 없음")

    st.subheader("후속 리서치 제안")
    for item in results["follow_up"]:
        st.write(f"• {item}")

    st.divider()
    st.subheader("분석 결과 다운로드")

    def _bytes(path: Path) -> bytes:
        return path.read_bytes() if path.exists() else b""

    d1, d2, d3, d4, d5 = st.columns(5)
    outputs = [
        ("cleaned_data.csv", "text/csv"),
        ("cleaning_log.csv", "text/csv"),
        ("keyword_analysis.csv", "text/csv"),
        ("hypothesis_analysis.csv", "text/csv"),
        ("behavior_summary.csv", "text/csv"),
    ]
    for col, (fname, mime) in zip([d1, d2, d3, d4, d5], outputs):
        p = OUTPUT_DIR / fname
        with col:
            if p.exists():
                st.download_button(fname, _bytes(p), fname, mime=mime, use_container_width=True)

    d6, d7, d8 = st.columns(3)
    persona_outputs = [
        ("persona_matches.csv", "text/csv"),
        ("virtual_idi.csv", "text/csv"),
        ("insight_validation.csv", "text/csv"),
    ]
    extra_cols = [d6, d7, d8]
    for col, (fname, mime) in zip(extra_cols, persona_outputs):
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
