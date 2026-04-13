import runpy
from datetime import datetime
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CRAWLER_APP_PATH = BASE_DIR / "Crawler" / "app.py"
ANALYSIS_APP_PATH = BASE_DIR / "Analysis" / "app.py"
STORAGE_DIR = BASE_DIR / "Storage"
STORAGE_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="리뷰 크롤링/애널리시스 통합", page_icon="📘", layout="wide")

if "nav_tab_index" not in st.session_state:
    st.session_state["nav_tab_index"] = 0

st.markdown(
    """
<style>
  :root {
    --rm-primary: #2563EB;
    --rm-primary-hover: #3B82F6;
    --rm-bg: #020617;
    --rm-surface: rgba(15, 23, 42, 0.5);
    --rm-border: #1E293B;
    --rm-border-active: rgba(59, 130, 246, 0.5);
    --rm-text: #f1f5f9;
    --rm-text-muted: #94a3b8;
    --rm-label: #64748b;
  }

  html, body,
  [data-testid="stAppViewContainer"],
  .stApp {
    background-color: var(--rm-bg) !important;
    color: var(--rm-text) !important;
  }

  header[data-testid="stHeader"] {
    background: rgba(2, 6, 23, 0.82) !important;
    border-bottom: 1px solid var(--rm-border) !important;
  }

  .main h1, .main h2, .main h3, .main p, .main li, .main span {
    color: var(--rm-text) !important;
  }

  .stCaption, [data-testid="stCaption"] {
    color: var(--rm-text-muted) !important;
  }

  .stTextInput input, .stTextArea textarea, div[data-baseweb="input"] input {
    background-color: #020617 !important;
    border: 1px solid var(--rm-border) !important;
    border-radius: 0.75rem !important;
    color: var(--rm-text) !important;
  }

  div[data-baseweb="select"] > div, [data-baseweb="datepicker"] input {
    background-color: #020617 !important;
    border: 1px solid var(--rm-border) !important;
    color: var(--rm-text) !important;
  }

  button[kind="primary"] {
    background: var(--rm-primary) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 0.75rem !important;
  }

  button[kind="secondary"], div[data-testid="stButton"] > button:not([kind="primary"]) {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid var(--rm-border) !important;
    color: var(--rm-text-muted) !important;
    border-radius: 0.75rem !important;
  }

  [data-testid="stDataFrame"] {
    border: 1px solid var(--rm-border) !important;
    border-radius: 0.75rem !important;
  }

  .hero {
    padding: 1.35rem 1.5rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(99, 102, 241, 0.12) 45%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid var(--rm-border);
    color: var(--rm-text);
    margin-bottom: 1rem;
  }
  .hero h1 {
    margin: 0 0 0.4rem 0;
    font-size: 1.55rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f8fafc, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero p { margin: 0; color: var(--rm-text-muted); }
  .rm-hero-pill {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 0.65rem;
  }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="rm-hero-pill">● AI POWERED ANALYSIS</div>
  <h1>리뷰 추출 및 데이터 분석 시스템</h1>
  <p>Google Play 리뷰 수집, CSV 분석, 자료 저장을 한 화면에서 실행합니다.</p>
</div>
""",
    unsafe_allow_html=True,
)

nav_labels = ["탭1. 리뷰 크롤링", "탭2. 리뷰 애널리시스", "탭3. 저장공간"]
nav_idx = int(st.session_state.get("nav_tab_index", 0))
nav_idx = max(0, min(2, nav_idx))
st.caption(f"현재 화면: **{nav_labels[nav_idx]}**")
with st.container(border=True):
    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        if st.button(nav_labels[0], type="primary" if nav_idx == 0 else "secondary", use_container_width=True):
            if nav_idx != 0:
                st.session_state["nav_tab_index"] = 0
                st.rerun()
    with nc2:
        if st.button(nav_labels[1], type="primary" if nav_idx == 1 else "secondary", use_container_width=True):
            if nav_idx != 1:
                st.session_state["nav_tab_index"] = 1
                st.rerun()
    with nc3:
        if st.button(nav_labels[2], type="primary" if nav_idx == 2 else "secondary", use_container_width=True):
            if nav_idx != 2:
                st.session_state["nav_tab_index"] = 2
                st.rerun()

st.divider()

if nav_idx == 0:
    if not CRAWLER_APP_PATH.exists():
        st.error(f"`Crawler/app.py`를 찾을 수 없습니다: {CRAWLER_APP_PATH}")
    else:
        runpy.run_path(str(CRAWLER_APP_PATH), run_name="__main__")
elif nav_idx == 1:
    if not ANALYSIS_APP_PATH.exists():
        st.error(f"`Analysis/app.py`를 찾을 수 없습니다: {ANALYSIS_APP_PATH}")
    else:
        runpy.run_path(str(ANALYSIS_APP_PATH), run_name="__main__")
else:
    st.subheader("저장공간")
    st.caption("CSV/PDF를 업로드해 두면 이 탭에서 다시 다운로드할 수 있습니다.")

    upload = st.file_uploader(
        "CSV 또는 PDF 업로드",
        type=["csv", "pdf"],
        accept_multiple_files=True,
    )
    if upload:
        for f in upload:
            filename = f.name
            target = STORAGE_DIR / filename
            if target.exists():
                stem = target.stem
                suffix = target.suffix
                filename = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
                target = STORAGE_DIR / filename
            target.write_bytes(f.getbuffer())
        st.success(f"{len(upload)}개 파일을 저장했습니다.")

    files = sorted(
        [p for p in STORAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in [".csv", ".pdf"]],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        st.info("저장된 파일이 없습니다.")
    else:
        for p in files:
            c1, c2, c3 = st.columns([3.5, 1.5, 1])
            with c1:
                st.write(f"`{p.name}`")
            with c2:
                st.caption(datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"))
            with c3:
                st.download_button(
                    "다운로드",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="application/pdf" if p.suffix.lower() == ".pdf" else "text/csv",
                    key=f"download_{p.name}",
                    use_container_width=True,
                )