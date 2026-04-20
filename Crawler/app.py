"""
Google Play 앱 리뷰 수집 MVP (Streamlit 단일 파일 버전)
"""

# =========================
# 1) 라이브러리 임포트
# =========================
import io
import re
from pathlib import Path
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import streamlit as st
from google_play_scraper import Sort, reviews, search

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "Storage"
STORAGE_DIR.mkdir(exist_ok=True)


# =========================
# 2) 페이지 기본 설정 및 스타일
# =========================

st.markdown(
    """
    <style>
    .main {
        max-width: 820px;
        margin: 0 auto;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    }
    .hero h1 {
        margin: 0 0 0.2rem 0;
        font-size: 1.45rem;
        font-weight: 700;
    }
    .hero p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 3) 헬퍼 함수
# =========================
def parse_app_id_from_input(raw_input: str) -> Optional[str]:
    """입력값에서 앱 아이디(package name)를 추출."""
    text = raw_input.strip()
    if not text:
        return None

    # Play Store URL에서 id 파라미터 추출
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        query_params = parse_qs(parsed.query)
        app_id = query_params.get("id", [None])[0]
        if app_id:
            return app_id.strip()

    # 일반적인 package 형식이면 그대로 사용
    package_pattern = r"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z0-9_]+)+$"
    if re.match(package_pattern, text):
        return text

    # URL 일부나 문장에 id=com.xxx 형태가 섞여있는 경우
    inline_match = re.search(r"id=([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)+)", text)
    if inline_match:
        return inline_match.group(1)

    return None


def normalize_text(text: str) -> str:
    """검색 정확도를 높이기 위한 간단 정규화."""
    return re.sub(r"\s+", "", text or "").strip().lower()


def search_app_candidates(query: str, lang: str, country: str, n_hits: int = 8) -> List[Dict]:
    """검색 예외를 흡수하고 후보 리스트 반환."""
    try:
        return search(query, lang=lang, country=country, n_hits=n_hits) or []
    except Exception:
        return []


def get_ranked_app_candidates(user_input: str, max_candidates: int = 8) -> List[Dict]:
    """앱 이름 기준으로 중복 제거된 후보 목록을 반환."""
    raw_query = user_input.strip()
    if not raw_query:
        return []

    normalized_query = normalize_text(raw_query)
    query_variants = [raw_query]
    compact_query = re.sub(r"\s+", " ", raw_query).strip()
    if compact_query and compact_query not in query_variants:
        query_variants.append(compact_query)

    search_scopes = [("ko", "kr"), ("ko", "us"), ("en", "kr"), ("en", "us")]
    all_candidates: List[Dict] = []
    seen_app_ids = set()

    for query in query_variants:
        for lang, country in search_scopes:
            candidates = search_app_candidates(query=query, lang=lang, country=country, n_hits=max_candidates)
            for item in candidates:
                app_id = item.get("appId")
                if app_id and app_id not in seen_app_ids:
                    seen_app_ids.add(app_id)
                    all_candidates.append(item)

    if not all_candidates:
        return []

    exact_matches = [
        item for item in all_candidates if normalize_text(item.get("title", "")) == normalized_query
    ]
    contains_matches = [
        item for item in all_candidates if normalized_query in normalize_text(item.get("title", ""))
    ]
    remainder = [item for item in all_candidates if item not in exact_matches and item not in contains_matches]
    ranked = exact_matches + contains_matches + remainder
    return ranked[:max_candidates]


def format_candidate_label(candidate: Dict) -> str:
    """셀렉트박스에 표시할 후보 라벨 생성."""
    title = candidate.get("title", "제목 없음")
    app_id = candidate.get("appId", "appId 없음")
    developer = candidate.get("developer", "개발자 정보 없음")
    score = candidate.get("score")
    score_text = f"{score:.1f}" if isinstance(score, (int, float)) else "-"
    return f"{title} | {app_id} | {developer} | ★ {score_text}"


def sanitize_filename_part(text: str, default_value: str = "app") -> str:
    """파일명에 안전한 문자열로 정규화."""
    safe = re.sub(r'[\\/:*?"<>|]+', "_", str(text or "")).strip()
    safe = re.sub(r"\s+", "_", safe)
    safe = safe.strip("._")
    return safe or default_value


def resolve_app_id(user_input: str) -> Tuple[str, str]:
    """앱 이름/URL/앱ID 입력을 앱ID로 해석하고 설명 텍스트를 반환."""
    direct_app_id = parse_app_id_from_input(user_input)
    if direct_app_id:
        return direct_app_id, f"입력값에서 앱 아이디를 인식했습니다: {direct_app_id}"

    raw_query = user_input.strip()
    normalized_query = normalize_text(raw_query)
    query_variants = [raw_query]
    compact_query = re.sub(r"\s+", " ", raw_query).strip()
    if compact_query and compact_query not in query_variants:
        query_variants.append(compact_query)

    # 한국어/영어 및 KR/US 조합으로 단계적 탐색
    search_scopes = [("ko", "kr"), ("ko", "us"), ("en", "kr"), ("en", "us")]
    all_candidates: List[Dict] = []
    seen_app_ids = set()

    for query in query_variants:
        for lang, country in search_scopes:
            candidates = search_app_candidates(query=query, lang=lang, country=country, n_hits=8)
            for item in candidates:
                app_id = item.get("appId")
                if app_id and app_id not in seen_app_ids:
                    seen_app_ids.add(app_id)
                    all_candidates.append(item)

    if not all_candidates:
        raise ValueError("앱 이름으로 검색 결과를 찾지 못했습니다.")

    # 1순위: 제목 완전 일치(공백 무시)
    exact_matches = [
        item for item in all_candidates if normalize_text(item.get("title", "")) == normalized_query
    ]
    if exact_matches:
        picked = exact_matches[0]
        return picked["appId"], f"앱 이름 정확 일치 결과: {picked.get('title', '')} ({picked['appId']})"

    # 2순위: 제목 포함 일치(공백 무시)
    contains_matches = [
        item for item in all_candidates if normalized_query in normalize_text(item.get("title", ""))
    ]
    if contains_matches:
        picked = contains_matches[0]
        return picked["appId"], f"앱 이름 유사 일치 결과: {picked.get('title', '')} ({picked['appId']})"

    # 3순위: 첫 후보
    picked = all_candidates[0]
    app_id = picked.get("appId")
    title = picked.get("title", "")
    if not app_id:
        raise ValueError("검색 결과에서 앱 아이디를 확인하지 못했습니다.")
    return app_id, f"앱 이름 검색 결과(최상위 후보): {title} ({app_id})"


def fetch_reviews_by_period(
    app_id: str,
    start_date: date,
    end_date: date,
    batch_size: int = 200,
    max_batches: int = 30,
) -> pd.DataFrame:
    """기간 조건에 맞는 리뷰를 페이지네이션으로 수집."""
    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)
    collected: List[Dict] = []
    continuation_token = None

    for _ in range(max_batches):
        result, continuation_token = reviews(
            app_id,
            lang="ko",
            country="kr",
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=continuation_token,
        )

        if not result:
            break

        for row in result:
            review_dt = pd.to_datetime(row.get("at"), errors="coerce")
            if pd.isna(review_dt):
                continue

            review_dt = review_dt.to_pydatetime()
            if review_dt < start_dt:
                # 최신순 정렬이므로 시작일보다 오래된 리뷰가 나오면 종료
                continuation_token = None
                break
            if start_dt <= review_dt <= end_dt:
                collected.append(row)

        if continuation_token is None:
            break

    if not collected:
        return pd.DataFrame()

    df = pd.DataFrame(collected)
    return df


def build_display_df(raw_df: pd.DataFrame, selected_labels: list[str]) -> pd.DataFrame:
    """선택된 항목 기준으로 표시용 DataFrame 생성."""
    if raw_df.empty:
        return pd.DataFrame()

    column_map = {
        "리뷰 아이디": ("reviewId", "리뷰 아이디"),
        "유저 이름": ("userName", "유저 이름"),
        "별점": ("score", "별점"),
        "내용": ("content", "내용"),
        "날짜": ("at", "날짜"),
        "도움되요 수": ("thumbsUpCount", "도움되요 수"),
        "앱버전": ("reviewCreatedVersion", "앱버전"),
        "개발자 답변 내용": ("replyContent", "개발자 답변 내용"),
        "답변 일시": ("repliedAt", "답변 일시"),
    }

    selected_pairs = [column_map[label] for label in selected_labels if label in column_map]
    existing_pairs = [(src, dst) for src, dst in selected_pairs if src in raw_df.columns]
    if not existing_pairs:
        return pd.DataFrame()

    selected_source_cols = [src for src, _ in existing_pairs]
    rename_map = {src: dst for src, dst in existing_pairs}
    df = raw_df[selected_source_cols].copy().rename(columns=rename_map)

    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    if "답변 일시" in df.columns:
        df["답변 일시"] = pd.to_datetime(df["답변 일시"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """DataFrame을 CSV 바이트로 변환."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    return csv_buffer.getvalue().encode("utf-8-sig")


# =========================
# 4) 세션 상태 초기화
# =========================
if "reviews_df" not in st.session_state:
    st.session_state.reviews_df = pd.DataFrame()
if "display_df" not in st.session_state:
    st.session_state.display_df = pd.DataFrame()
if "last_count" not in st.session_state:
    st.session_state.last_count = 0
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "resolved_app_id" not in st.session_state:
    st.session_state.resolved_app_id = ""
if "resolved_app_title" not in st.session_state:
    st.session_state.resolved_app_title = ""
if "show_preview" not in st.session_state:
    st.session_state.show_preview = False
if "date_start" not in st.session_state:
    st.session_state.date_start = date.today() - timedelta(days=7)
if "date_end" not in st.session_state:
    st.session_state.date_end = date.today()
if "app_candidates" not in st.session_state:
    st.session_state.app_candidates = []
if "selected_candidate_index" not in st.session_state:
    st.session_state.selected_candidate_index = 0
if "last_candidate_query" not in st.session_state:
    st.session_state.last_candidate_query = ""
if "last_saved_crawler_csv_sig" not in st.session_state:
    st.session_state.last_saved_crawler_csv_sig = ""


# =========================
# 5) 상단 헤더 UI
# =========================
st.markdown(
    """
    <div class="hero">
        <h1>Google Play 앱 리뷰 수집기</h1>
        <p>앱 이름/URL/앱 아이디와 기간을 입력하면 리뷰를 간편하게 수집합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# 6) 기간 초기화 버튼
# =========================
reset_col1, reset_col2 = st.columns([1, 2])
with reset_col1:
    if st.button("기간 초기화", use_container_width=True):
        st.session_state.date_start = date.today() - timedelta(days=7)
        st.session_state.date_end = date.today()
        st.rerun()


# =========================
# 7) 입력 영역 / 자동 후보 검색 / 전송 버튼
# =========================
app_input = st.text_input(
    "앱 이름 / 앱 아이디 / Play Store URL",
    placeholder="예: 카카오톡 또는 com.kakao.talk 또는 https://play.google.com/store/apps/details?id=com.kakao.talk&hl=ko",
    help="앱 이름을 입력하면 아래에 후보가 자동으로 표시됩니다.",
)
picked_dates = st.date_input(
    "리뷰 수집 기간 (시작일 ~ 종료일)",
    value=(st.session_state.date_start, st.session_state.date_end),
)

selectable_columns = [
    "리뷰 아이디",
    "유저 이름",
    "별점",
    "내용",
    "날짜",
    "도움되요 수",
    "앱버전",
    "개발자 답변 내용",
    "답변 일시",
]
selected_columns = st.multiselect(
    "수집할 항목 선택",
    options=selectable_columns,
    default=selectable_columns,
    help="원하는 항목만 선택해 결과를 확인/다운로드할 수 있습니다.",
)

submitted = st.button("전송", type="primary", use_container_width=True)

normalized_input = app_input.strip()
if not normalized_input:
    st.session_state.app_candidates = []
    st.session_state.selected_candidate_index = 0
    st.session_state.last_candidate_query = ""
else:
    direct_app_id = parse_app_id_from_input(normalized_input)
    if direct_app_id:
        st.session_state.app_candidates = []
        st.session_state.selected_candidate_index = 0
        st.session_state.last_candidate_query = normalized_input
        st.caption("앱 아이디/URL 형식이 인식되어 후보 검색을 생략합니다.")
    elif st.session_state.last_candidate_query != normalized_input:
        with st.spinner("앱 후보를 검색하는 중입니다..."):
            candidates = get_ranked_app_candidates(normalized_input, max_candidates=8)
        st.session_state.app_candidates = candidates
        st.session_state.selected_candidate_index = 0
        st.session_state.last_candidate_query = normalized_input

if st.session_state.app_candidates:
    candidate_labels = [format_candidate_label(item) for item in st.session_state.app_candidates]
    st.session_state.selected_candidate_index = st.selectbox(
        "검색된 앱 후보",
        options=range(len(candidate_labels)),
        format_func=lambda idx: candidate_labels[idx],
        index=min(st.session_state.selected_candidate_index, len(candidate_labels) - 1),
        help="전송 시 선택된 후보의 앱 아이디로 리뷰를 수집합니다.",
    )
elif normalized_input and not parse_app_id_from_input(normalized_input):
    st.caption("검색된 앱 후보가 없습니다.")


# =========================
# 8) 리뷰 수집 처리 및 결과 요약
# =========================
if submitted:
    st.session_state.last_error = ""
    st.session_state.reviews_df = pd.DataFrame()
    st.session_state.display_df = pd.DataFrame()
    st.session_state.last_count = 0
    st.session_state.show_preview = False

    # date_input 반환값 정규화
    if isinstance(picked_dates, tuple) and len(picked_dates) == 2:
        start_date, end_date = picked_dates
    else:
        start_date = picked_dates
        end_date = picked_dates

    st.session_state.date_start = start_date
    st.session_state.date_end = end_date

    if not app_input.strip():
        st.session_state.last_error = "앱 이름 또는 앱 아이디를 입력해 주세요."
    elif start_date > end_date:
        st.session_state.last_error = "시작일은 종료일보다 늦을 수 없습니다."
    elif not selected_columns:
        st.session_state.last_error = "최소 1개 이상의 항목을 선택해 주세요."
    else:
        try:
            with st.spinner("리뷰를 불러오는 중입니다..."):
                direct_app_id = parse_app_id_from_input(app_input.strip())
                if direct_app_id:
                    resolved_app_id = direct_app_id
                    resolved_app_title = resolved_app_id
                    resolve_text = f"입력값에서 앱 아이디를 인식했습니다: {resolved_app_id}"
                else:
                    candidates = st.session_state.app_candidates
                    if not candidates:
                        candidates = get_ranked_app_candidates(app_input.strip(), max_candidates=8)
                        st.session_state.app_candidates = candidates
                        st.session_state.selected_candidate_index = 0

                    if not candidates:
                        raise ValueError("앱 이름으로 검색 결과를 찾지 못했습니다.")

                    selected_idx = min(
                        st.session_state.selected_candidate_index,
                        len(candidates) - 1,
                    )
                    selected = candidates[selected_idx]
                    resolved_app_id = selected.get("appId")
                    resolved_app_title = selected.get("title", resolved_app_id or "")
                    if not resolved_app_id:
                        raise ValueError("선택한 후보에서 앱 아이디를 확인하지 못했습니다.")
                    resolve_text = (
                        f"선택한 앱 후보로 수집합니다: {selected.get('title', '')} ({resolved_app_id})"
                    )

                raw_df = fetch_reviews_by_period(
                    app_id=resolved_app_id,
                    start_date=start_date,
                    end_date=end_date,
                )
                display_df = build_display_df(raw_df, selected_columns)

            st.session_state.reviews_df = raw_df
            st.session_state.display_df = display_df
            st.session_state.last_count = len(raw_df)
            st.session_state.resolved_app_id = resolved_app_id
            st.session_state.resolved_app_title = resolved_app_title
            if not display_df.empty:
                csv_bytes = to_csv_bytes(display_df)
                app_title_for_file = sanitize_filename_part(selected.get("title", "") if not direct_app_id else resolved_app_id)
                period_for_file = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
                saved_name = f"{app_title_for_file}_{period_for_file}.csv"
                st.session_state["crawler_latest_csv_bytes"] = csv_bytes
                st.session_state["crawler_latest_csv_name"] = saved_name
                st.session_state["crawler_latest_source"] = "탭1 리뷰 크롤링 결과"

                save_sig = f"{resolved_app_id}:{len(display_df)}:{start_date}:{end_date}:{hash(csv_bytes)}"
                if st.session_state.last_saved_crawler_csv_sig != save_sig:
                    (STORAGE_DIR / saved_name).write_bytes(csv_bytes)
                    st.session_state.last_saved_crawler_csv_sig = save_sig
            st.info(resolve_text)
        except Exception as error:  # MVP 단계에서 사용자 안내를 위해 예외 메시지 표기
            st.session_state.last_error = (
                "리뷰를 불러오는 중 오류가 발생했습니다. "
                "앱 이름/앱 아이디/기간을 확인하거나 잠시 후 다시 시도해 주세요. "
                f"(상세: {error})"
            )


if st.session_state.last_error:
    st.error(st.session_state.last_error)
elif st.session_state.last_count > 0:
    st.success(f"결과: {st.session_state.last_count}개의 메세지가 불러와짐")
elif submitted:
    st.warning("불러온 리뷰가 없습니다. 앱 입력값 또는 기간을 확인해 주세요.")


# =========================
# 9) 바로보기 / 다운로드 기능
# =========================
if not st.session_state.display_df.empty:
    st.markdown("### 결과 활용")
    period_for_file = f"{st.session_state.date_start.strftime('%Y%m%d')}-{st.session_state.date_end.strftime('%Y%m%d')}"
    app_for_file = sanitize_filename_part(
        st.session_state.get("resolved_app_title") or st.session_state.get("resolved_app_id", ""),
        default_value="reviews",
    )
    download_filename = f"{app_for_file}_{period_for_file}.csv"
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("바로보기", use_container_width=True):
            st.session_state.show_preview = True

    with col2:
        st.download_button(
            "다운로드",
            data=to_csv_bytes(st.session_state.display_df),
            file_name=download_filename,
            mime="text/csv",
            use_container_width=True,
        )
    with col3:
        if st.button("리뷰 분석하기", use_container_width=True):
            st.session_state["nav_tab_index"] = 1
            st.rerun()

    # 버튼 아래에서 전체 폭으로 테이블 표시
    if st.session_state.show_preview:
        st.dataframe(
            st.session_state.display_df,
            use_container_width=True,
            hide_index=True,
        )


# (날짜별 리뷰 수 그래프 기능은 요청에 따라 삭제되었습니다.)
