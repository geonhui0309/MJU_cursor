import os
import re
import json
import urllib.request
from collections import Counter

import streamlit as st
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML as WeasyHTML
from datetime import datetime

# -------------------------
# 페이지 상단: 제목/설명
# -------------------------
st.title("OpenAI 리뷰 분석 리포트 대시보드")
st.caption(
    "CSV 업로드 후 기본 분석 결과를 보고, OpenAI 해석을 통해 각 섹션의 의미를 더 깊게 이해할 수 있습니다."
)

# OpenAI 해석 기능 토글(키가 없어도 앱은 실행됩니다)
OPENAI_ENABLED = False
_api_key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
if _api_key_present:
    OPENAI_ENABLED = st.checkbox("OpenAI 해석 포함", value=True)
else:
    st.info("OpenAI 해석을 사용하려면 `OPENAI_API_KEY` 환경변수를 설정하세요. (기본 분석만 표시됩니다.)")

# 카드형 레이아웃을 위한 간단한 스타일
st.markdown(
    """
<style>
  .section-card {
    background: #ffffff;
    border: 1px solid rgba(49, 51, 63, 0.12);
    border-radius: 16px;
    padding: 18px 18px 8px 18px;
    margin: 12px 0 18px 0;
    box-shadow: 0 1px 10px rgba(0,0,0,0.04);
  }
  .section-title {
    font-size: 20px;
    font-weight: 700;
    margin: 0 0 6px 0;
  }
  .section-desc {
    color: rgba(49, 51, 63, 0.75);
    margin: 0 0 10px 0;
    font-size: 14px;
  }
  .kpi-card {
    background: #f7f9ff;
    border: 1px solid rgba(49, 51, 63, 0.10);
    border-radius: 16px;
    padding: 14px 14px 6px 14px;
    margin: 6px 0 10px 0;
  }
  .note-box {
    background: rgba(25, 118, 210, 0.06);
    border: 1px solid rgba(25, 118, 210, 0.20);
    border-radius: 12px;
    padding: 12px 12px;
    margin-top: 12px;
    color: rgba(25, 118, 210, 0.96);
  }
</style>
""",
    unsafe_allow_html=True,
)


def _norm_col(x):
    # 컬럼명 비교를 쉽게 하기 위해 공백 제거 + 대소문자 무시
    return str(x).strip().lower()


def simple_tokenize(text: str):
    """
    아주 단순한 토큰화:
    - 영문/숫자/한글을 남기고 나머지는 공백 처리
    - 너무 짧은 단어(1글자)는 제거
    """
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", str(text))
    words = text.split()
    words = [w for w in words if len(w) >= 2]
    return words


def render_wordcloud_tagcloud(word_freqs, max_words: int = 40):
    """
    외부 wordcloud 패키지 없이, 태그형 워드클라우드를 HTML/CSS로 표시합니다.
    """
    items = word_freqs[:max_words]
    if not items:
        st.caption("워드클라우드를 표시할 키워드가 없습니다.")
        return

    values = [v for _, v in items if isinstance(v, (int, float))]
    if not values:
        st.caption("워드클라우드를 표시할 키워드 값이 없습니다.")
        return

    max_v = max(values)
    min_v = min(values)
    span_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    html_spans = []
    for idx, (w, v) in enumerate(items):
        # 빈도가 비슷할 때는 크기 계산이 흔들리지 않게 보정
        if max_v == min_v:
            size = 22
        else:
            ratio = (v - min_v) / (max_v - min_v)
            size = int(12 + 24 * (ratio ** 0.6))
        color = span_colors[idx % len(span_colors)]
        safe_w = str(w).replace("<", "").replace(">", "")
        html_spans.append(
            f"""
<span style="
  display:inline-block;
  padding:6px 10px;
  margin:4px 4px;
  border-radius:999px;
  background: rgba(0,0,0,0.04);
  border: 1px solid rgba(0,0,0,0.07);
  color: {color};
  font-weight: 700;
  font-size: {size}px;
  line-height: 1;
">{safe_w}</span>
"""
        )

    st.markdown(
        f"""
<div style="display:flex;flex-wrap:wrap;justify-content:flex-start;align-items:center;">
  {''.join(html_spans)}
</div>
""",
        unsafe_allow_html=True,
    )


def call_openai_interpretation(section_name: str, payload_text: str):
    """
    OpenAI API를 직접 HTTP로 호출합니다.
    - 별도 openai 패키지 설치가 필요 없습니다.
    - OPENAI_API_KEY가 없으면 None을 반환합니다.
    """
    global OPENAI_ENABLED
    if not OPENAI_ENABLED:
        return None

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    system_msg = (
        "너는 데이터 분석 리포트 해설 전문가다. "
        "아래에 주어진 CSV 리뷰 분석 결과를 바탕으로 "
        "초보자도 이해할 수 있는 쉬운 한국어로 짧고 명확하게 설명해라."
    )

    extra_requirements = ""
    if section_name == "감성 분석":
        extra_requirements = """
추가 지침(반드시 포함):
4) 감성 키워드가 가리키는 원인을 'UX 문제', '기능 문제', '성능 문제'로 분류해서 설명해라.
5) '높은 별점인데도' 부정 원인 키워드가 많이 보이는 경우(positive 리뷰 안에 negative 이슈 키워드가 섞인 패턴)와,
   '낮은 별점인데도' 긍정 키워드가 보이는 경우(negative 리뷰 안에 positive 이슈 키워드가 섞인 패턴)에 대해
   각각 어떤 인사이트가 가능한지 제시해라.
6) 마지막에는 개선 우선순위를 2~3개로 요약해라.
"""

    user_msg = f"""섹션: {section_name}

분석 결과 요약(숫자/키워드/표):
{payload_text}

요구:
1) 왜 이런 패턴이 나오는지 가능한 원인을 3-5줄로 설명
2) 다음 액션(제품 개선 방향)을 2-3개 제안
3) 불필요한 수식/복잡한 용어는 피할 것
{extra_requirements}
"""

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
    }

    req = urllib.request.Request(
        url="https://api.openai.com/v1/chat/completions",
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        return obj["choices"][0]["message"]["content"].strip()
    except Exception:
        # OpenAI 실패는 앱 실행을 막지 않도록 무조건 안전하게 처리
        return None


uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is not None:
    # -------------------------
    # 업로드: pandas로 CSV 읽기
    # -------------------------
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("CSV를 읽는 중 오류가 발생했습니다. 파일 인코딩/형식을 확인해주세요.")
        st.code(str(e))
        st.stop()

    st.success("CSV 파일이 성공적으로 업로드되었습니다.")

    # -------------------------
    # 파일 정보 + 컬럼 매핑(영문/다른 이름도 자동 인식)
    # -------------------------
    normalized_to_original = {}
    for c in df.columns:
        normalized_to_original[_norm_col(c)] = c

    # 목표(한국어 컬럼) -> 후보(원본 컬럼명들)
    target_to_candidates = {
        "평점": ["평점", "별점", "score", "rating", "star", "stars"],
        "리뷰": ["리뷰", "내용", "review", "reviews", "content", "text", "comment", "body", "commentcontent"],
        "작성일": ["작성일", "날짜", "date", "datetime", "time", "at", "createdat", "created_at"],
        "리뷰 아이디": ["리뷰 아이디", "reviewid", "review_id", "id"],
        "유저 이름": ["유저 이름", "username", "user_name", "user"],
        "도움되요 수": ["도움되요 수", "thumbsupcount", "thumbs_up_count", "likes", "likecount"],
        "앱버전": [
            "앱버전",
            "reviewcreatedversion",
            "review_created_version",
            "appversion",
            "app_version",
            "version",
        ],
        "개발자 답변 내용": ["개발자 답변 내용", "replycontent", "reply_content", "reply"],
        "답변 일시": ["답변 일시", "repliedat", "replied_at", "replytime", "reply_time"],
    }

    for target_col, candidates in target_to_candidates.items():
        if target_col in df.columns:
            continue
        found_original = None
        for cand in candidates:
            key = _norm_col(cand)
            if key in normalized_to_original:
                found_original = normalized_to_original[key]
                break
        if found_original is not None:
            df[target_col] = df[found_original]

    # -------------------------
    # 파일 정보(기본 정보)
    # -------------------------
    st.markdown(
        """
<div class="section-card">
  <div class="section-title">1) 기본 정보</div>
  <div class="section-desc">업로드된 파일 요약과 상위 5개 행을 확인합니다.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    info_cols = st.columns([1.2, 1, 1, 1])
    with info_cols[0]:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("파일명", uploaded_file.name)
        st.markdown("</div>", unsafe_allow_html=True)
    with info_cols[1]:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("행(리뷰) 수", f"{df.shape[0]:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with info_cols[2]:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("열(컬럼) 수", f"{df.shape[1]:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with info_cols[3]:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("컬럼 목록", f"{len(df.columns)}개")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # 데이터 미리보기
    # -------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">데이터 미리보기</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">상위 5개 행을 먼저 확인하세요. (컬럼명은 `평점`, `작성일`, `리뷰`를 기준으로 자동 매핑합니다.)</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(5), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # 분석 시작 버튼
    # -------------------------
    st.markdown("---")
    start = st.button("분석 시작", type="primary", use_container_width=True)

    if start:
        # ====================================
        # 분석 공통 준비
        # ====================================
        df_work = df.copy()

        # PDF 생성용 기본값(분석 조건에 따라 일부는 비어있을 수 있습니다)
        overview_month_counts = []
        pain_rows = []
        pain_interpret = None
        stage_df = None
        feature_candidates = []
        feat_interpret = None
        overview_interpret = None
        kw_interpret = None
        sent_interpret = None
        final_interpret = None

        has_rating = "평점" in df_work.columns
        has_date = "작성일" in df_work.columns
        has_review = "리뷰" in df_work.columns

        # 날짜/평점/리뷰 타입 정리
        if has_date:
            df_work["작성일"] = pd.to_datetime(df_work["작성일"], errors="coerce")
        if has_rating:
            df_work["평점"] = pd.to_numeric(df_work["평점"], errors="coerce")
        if has_review:
            df_work["리뷰"] = df_work["리뷰"].fillna("").astype(str)

        # 단순 불용어(무의미 단어) 제거용
        stopwords = set(
            [
                "정말",
                "너무",
                "그냥",
                "그리고",
                "하지만",
                "또",
                "정도",
                "것",
                "수",
                "때문",
                "때문에",
                "같이",
                "더",
                "진짜",
                "거",
                "수도",
                "합니다",
                "있습니다",
                "없습니다",
                "좋아요",
                "별로",
                "최고",
                "최악",
                "사용",
                "이용",
                "됩니다",
                "해요",
                "하다",
                "근데",
                "근",
                "내",
                "저",
                "우리",
                "그",
                "거기",
                "여기",
            ]
        )

        # -------------------------
        # 개요 분석
        # -------------------------
        st.markdown(
            """
<div class="section-card">
  <div class="section-title">2) 개요 분석</div>
  <div class="section-desc">전체 규모, 평균 평점, 리뷰 작성 기간을 확인하고 기간별 리뷰 흐름을 봅니다.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.header("개요 분석")

        total_reviews = int(df_work.shape[0])

        avg_rating = None
        if has_rating and df_work["평점"].dropna().shape[0] > 0:
            avg_rating = df_work["평점"].dropna().mean()

        period_text = None
        if has_date:
            valid_dates = df_work["작성일"].dropna()
            if not valid_dates.empty:
                date_min = valid_dates.min()
                date_max = valid_dates.max()
                days = (date_max - date_min).days
                period_text = f"{date_min.date()} ~ {date_max.date()} ({days}일)"

        # KPI 카드(상단 크게)
        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            st.metric("전체 리뷰 수", f"{total_reviews:,}")
        with kpi_cols[1]:
            st.metric("평균 평점", "-" if avg_rating is None else f"{avg_rating:.2f}")
        with kpi_cols[2]:
            st.metric("리뷰 작성 기간", "-" if period_text is None else period_text)

        # 기간별 리뷰 수: 월 단위 추천(점이 너무 많지 않게)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">기간별 리뷰 수</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-desc">작성일 기준으로 월별 리뷰 개수를 집계합니다.</div>',
            unsafe_allow_html=True,
        )

        change_notes = []
        if has_date:
            df_month = df_work.dropna(subset=["작성일"]).copy()
            if df_month.empty:
                st.warning("`작성일` 값이 비어있어 차트를 만들 수 없습니다.")
            else:
                df_month["월"] = df_month["작성일"].dt.to_period("M").astype(str)
                counts = df_month.groupby("월").size().reset_index(name="리뷰 수")
                st.line_chart(counts.set_index("월"))
                # PDF용 월별 표 데이터 저장(템플릿은 month/count 키를 기대)
                overview_month_counts = [
                    {"month": r.get("월", ""), "count": int(r.get("리뷰 수", 0))}
                    for r in counts.to_dict(orient="records")
                ]

                # 급격한 변화 감지: 이전 달 대비 증감률
                counts = counts.sort_values("월")
                counts["이전 리뷰 수"] = counts["리뷰 수"].shift(1)
                counts["증감률"] = (counts["리뷰 수"] - counts["이전 리뷰 수"]) / counts["이전 리뷰 수"]

                # 상승/하락 이벤트 추출(너무 민감하지 않게 임계값 적용)
                for _, row in counts.iterrows():
                    prev = row["이전 리뷰 수"]
                    if pd.isna(prev) or prev == 0:
                        continue
                    rate = row["증감률"]
                    month = row["월"]

                    # 이벤트 기준(초보자용 단순 규칙)
                    # - 상승: +50% 이상
                    # - 하락: -50% 이하
                    if rate >= 0.5 and row["리뷰 수"] >= 3:
                        if "앱버전" in df_month.columns:
                            sub = df_month[df_month["월"] == month]
                            ver = sub["앱버전"].dropna()
                            ver_name = ver.value_counts().head(1)
                            ver_text = (
                                f"{ver_name.index[0]} (동일 월 언급 최다)"
                                if len(ver_name) > 0
                                else "버전 정보 없음"
                            )
                        else:
                            ver_text = "버전 정보 없음"
                        change_notes.append(f"{month}: 급증(+{int(rate*100)}%) / {ver_text}")

                    if rate <= -0.5 and row["리뷰 수"] <= prev * 0.7:
                        if "앱버전" in df_month.columns:
                            sub = df_month[df_month["월"] == month]
                            ver = sub["앱버전"].dropna()
                            ver_name = ver.value_counts().head(1)
                            ver_text = (
                                f"{ver_name.index[0]} (동일 월 언급 최다)"
                                if len(ver_name) > 0
                                else "버전 정보 없음"
                            )
                        else:
                            ver_text = "버전 정보 없음"
                        change_notes.append(f"{month}: 급감({int(rate*100)}%) / {ver_text}")

                if len(change_notes) == 0:
                    st.caption("급격한 증감이 뚜렷하지 않아 업데이트 버전 표시는 생략했습니다.")
                else:
                    # 업데이트/변화 이벤트 요약
                    st.info("차트 변동이 큰 구간(월) 요약:")
                    for note in change_notes[:6]:
                        st.write(f"- {note}")
        else:
            st.warning("`작성일` 컬럼이 없어 기간별 차트를 표시할 수 없습니다.")

        st.markdown("</div>", unsafe_allow_html=True)

        # OpenAI 해석(개요)
        overview_payload = f"총 리뷰: {total_reviews}, 평균 평점: {avg_rating}, 기간: {period_text}\n변동 이벤트(상위 6개): {change_notes[:6]}"
        overview_interpret = call_openai_interpretation("개요 분석", overview_payload)
        if overview_interpret:
            st.markdown('<div class="note-box">', unsafe_allow_html=True)
            st.subheader("OpenAI 해석(개요)")
            st.write(overview_interpret)
            st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # 키워드 분석 / VoC
        # -------------------------
        st.markdown(
            """
<div class="section-card">
  <div class="section-title">3) 키워드 / Voice of Customer</div>
  <div class="section-desc">리뷰 텍스트 기반 TOP 키워드와 구간별(월 단위) 빈도 변화에서 핵심 변화를 찾습니다.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.header("키워드 분석")

        top20 = []
        keyword_changes = []

        if not has_review:
            st.warning("`리뷰` 컬럼이 없어 키워드 분석을 수행할 수 없습니다.")
        else:
            all_tokens = []
            for t in df_work["리뷰"].tolist():
                all_tokens.extend(simple_tokenize(t))
            all_tokens = [w for w in all_tokens if w not in stopwords]
            counter = Counter(all_tokens)
            top20 = [(k, v) for k, v in counter.most_common(20)]

            if len(top20) == 0:
                st.info("추출할 키워드가 없습니다.")
            else:
                kw_df = pd.DataFrame(top20, columns=["키워드", "빈도"])
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">TOP 20 키워드</div>', unsafe_allow_html=True)
                    st.dataframe(kw_df, use_container_width=True, height=320)
                    # 태그형 워드클라우드(외부 패키지 없이 HTML/CSS로 표현)
                    st.markdown(
                        '<div class="section-desc" style="margin-top:10px;">워드클라우드(태그형)</div>',
                        unsafe_allow_html=True,
                    )
                    render_wordcloud_tagcloud(top20, max_words=40)
                    st.markdown("</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">키워드 요약</div>', unsafe_allow_html=True)
                    for i in range(min(5, len(top20))):
                        st.metric(f"{i+1}위", f"{top20[i][0]} ({top20[i][1]:,}회)")
                    st.markdown("</div>", unsafe_allow_html=True)

            # 구간별 키워드 변화(월 단위)
            if has_date and len(df_work.dropna(subset=["작성일"])) > 0 and len(top20) > 0:
                df_month2 = df_work.dropna(subset=["작성일"]).copy()
                df_month2["월"] = df_month2["작성일"].dt.to_period("M").astype(str)

                tracked = [k for k, _ in top20[:12]]
                # 월별 tracked 키워드 카운트
                month_counters = {}
                for month, sub in df_month2.groupby("월"):
                    tokens = []
                    for t in sub["리뷰"].tolist():
                        tokens.extend(simple_tokenize(t))
                    tokens = [w for w in tokens if w not in stopwords]
                    month_counters[month] = Counter(tokens)

                months_sorted = sorted(month_counters.keys())
                if len(months_sorted) >= 2:
                    first_month = months_sorted[0]
                    last_month = months_sorted[-1]

                    for kw in tracked:
                        a = month_counters[first_month].get(kw, 0)
                        b = month_counters[last_month].get(kw, 0)
                        # 무의미한 변화 제거: 변화 폭이 너무 작으면 제외
                        if a == 0 and b == 0:
                            continue
                        if abs(b - a) < 3:
                            continue
                        keyword_changes.append((kw, a, b, b - a))

                    keyword_changes = sorted(keyword_changes, key=lambda x: abs(x[3]), reverse=True)[:8]

            # 변화 결과 표시
            if has_date and keyword_changes:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">구간별 핵심 키워드 변화</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-desc">첫 달 대비 마지막 달에서 빈도 변화가 큰 키워드입니다.</div>', unsafe_allow_html=True)
                ch_df = pd.DataFrame(keyword_changes, columns=["키워드", "첫 달 빈도", "마지막 달 빈도", "변화량"])
                st.dataframe(ch_df, use_container_width=True, height=220)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.caption("키워드 변화(월별)가 뚜렷하지 않거나, 리뷰 작성일 데이터가 부족합니다.")

            # OpenAI 해석(키워드)
            if len(top20) > 0:
                changes_text = ", ".join([f"{k}({a}->{b})" for k, a, b, _ in keyword_changes[:5]]) if keyword_changes else "뚜렷한 변화 없음"
                kw_payload = f"TOP20: {top20[:10]}\n키워드 변화: {changes_text}"
                kw_interpret = call_openai_interpretation("키워드 분석", kw_payload)
                if kw_interpret:
                    st.markdown('<div class="note-box">', unsafe_allow_html=True)
                    st.subheader("OpenAI 해석(키워드)")
                    st.write(kw_interpret)
                    st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # 감성 분석
        # -------------------------
        st.markdown(
            """
<div class="section-card">
  <div class="section-title">4) 감성 분석</div>
  <div class="section-desc">평점 기준으로 positive/neutral/negative 라벨을 만들고 분포를 시각화합니다.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.header("감성 분석")

        # PDF 생성용 기본값(감성 분석 조건에 따라 값이 비어있을 수 있습니다)
        dist_dict = {"positive": 0, "neutral": 0, "negative": 0}
        top_pos = []
        top_neg = []
        neg_in_pos = []
        pos_in_neg = []

        if not has_rating:
            st.warning("`평점` 컬럼이 없어 감성 분석을 수행할 수 없습니다.")
        else:
            df_sent = df_work.dropna(subset=["평점"]).copy()
            if df_sent.empty:
                st.warning("`평점` 값이 비어있어 감성 분석을 할 수 없습니다.")
            else:
                def rating_to_sentiment(x):
                    if x >= 4:
                        return "positive"
                    if x == 3:
                        return "neutral"
                    return "negative"

                df_sent["감성"] = df_sent["평점"].apply(rating_to_sentiment)
                dist = df_sent["감성"].value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
                dist_dict = dist.to_dict()
                dist_df = dist.reset_index()
                dist_df.columns = ["감성", "리뷰 수"]

                st.bar_chart(dist_df.set_index("감성"), height=260)

                s_cols = st.columns(3)
                with s_cols[0]:
                    st.metric("positive", f"{int(dist.get('positive', 0)):,}")
                with s_cols[1]:
                    st.metric("neutral", f"{int(dist.get('neutral', 0)):,}")
                with s_cols[2]:
                    st.metric("negative", f"{int(dist.get('negative', 0)):,}")

                # 감성 키워드(positive/negative 각각) 추출:
                # - OpenAI이 감성의 원인을 UX/기능/성능 관점으로 해석할 수 있게 돕습니다.
                pos_counter = Counter()
                neg_counter = Counter()
                if "리뷰" in df_sent.columns:
                    for txt in df_sent[df_sent["감성"] == "positive"]["리뷰"].tolist():
                        for w in simple_tokenize(txt):
                            if w not in stopwords:
                                pos_counter[w] += 1
                    for txt in df_sent[df_sent["감성"] == "negative"]["리뷰"].tolist():
                        for w in simple_tokenize(txt):
                            if w not in stopwords:
                                neg_counter[w] += 1

                top_pos = pos_counter.most_common(15)
                top_neg = neg_counter.most_common(15)

                # 별점은 높/낮지만 키워드가 반대 방향으로 섞이는 패턴 추출
                # - 높은 별점(positive)인데 negative 키워드가 같이 등장: high_star_contains_negative
                # - 낮은 별점(negative)인데 positive 키워드가 같이 등장: low_star_contains_positive
                neg_in_pos = []
                for w, neg_cnt in top_neg:
                    p_cnt = pos_counter.get(w, 0)
                    if p_cnt > 0:
                        neg_in_pos.append((w, p_cnt, neg_cnt))

                pos_in_neg = []
                for w, pos_cnt in top_pos:
                    n_cnt = neg_counter.get(w, 0)
                    if n_cnt > 0:
                        pos_in_neg.append((w, n_cnt, pos_cnt))

                # OpenAI 해석(감성)
                sent_payload = (
                    f"분포(positive/neutral/negative): {dist.to_dict()}\n"
                    f"positive TOP 키워드: {top_pos}\n"
                    f"negative TOP 키워드: {top_neg}\n"
                    f"높은 별점(positive) 안에 부정 키워드가 섞인 패턴(top): {neg_in_pos[:10]}\n"
                    f"낮은 별점(negative) 안에 긍정 키워드가 섞인 패턴(top): {pos_in_neg[:10]}\n"
                )
                sent_interpret = call_openai_interpretation("감성 분석", sent_payload)
                if sent_interpret:
                    st.markdown('<div class="note-box">', unsafe_allow_html=True)
                    st.subheader("OpenAI 해석(감성)")
                    st.write(sent_interpret)
                    st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # painpoint 분석
        # -------------------------
        st.markdown(
            """
<div class="section-card">
  <div class="section-title">5) Painpoint 분석</div>
  <div class="section-desc">negative 리뷰에서 가장 많이 반복되는 문제를 TOP 10으로 정리하고, AARRR 관점의 이탈 구간을 찾습니다.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.header("Painpoint 분석")

        # AARRR 지표(리뷰 텍스트에서 키워드가 등장하면 해당 단계 이슈로 간주)
        aarrr_rules = {
            "Acquisition(유입)": ["광고", "검색", "스토어", "다운로드", "소개", "유입", "추천", "검색어"],
            "Activation(첫 사용)": ["로그인", "가입", "회원가입", "인증", "처음", "튜토리얼", "권한", "동의", "설정"],
            "Retention(유지/재사용)": ["버그", "오류", "크래시", "튕김", "렉", "느림", "로딩", "불안정", "간헐", "업데이트 후"],
            "Referral(공유/추천)": ["공유", "추천", "친구", "카톡", "링크", "전달", "리뷰", "평가"],
            "Revenue(수익/결제)": ["결제", "구독", "환불", "가격", "결제오류", "구매", "인앱", "해지"],
        }

        # Painpoint 후보 키워드(간단 프리셋)
        pain_indicator = {
            "로그인/인증": ["로그인", "인증", "비밀번호", "아이디", "권한", "동의"],
            "로딩/속도": ["로딩", "느림", "버벅", "렉", "속도", "대기"],
            "오류/크래시": ["오류", "에러", "크래시", "튕김", "문제", "작동", "멈춤"],
            "결제/구독": ["결제", "구독", "환불", "구매", "해지"],
            "업데이트 후": ["업데이트", "이후", "후", "버전", "업데이트후"],
            "권한/설정": ["권한", "설정", "알림", "동의", "허용"],
            "컨텐츠/기능": ["기능", "사용", "검색", "화면", "UI", "버튼", "가이드"],
        }

        # negative 리뷰 우선
        if has_rating:
            df_neg = df_work.dropna(subset=["평점"]).copy()
            df_neg = df_neg[df_neg["평점"] <= 2]
        else:
            df_neg = df_work.copy()

        if not has_review:
            st.warning("`리뷰` 컬럼이 없어 Painpoint를 분석할 수 없습니다.")
        elif df_neg.empty:
            st.caption("negative 리뷰(평점 2 이하)가 없어 Painpoint 분석이 어렵습니다. 전체 리뷰 기준으로 진행합니다.")
            df_neg = df_work.copy()

        # negative 리뷰 텍스트 기반 토큰 TOP 10(문제 후보)
        pain_top10 = []
        if has_review and not df_neg.empty:
            all_neg_tokens = []
            for t in df_neg["리뷰"].tolist():
                all_neg_tokens.extend(simple_tokenize(t))
            all_neg_tokens = [w for w in all_neg_tokens if w not in stopwords]
            pain_counter = Counter(all_neg_tokens)
            pain_top10 = pain_counter.most_common(10)

        if pain_top10:
            # 문제 원인 분석(휴리스틱)
            def guess_cause(word: str):
                for cause_name, indicators in pain_indicator.items():
                    for ind in indicators:
                        if ind in word or word in ind:
                            return f"{cause_name} 관련 이슈 가능"
                return "원인 키워드 기반 추정(추가 확인 권장)"

            pain_rows = []
            for w, cnt in pain_top10:
                pain_rows.append(
                    {
                        "문제 키워드": w,
                        "발생 빈도": cnt,
                        "가능한 원인(간단)": guess_cause(w),
                    }
                )

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">문제 TOP 10</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(pain_rows), use_container_width=True, height=320)
            st.markdown("</div>", unsafe_allow_html=True)

        # AARRR 구간 이탈(리뷰 언급 비중 기반)
        if has_review and not df_neg.empty:
            stage_counts = []
            for stage, inds in aarrr_rules.items():
                hit = 0
                for t in df_neg["리뷰"].tolist():
                    text = str(t)
                    if any(ind in text for ind in inds):
                        hit += 1
                stage_counts.append({"단계": stage, "이슈 언급(negative 리뷰 내)": hit})

            stage_df = pd.DataFrame(stage_counts).sort_values("이슈 언급(negative 리뷰 내)", ascending=False)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">AARRR 관점 핵심 이탈 구간</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-desc">주의: 리뷰 데이터만으로 실제 퍼널 전환율을 계산하긴 어렵습니다. 여기서는 “negative 리뷰에서 해당 단계 키워드 언급이 많을수록 이탈 가능”으로 간주합니다.</div>',
                unsafe_allow_html=True,
            )
            st.bar_chart(stage_df.set_index("단계")["이슈 언급(negative 리뷰 내)"], height=260)
            st.dataframe(stage_df.head(5), use_container_width=True, height=200)
            st.markdown("</div>", unsafe_allow_html=True)

            # OpenAI 해석(페인포인트)
            pain_payload = f"negative 리뷰 토큰 TOP10: {pain_top10}\nAARRR stage: {stage_df.to_dict(orient='records')[:5]}"
            pain_interpret = call_openai_interpretation("Painpoint 분석", pain_payload)
            if pain_interpret:
                st.markdown('<div class="note-box">', unsafe_allow_html=True)
                st.subheader("OpenAI 해석(Painpoint)")
                st.write(pain_interpret)
                st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # Feature request analysis
        # -------------------------
        st.markdown(
            """
<div class="section-card">
  <div class="section-title">6) Feature request analysis</div>
  <div class="section-desc">부정 평가에서 반복되는 기능/개선 요청을 정리하고 우선순위를 추천합니다.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.header("Feature request analysis")

        if not has_review:
            st.warning("`리뷰` 컬럼이 없어 Feature request 분석을 수행할 수 없습니다.")
        else:
            # 부정 리뷰에서 “핵심 기능 후보”를 pain_top10에서 가져오기
            # (실행 안정성을 위해 휴리스틱 우선 + OpenAI 보강)
            feature_candidates = []
            if pain_top10:
                for w, cnt in pain_top10:
                    feature_candidates.append((w, cnt))

            # OpenAI가 우선순위 추천을 더 자연스럽게 생성
            feat_payload = f"부정 페인포인트 TOP10: {pain_top10}\nAARRR 단계 top: (stage_df head 5가 있으면 참고)"
            if "stage_df" in locals():
                feat_payload += f"\nAARRR top5: {stage_df.head(5).to_dict(orient='records')}"

            feat_interpret = call_openai_interpretation("Feature request analysis", feat_payload)

            # UI: 추천 카드
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">우선순위 추천 리스트</div>', unsafe_allow_html=True)
            if feat_interpret:
                st.write(feat_interpret)
            else:
                if feature_candidates:
                    st.info("OpenAI 해석을 표시하려면 `OPENAI_API_KEY` 환경변수를 설정하세요.")
                    st.write("대신, 부정 토큰 TOP10을 기능 후보로 간단히 나열합니다.")
                    feat_df = pd.DataFrame(feature_candidates, columns=["핵심 기능 후보(토큰)", "빈도"])
                    st.dataframe(feat_df.head(10), use_container_width=True, height=260)
                else:
                    st.caption("Feature request 후보를 만들 데이터가 부족합니다.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================================
        # 7) 인사이트
        # ==========================================================
        st.markdown(
            """
<div class="section-card">
  <div class="section-title">7) 인사이트</div>
  <div class="section-desc">분석 결과를 한 번에 요약해서 다음 액션을 정리합니다.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.header("인사이트")

        insights = []
        insights.append(f"전체 리뷰는 **{total_reviews:,}개** 입니다.")
        if avg_rating is not None:
            insights.append(f"평균 평점은 **{avg_rating:.2f}** 로, 전반적인 만족도를 가늠할 수 있습니다.")
        if period_text is not None:
            insights.append(f"리뷰 작성 기간은 **{period_text}** 입니다.")

        # 키워드 요약
        if len(top20) > 0:
            k1, v1 = top20[0]
            insights.append(f"가장 많이 언급된 키워드는 **{k1}**(총 {v1:,}회) 입니다.")

        # painpoint 요약
        if pain_top10:
            pk, pv = pain_top10[0]
            insights.append(f"부정에서 가장 크게 보이는 문제 키워드는 **{pk}** 입니다(총 {pv:,}회).")

        # negative 비율(가능하면)
        if has_rating and not df_work.dropna(subset=["평점"]).empty:
            df_tmp = df_work.dropna(subset=["평점"]).copy()
            df_tmp["감성"] = df_tmp["평점"].apply(lambda x: "positive" if x >= 4 else ("neutral" if x == 3 else "negative"))
            neg_cnt = int((df_tmp["감성"] == "negative").sum())
            insights.append(f"negative 리뷰 비중은 **{neg_cnt:,}개** 입니다(평점 2 이하 기준).")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        for s in insights[:6]:
            st.markdown(f"- {s}")
        st.markdown("</div>", unsafe_allow_html=True)

        # OpenAI 최종 인사이트(짧게)
        final_payload = (
            f"총 리뷰 {total_reviews}, 평균 평점 {avg_rating}, 기간 {period_text}\n"
            f"TOP 키워드 {top20[:5]}\n"
            f"Pain top10 {pain_top10[:5]}\n"
        )
        final_interpret = call_openai_interpretation("최종 인사이트", final_payload)
        if final_interpret:
            st.markdown('<div class="note-box">', unsafe_allow_html=True)
            st.subheader("OpenAI 해석(인사이트)")
            st.write(final_interpret)
            st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # PDF 생성(전용 report.html 템플릿)
        # -------------------------
        report_title = "리뷰 분석 리포트"
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 분석 결과를 dict 형태로 정리
        overview = {
            "total_reviews": total_reviews,
            "avg_rating": "-" if avg_rating is None else f"{avg_rating:.2f}",
            "period_text": "-" if period_text is None else period_text,
            "monthly_reviews": overview_month_counts,
            "change_events": change_notes,
        }

        top_keywords = [{"keyword": k, "count": int(v)} for k, v in top20]
        keyword_changes_rows = []
        for item in keyword_changes:
            # item: (kw, first_count, last_count, delta)
            kw = item[0]
            first_count = item[1]
            last_count = item[2]
            delta = item[3]
            keyword_changes_rows.append(
                {"keyword": kw, "first_count": int(first_count), "last_count": int(last_count), "delta": int(delta)}
            )

        keywords = {
            "top_keywords": top_keywords,
            "keyword_changes": keyword_changes_rows,
            "openai_interpret_text": kw_interpret,
        }

        distribution_rows = []
        for label in ["positive", "neutral", "negative"]:
            distribution_rows.append({"label": label, "count": int(dist_dict.get(label, 0))})

        sentiment = {
            "distribution_rows": distribution_rows,
            "positive_keywords": [{"keyword": k, "count": int(v)} for k, v in top_pos],
            "negative_keywords": [{"keyword": k, "count": int(v)} for k, v in top_neg],
            "high_star_contains_negative": [
                {"keyword": w, "pos_count": int(pos_cnt), "neg_count": int(neg_cnt)}
                for (w, pos_cnt, neg_cnt) in neg_in_pos
            ],
            "low_star_contains_positive": [
                {"keyword": w, "neg_count": int(neg_cnt), "pos_count": int(pos_cnt)}
                for (w, neg_cnt, pos_cnt) in pos_in_neg
            ],
            "openai_interpret_text": sent_interpret,
        }

        top_pain_rows = []
        for row in pain_rows:
            top_pain_rows.append(
                {
                    "keyword": row.get("문제 키워드", ""),
                    "count": int(row.get("발생 빈도", 0)),
                    "cause": row.get("가능한 원인(간단)", ""),
                }
            )

        aarrr_rows = []
        if stage_df is not None and isinstance(stage_df, pd.DataFrame) and not stage_df.empty:
            for _, r in stage_df.head(5).iterrows():
                aarrr_rows.append(
                    {"stage": r.get("단계", ""), "issue_hit_count": int(r.get("이슈 언급(negative 리뷰 내)", 0))}
                )

        painpoints = {
            "top_pain_rows": top_pain_rows,
            "openai_interpret_text": pain_interpret,
            "aarrr_rows": aarrr_rows,
        }

        feature_candidates_rows = [{"keyword": w, "count": int(cnt)} for w, cnt in feature_candidates]
        features = {
            "recommendation_text": feat_interpret,
            "feature_candidates_rows": feature_candidates_rows[:10],
        }

        report = {"insights": insights[:10], "final_openai_interpret_text": final_interpret}

        cache_key = f"{uploaded_file.name}_{total_reviews}_{len(df.columns)}_{int(OPENAI_ENABLED)}"
        if st.session_state.get("pdf_cache_key") != cache_key:
            try:
                template_dir = os.path.dirname(__file__)
                template_path = os.path.join(template_dir, "report.html")
                with open(template_path, "r", encoding="utf-8") as f:
                    template_src = f.read()

                env = Environment(
                    autoescape=select_autoescape(["html", "xml"]),
                )
                template = env.from_string(template_src)

                rendered_html = template.render(
                    report_title=report_title,
                    generated_at=generated_at,
                    overview=overview,
                    keywords=keywords,
                    sentiment=sentiment,
                    painpoints=painpoints,
                    features=features,
                    report=report,
                )

                pdf_bytes = WeasyHTML(string=rendered_html, base_url=template_dir).write_pdf()
                st.session_state["pdf_cache_key"] = cache_key
                st.session_state["pdf_bytes"] = pdf_bytes
            except Exception as e:
                st.error("PDF 생성에 실패했습니다.")
                st.code(str(e))
                pdf_bytes = None
        else:
            pdf_bytes = st.session_state.get("pdf_bytes")

        if pdf_bytes:
            st.markdown(
                """
<div class="section-card">
  <div class="section-title">PDF 다운로드</div>
  <div class="section-desc">웹 UI가 포함되지 않는 문서형 PDF를 생성합니다.</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.download_button(
                label="PDF로 다운로드",
                data=pdf_bytes,
                file_name="review_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
else:
    st.info("먼저 CSV 파일을 업로드해주세요.")

