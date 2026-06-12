# Survey Insight App

Google Forms CSV 설문 응답을 업로드하면 문항 구조 파악, 데이터 정제, 정량·정성 분석, 가설 검토, 인사이트 도출까지 수행하는 **UX Research 보조 Streamlit 앱**입니다.

## 실행 방법

```bash
cd survey-insight-app
pip install -r requirements.txt
streamlit run app.py
```

## 주요 기능

1. CSV 업로드 및 데이터 미리보기
2. 문항 유형 자동 분류
3. 데이터 정제 및 `cleaning_log.csv` 생성
4. 정량·교차 분석
5. 정성·키워드·텍스트 구조·감성 분석
6. 사용자 여정 매핑
7. 가설 기반 분석
8. Fact / Interpretation / Action 인사이트
9. HF Persona DB 기반 유사 사용자 추천 (가중치 + 유사도 랭킹)
10. Virtual IDI / Insight Validation (OpenAI acting)
11. HTML 리포트 (PDF는 WeasyPrint 설치 시)

## 샘플 데이터

`data/sample_survey.csv`를 업로드하여 동작을 확인할 수 있습니다.

HF Korea Nemotron persona DB는 세 가지 방식으로 준비할 수 있습니다.

1. 탭1에서 `.db/.sqlite` 파일 업로드
2. `data/personas.db` 경로에 미리 두기
3. 탭1의 `Persona DB 소스`에서 preset 선택 또는 `직접 입력`

배포 환경에서 아래 값을 넣어두면 사용자는 `Korea Nemotron Persona DB` preset만 선택하면 됩니다.

- `HF_KOREA_NEMOTRON_REPO_ID`
- `HF_KOREA_NEMOTRON_FILENAME` (선택)
- `HF_KOREA_NEMOTRON_REVISION` (선택)

직접 입력이 필요할 때 `HF repo id`는 보통 `조직또는사용자명/리포지토리명` 형태입니다.
예: `my-org/korean-persona-db`

공개 HF repo는 `HF Token` 없이도 DB 다운로드가 가능할 수 있습니다.
비공개/gated repo일 때만 `HF Token`을 넣으면 됩니다.

## 출력 파일 (`outputs/`)

- `cleaned_data.csv`
- `cleaning_log.csv`
- `quantitative_analysis.csv`
- `qualitative_analysis.csv`
- `keyword_analysis.csv`
- `sentiment_analysis.csv`
- `journey_mapping.csv`
- `hypothesis_analysis.csv`
- `persona_matches.csv`
- `virtual_idi.csv`
- `insight_validation.csv`
- `insight_report.html`
- `insight_report.pdf` (옵션)

## 프로젝트 구조

```
survey-insight-app/
├── app.py
├── requirements.txt
├── data/sample_survey.csv
├── modules/          # 분석 모듈
├── prompts/          # LLM 프롬프트 (확장용)
├── templates/        # HTML 리포트 템플릿
└── outputs/          # 생성 결과물
```

## PDF (옵션)

WeasyPrint는 시스템 의존성이 필요할 수 있습니다. 설치되지 않은 환경에서도 HTML 리포트와 CSV 다운로드는 정상 동작합니다.

```bash
pip install weasyprint
```
