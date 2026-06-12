# Survey Insight App

Google Forms CSV 설문 데이터 UX Research 인사이트 도출 앱입니다.

## 실행

로컬:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud는 **Main file path**를 `app.py`(저장소 루트)로 두면 됩니다.  
이전 리뷰분석기 앱을 쓰던 경우, 설정에서 Main file path만 바꾸고 **Reboot app** 하면 재연결 없이도 됩니다.

자세한 내용은 [survey-insight-app/README.md](survey-insight-app/README.md)를 참고하세요.

## 배포

이 앱은 메모리 사용량 때문에 Streamlit Community Cloud보다 `Docker 기반 배포`가 더 적합합니다.

### 1. 로컬 Docker 실행

```bash
docker build -t survey-insight-app .
docker run --rm -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  survey-insight-app
```

브라우저에서 `http://localhost:8501`로 접속합니다.

### 2. Render에 GitHub로 배포

이 저장소에는 [render.yaml](/Users/songeonhui/Desktop/대하%E1%86%A8/2026%204-1/%E1%84%83%E1%85%B5%E1%84%8C%E1%85%B5%E1%84%90%E1%85%A5%E1%86%AF%E1%84%8F%E1%85%A9%E1%86%AB%E1%84%90%E1%85%A6%E1%86%AB%E1%84%8E%E1%85%B3%E1%84%83%E1%85%B5%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AB%20%E1%84%90%E1%85%B3%E1%86%A8%E1%84%85%E1%85%A9%E1%86%AB/MJU_cursor/render.yaml)이 포함되어 있습니다.

1. 이 프로젝트를 GitHub에 push
2. Render에서 `New +` → `Blueprint`
3. GitHub 저장소 연결
4. `render.yaml` 인식 후 생성
5. 환경변수 설정

권장 환경변수:

- `OPENAI_API_KEY`
- `HF_KOREA_NEMOTRON_REPO_ID`
- `HF_KOREA_NEMOTRON_FILENAME` 선택
- `HF_KOREA_NEMOTRON_REVISION` 선택

### 3. VPS/클라우드 서버에 Docker로 직접 배포

서버에 Docker만 있으면 됩니다.

```bash
git clone <your-repo>
cd MJU_cursor
docker build -t survey-insight-app .
docker run -d --name survey-insight-app \
  -p 80:8501 \
  -e OPENAI_API_KEY=your_key \
  survey-insight-app
```

도메인을 붙일 때는 보통 Nginx 또는 Caddy를 앞단 reverse proxy로 둡니다.
