import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI 연결 테스트", layout="wide")
st.title("AI 연결 테스트")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if st.button("AI 연결 테스트"):
    try:
        response = client.responses.create(
            model="gpt-5",
            input="연결 테스트입니다. 한 줄로 짧게 답해주세요."
        )
        st.success("연결 성공")
        st.write(response.output_text)
    except Exception as e:
        st.error(f"오류 발생: {e}")