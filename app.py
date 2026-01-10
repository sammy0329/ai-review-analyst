"""
AI Review Analyst - Streamlit Application Entry Point
"""

import streamlit as st

st.set_page_config(
    page_title="AI Review Analyst",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("🛒 AI Review Analyst")
    st.markdown(
        "> **리뷰를 읽는 시간 30분을 30초로 단축하다.**"
    )
    st.markdown("---")

    # URL Input
    url = st.text_input(
        "분석할 상품 URL을 입력하세요",
        placeholder="https://www.coupang.com/...",
    )

    if st.button("분석 시작", type="primary"):
        if not url:
            st.warning("URL을 입력해주세요.")
        else:
            st.info("🚧 분석 기능은 개발 중입니다.")

    st.markdown("---")
    st.caption("Built with ❤️ for Levit AI Agent Internship")


if __name__ == "__main__":
    main()
