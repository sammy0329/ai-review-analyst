"""
AI Review Analyst - Streamlit 대시보드 (쇼핑몰 스타일).

제품 목록 → 제품 상세 → 리뷰 분석/Q&A 형태의 UI를 제공합니다.
"""

import os
import sys
from pathlib import Path

import streamlit as st

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

from src.pipeline.aihub_loader import AIHubDataLoader, Product
from src.pipeline.aspect_extractor import create_aspect_extractor
from src.pipeline.preprocessor import create_default_preprocessor
from src.pipeline.embedder import create_embedder
from src.chains.rag_chain import create_rag_chain


# =============================================================================
# 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="AI Review Analyst",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 세션 상태 초기화
# =============================================================================

def init_session_state():
    """세션 상태 초기화."""
    # 페이지 네비게이션
    if "current_page" not in st.session_state:
        st.session_state.current_page = "product_list"  # product_list or product_detail

    # 제품 목록
    if "products" not in st.session_state:
        st.session_state.products = []

    # 선택된 제품
    if "selected_product" not in st.session_state:
        st.session_state.selected_product = None

    # 채팅 메시지 (제품별)
    if "product_messages" not in st.session_state:
        st.session_state.product_messages = {}  # {product_name: [messages]}

    # RAG Chain (제품별)
    if "product_rag_chain" not in st.session_state:
        st.session_state.product_rag_chain = None

    # 속성 분석 결과 (제품별)
    if "product_aspects" not in st.session_state:
        st.session_state.product_aspects = {}


init_session_state()


# =============================================================================
# 사이드바
# =============================================================================

def render_sidebar():
    """사이드바 렌더링."""
    with st.sidebar:
        st.title("🛒 AI Review Analyst")
        st.markdown("---")

        # API 키 상태 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ OpenAI API 연결됨")
        else:
            st.error("❌ OPENAI_API_KEY 필요")
            st.info("`.env` 파일에 API 키를 설정하세요.")
            return False

        st.markdown("---")

        # 카테고리 필터
        st.subheader("🏷️ 카테고리")
        categories = ["전체", "패션", "화장품", "가전", "IT기기", "생활용품"]
        selected_category = st.selectbox(
            "카테고리 선택",
            categories,
            key="category_filter",
        )

        # 제품 로드 버튼
        if st.button("📦 제품 불러오기", use_container_width=True):
            load_products(selected_category)

        st.markdown("---")

        # 현재 상태 표시
        if st.session_state.products:
            st.info(f"📦 {len(st.session_state.products)}개 제품 로드됨")

        if st.session_state.selected_product:
            st.success(f"📌 {st.session_state.selected_product.name[:20]}...")

        # 홈으로 돌아가기
        if st.session_state.current_page == "product_detail":
            st.markdown("---")
            if st.button("🏠 제품 목록으로", use_container_width=True):
                st.session_state.current_page = "product_list"
                st.session_state.selected_product = None
                st.rerun()

        return True


# =============================================================================
# 제품 로드
# =============================================================================

def load_products(category: str):
    """제품 목록 로드."""
    with st.spinner("📦 제품 로드 중..."):
        try:
            loader = AIHubDataLoader(data_dir="data/aihub_data")

            cat_filter = None if category == "전체" else category

            products = loader.get_products(
                category=cat_filter,
                min_reviews=3,
                limit=None,  # 페이지네이션으로 처리
            )

            st.session_state.products = products
            st.session_state.current_page = "product_list"
            st.success(f"✅ {len(products)}개 제품 로드 완료!")
            st.rerun()

        except Exception as e:
            st.error(f"제품 로드 실패: {e}")


# =============================================================================
# 제품 목록 페이지
# =============================================================================

def render_product_list():
    """제품 목록 페이지 렌더링."""
    st.title("🛒 제품 목록")
    st.markdown("리뷰를 분석하고 싶은 제품을 선택하세요.")

    products = st.session_state.products

    if not products:
        st.info("👈 왼쪽 사이드바에서 제품을 불러오세요.")

        # 빠른 시작 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 화장품 제품 불러오기", use_container_width=True):
                load_products("화장품")
        return

    # 검색 및 정렬
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 제품 검색", placeholder="제품명 검색...")
    with col2:
        sort_option = st.selectbox(
            "정렬",
            ["리뷰 많은순", "평점 높은순", "평점 낮은순"],
        )

    # 필터링 및 정렬
    filtered_products = products
    if search_query:
        filtered_products = [
            p for p in products
            if search_query.lower() in p.name.lower()
        ]

    if sort_option == "리뷰 많은순":
        filtered_products.sort(key=lambda p: p.review_count, reverse=True)
    elif sort_option == "평점 높은순":
        filtered_products.sort(key=lambda p: p.avg_rating, reverse=True)
    elif sort_option == "평점 낮은순":
        filtered_products.sort(key=lambda p: p.avg_rating)

    # 페이지네이션 설정
    products_per_page = 12  # 3열 x 4행
    total_products = len(filtered_products)
    total_pages = max(1, (total_products + products_per_page - 1) // products_per_page)

    # 페이지 상태
    if "product_list_page" not in st.session_state:
        st.session_state.product_list_page = 0

    # 검색/정렬 변경 시 페이지 리셋
    current_page = st.session_state.product_list_page
    if current_page >= total_pages:
        current_page = 0
        st.session_state.product_list_page = 0

    # 페이지네이션 UI (상단)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown(f"**{total_products}개** 제품")
    with col2:
        if total_pages > 1:
            st.markdown(f"<div style='text-align: center;'>{current_page + 1} / {total_pages} 페이지</div>", unsafe_allow_html=True)
    with col3:
        if total_pages > 1:
            nav_cols = st.columns(2)
            with nav_cols[0]:
                if st.button("◀", key="prev_top", disabled=current_page == 0):
                    st.session_state.product_list_page = current_page - 1
                    st.rerun()
            with nav_cols[1]:
                if st.button("▶", key="next_top", disabled=current_page >= total_pages - 1):
                    st.session_state.product_list_page = current_page + 1
                    st.rerun()

    st.markdown("---")

    # 현재 페이지 제품
    start_idx = current_page * products_per_page
    end_idx = min(start_idx + products_per_page, total_products)
    page_products = filtered_products[start_idx:end_idx]

    # 제품 카드 그리드 (3열)
    cols = st.columns(3)

    for i, product in enumerate(page_products):
        with cols[i % 3]:
            render_product_card(product)

    # 페이지네이션 UI (하단)
    if total_pages > 1:
        st.markdown("---")
        bottom_cols = st.columns([1, 2, 1])
        with bottom_cols[1]:
            nav_cols = st.columns([1, 2, 1])
            with nav_cols[0]:
                if st.button("◀ 이전", key="prev_bottom", disabled=current_page == 0):
                    st.session_state.product_list_page = current_page - 1
                    st.rerun()
            with nav_cols[1]:
                st.markdown(f"<div style='text-align: center; padding-top: 5px;'>{current_page + 1} / {total_pages}</div>", unsafe_allow_html=True)
            with nav_cols[2]:
                if st.button("다음 ▶", key="next_bottom", disabled=current_page >= total_pages - 1):
                    st.session_state.product_list_page = current_page + 1
                    st.rerun()


def render_product_card(product: Product):
    """제품 카드 렌더링."""
    with st.container():
        # 카드 스타일
        sentiment_ratio = product.get_sentiment_ratio()
        positive_ratio = sentiment_ratio["긍정"]

        # 감정에 따른 색상
        if positive_ratio >= 70:
            sentiment_color = "🟢"
            sentiment_text = "매우 긍정"
        elif positive_ratio >= 50:
            sentiment_color = "🟡"
            sentiment_text = "보통"
        else:
            sentiment_color = "🔴"
            sentiment_text = "주의"

        # 카드 내용
        st.markdown(f"### 📦 {product.name[:25]}{'...' if len(product.name) > 25 else ''}")
        st.caption(f"{product.category} > {product.main_category}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("평점", f"⭐ {product.avg_rating:.1f}")
        with col2:
            st.metric("리뷰", f"📝 {product.review_count}개")

        st.markdown(f"{sentiment_color} **{sentiment_text}** ({positive_ratio:.0f}% 긍정)")

        # 주요 속성 태그
        if product.top_aspects:
            tags = " ".join([f"`{a}`" for a in product.top_aspects[:3]])
            st.markdown(f"🏷️ {tags}")

        # 상세 보기 버튼
        if st.button("상세 보기", key=f"view_{product.name}", use_container_width=True):
            st.session_state.selected_product = product
            st.session_state.current_page = "product_detail"
            st.rerun()

        st.markdown("---")


# =============================================================================
# 제품 상세 페이지
# =============================================================================

def render_product_detail():
    """제품 상세 페이지 렌더링."""
    product = st.session_state.selected_product

    if not product:
        st.warning("선택된 제품이 없습니다.")
        return

    # 헤더
    st.title(f"📦 {product.name}")
    st.caption(f"{product.category} > {product.main_category}")

    # 요약 메트릭
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("평균 평점", f"⭐ {product.avg_rating:.1f}")

    with col2:
        st.metric("리뷰 수", f"{product.review_count}개")

    with col3:
        sentiment_ratio = product.get_sentiment_ratio()
        st.metric("긍정 비율", f"{sentiment_ratio['긍정']:.0f}%")

    with col4:
        st.metric("부정 비율", f"{sentiment_ratio['부정']:.0f}%")

    st.markdown("---")

    # 탭
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 요약", "🏷️ 속성 분석", "💬 Q&A", "📋 리뷰 목록"
    ])

    with tab1:
        render_product_summary(product)

    with tab2:
        render_product_aspects(product)

    with tab3:
        render_product_qa(product)

    with tab4:
        render_product_reviews(product)


def render_product_summary(product: Product):
    """제품 요약 탭."""
    # 요약 탭 전용 컨테이너
    summary_container = st.container()

    with summary_container:
        st.subheader("📊 리뷰 요약")

        # 감정 분포 차트
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**감정 분포**")
            st.bar_chart(product.sentiment_distribution)

        with col2:
            st.markdown("**주요 언급 속성**")
            if product.top_aspects:
                # 속성별 언급 횟수 계산
                from collections import Counter
                aspect_counter: Counter = Counter()
                for review in product.reviews:
                    for aspect in review.aspects:
                        aspect_name = aspect.get("Aspect", "")
                        if aspect_name:
                            aspect_counter[aspect_name] += 1

                top_5 = dict(aspect_counter.most_common(5))
                st.bar_chart(top_5)
            else:
                st.info("속성 정보가 없습니다.")

        st.markdown("---")

        # 인사이트
        st.subheader("💡 주요 인사이트")

        sentiment_ratio = product.get_sentiment_ratio()

        if sentiment_ratio["긍정"] >= 70:
            st.success(f"✅ 이 제품은 **{sentiment_ratio['긍정']:.0f}%**의 긍정 리뷰를 받고 있습니다.")
        elif sentiment_ratio["부정"] >= 50:
            st.warning(f"⚠️ 이 제품은 **{sentiment_ratio['부정']:.0f}%**의 부정 리뷰가 있어 주의가 필요합니다.")
        else:
            st.info(f"📊 이 제품의 리뷰는 긍정 {sentiment_ratio['긍정']:.0f}%, 부정 {sentiment_ratio['부정']:.0f}%로 혼재되어 있습니다.")

        # 자주 언급되는 속성
        if product.top_aspects:
            st.markdown(f"🏷️ 가장 많이 언급되는 속성: **{', '.join(product.top_aspects[:3])}**")


def highlight_aspect_in_text(full_text: str, aspect_text: str, sentiment: str) -> str:
    """
    전체 리뷰 텍스트에서 속성 관련 부분을 하이라이트.

    Args:
        full_text: 전체 리뷰 텍스트
        aspect_text: 속성 관련 텍스트 (하이라이트할 부분)
        sentiment: 감정 ("긍정", "부정", "중립")

    Returns:
        HTML 형식의 하이라이트된 텍스트
    """
    import html

    # 감정별 스타일
    styles = {
        "긍정": "background-color: #e3f2fd; color: #1565c0; font-weight: bold; padding: 2px 4px; border-radius: 3px;",
        "부정": "background-color: #ffebee; color: #c62828; font-weight: bold; padding: 2px 4px; border-radius: 3px;",
        "중립": "background-color: #e8f5e9; color: #2e7d32; font-weight: bold; padding: 2px 4px; border-radius: 3px;",
    }

    style = styles.get(sentiment, styles["중립"])

    # HTML 이스케이프
    escaped_full = html.escape(full_text)
    escaped_aspect = html.escape(aspect_text)

    # 하이라이트 적용
    if escaped_aspect and escaped_aspect in escaped_full:
        highlighted = escaped_full.replace(
            escaped_aspect,
            f'<span style="{style}">{escaped_aspect}</span>',
            1  # 첫 번째 매칭만
        )
        return highlighted

    return escaped_full


def render_product_aspects(product: Product):
    """속성 분석 탭."""
    st.subheader("🏷️ 속성별 감정 분석")

    # AI Hub 라벨 데이터 활용
    from collections import Counter, defaultdict

    aspect_sentiment: dict[str, dict[str, int]] = defaultdict(lambda: {"긍정": 0, "부정": 0, "중립": 0})

    # 속성별 리뷰 데이터 수집 (전체 리뷰 + 속성 텍스트 + 감정)
    aspect_reviews: dict[str, list[dict]] = defaultdict(list)

    polarity_map = {"1": "긍정", "0": "중립", "-1": "부정", 1: "긍정", 0: "중립", -1: "부정"}

    for review in product.reviews:
        for aspect in review.aspects:
            aspect_name = aspect.get("Aspect", "")
            polarity = aspect.get("SentimentPolarity", 0)
            aspect_text = aspect.get("SentimentText", "")

            if aspect_name:
                sentiment_label = polarity_map.get(polarity, "중립")
                aspect_sentiment[aspect_name][sentiment_label] += 1

                if aspect_text:
                    aspect_reviews[aspect_name].append({
                        "full_text": review.raw_text,
                        "aspect_text": aspect_text,
                        "sentiment": sentiment_label,
                    })

    if not aspect_sentiment:
        st.info("속성 분석 데이터가 없습니다.")
        return

    # 속성별 감정 분포 차트
    import pandas as pd

    df_data = []
    for aspect, sentiments in aspect_sentiment.items():
        total = sum(sentiments.values())
        if total >= 2:  # 최소 2번 이상 언급된 속성만
            df_data.append({
                "속성": aspect,
                "긍정": sentiments["긍정"],
                "부정": sentiments["부정"],
                "중립": sentiments["중립"],
                "총합": total,
            })

    if df_data:
        df = pd.DataFrame(df_data)
        df = df.sort_values("총합", ascending=False).head(10)

        st.markdown("**속성별 감정 분포 (상위 10개)**")
        chart_df = df.set_index("속성")[["긍정", "부정", "중립"]]
        st.bar_chart(chart_df)

        st.markdown("---")

        # 범례 표시
        st.markdown("""
        <div style="margin-bottom: 15px;">
            <span style="background-color: #e3f2fd; color: #1565c0; padding: 3px 8px; border-radius: 3px; margin-right: 10px;">긍정</span>
            <span style="background-color: #ffebee; color: #c62828; padding: 3px 8px; border-radius: 3px; margin-right: 10px;">부정</span>
            <span style="background-color: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 3px;">중립</span>
        </div>
        """, unsafe_allow_html=True)

        # 상세 리뷰
        st.markdown("**속성별 상세 리뷰**")
        for idx, row in df.iterrows():
            aspect = row["속성"]
            total = row["총합"]
            pos_ratio = row["긍정"] / total * 100 if total > 0 else 0

            with st.expander(f"**{aspect}** ({total}회 언급, 긍정 {pos_ratio:.0f}%)"):
                all_reviews = aspect_reviews.get(aspect, [])

                if not all_reviews:
                    st.write("리뷰 텍스트가 없습니다.")
                    continue

                # 감정 필터 토글
                filter_cols = st.columns(3)
                with filter_cols[0]:
                    show_positive = st.toggle("😊 긍정", value=True, key=f"pos_{aspect}")
                with filter_cols[1]:
                    show_negative = st.toggle("😞 부정", value=True, key=f"neg_{aspect}")
                with filter_cols[2]:
                    show_neutral = st.toggle("😐 중립", value=True, key=f"neu_{aspect}")

                # 선택된 감정만 필터링
                selected_sentiments = []
                if show_positive:
                    selected_sentiments.append("긍정")
                if show_negative:
                    selected_sentiments.append("부정")
                if show_neutral:
                    selected_sentiments.append("중립")

                filtered_reviews = [r for r in all_reviews if r["sentiment"] in selected_sentiments]

                if not filtered_reviews:
                    st.info("선택한 감정의 리뷰가 없습니다.")
                    continue

                # 페이지네이션 설정
                reviews_per_page = 10
                total_reviews = len(filtered_reviews)
                total_pages = (total_reviews + reviews_per_page - 1) // reviews_per_page

                # 페이지 상태 키
                page_key = f"page_{aspect}"
                if page_key not in st.session_state:
                    st.session_state[page_key] = 0

                current_page = st.session_state[page_key]

                # 페이지네이션 UI
                st.caption(f"총 {total_reviews}개 리뷰")

                if total_pages > 1:
                    page_cols = st.columns([1, 2, 1])
                    with page_cols[0]:
                        if st.button("◀ 이전", key=f"prev_{aspect}", disabled=current_page == 0):
                            st.session_state[page_key] = current_page - 1
                            st.rerun()
                    with page_cols[1]:
                        st.markdown(f"<div style='text-align: center;'>{current_page + 1} / {total_pages} 페이지</div>", unsafe_allow_html=True)
                    with page_cols[2]:
                        if st.button("다음 ▶", key=f"next_{aspect}", disabled=current_page >= total_pages - 1):
                            st.session_state[page_key] = current_page + 1
                            st.rerun()

                # 현재 페이지 리뷰
                start_idx = current_page * reviews_per_page
                end_idx = min(start_idx + reviews_per_page, total_reviews)
                page_reviews = filtered_reviews[start_idx:end_idx]

                for i, review_data in enumerate(page_reviews):
                    highlighted_html = highlight_aspect_in_text(
                        review_data["full_text"],
                        review_data["aspect_text"],
                        review_data["sentiment"]
                    )

                    # 감정 아이콘
                    emoji = {"긍정": "😊", "부정": "😞", "중립": "😐"}.get(review_data["sentiment"], "")

                    st.markdown(
                        f'<div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {"#1565c0" if review_data["sentiment"] == "긍정" else "#c62828" if review_data["sentiment"] == "부정" else "#2e7d32"};">'
                        f'<span style="font-size: 0.85em; color: #666;">{emoji} {review_data["sentiment"]}</span><br>'
                        f'<span style="line-height: 1.6;">{highlighted_html}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )


def render_product_qa(product: Product):
    """Q&A 탭."""
    st.subheader("💬 이 제품에 대해 질문하세요")

    product_name = product.name

    # 메시지 히스토리 초기화
    if product_name not in st.session_state.product_messages:
        st.session_state.product_messages[product_name] = []

    # RAG Chain 초기화 (필요시)
    if st.session_state.product_rag_chain is None or st.session_state.get("current_rag_product") != product_name:
        with st.spinner("🔧 Q&A 시스템 초기화 중..."):
            try:
                # 리뷰를 Review 형식으로 변환
                reviews = [r.to_review() for r in product.reviews]

                # 전처리
                preprocessor = create_default_preprocessor(chunk_size=300)
                processed = preprocessor.process_batch(reviews)

                # 벡터 DB
                embedder = create_embedder(
                    collection_name=f"product_{hash(product_name) % 10000}",
                    persist_directory="./data/chroma_db_products",
                )
                embedder.reset_collection()
                embedder.add_reviews(processed, show_progress=False)

                # RAG Chain
                rag_chain = create_rag_chain(
                    embedder=embedder,
                    model_name="gpt-4o-mini",
                    top_k=5,
                )

                st.session_state.product_rag_chain = rag_chain
                st.session_state.current_rag_product = product_name

            except Exception as e:
                st.error(f"Q&A 시스템 초기화 실패: {e}")
                return

    # 예시 질문
    with st.expander("💡 예시 질문", expanded=False):
        example_questions = [
            "이 제품의 장점은 무엇인가요?",
            "단점이나 주의할 점은?",
            "품질은 어떤가요?",
            "가격 대비 만족도는?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(q, key=f"example_{product_name}_{i}", use_container_width=True):
                    st.session_state.product_messages[product_name].append({
                        "role": "user",
                        "content": q,
                    })
                    st.rerun()

    # 메시지 히스토리 표시
    messages = st.session_state.product_messages[product_name]
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("이 제품에 대해 질문하세요..."):
        # 사용자 메시지 추가
        messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    rag_chain = st.session_state.product_rag_chain
                    result = rag_chain.query_with_sources(prompt)

                    st.markdown(result["answer"])

                    # 출처 표시
                    if result["sources"]:
                        with st.expander("📚 참조 리뷰"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"**[{i}]**")
                                st.markdown(f"> {source['text'][:200]}...")
                                st.markdown("---")

                    # 메시지 저장
                    messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                    })

                except Exception as e:
                    st.error(f"오류 발생: {e}")


def render_product_reviews(product: Product):
    """리뷰 목록 탭."""
    st.subheader("📋 리뷰 목록")

    reviews = product.reviews

    # 필터
    col1, col2 = st.columns(2)
    with col1:
        sentiment_filter = st.selectbox(
            "감정 필터",
            ["전체", "긍정", "중립", "부정"],
            key=f"sentiment_filter_{product.name}",
        )
    with col2:
        sort_option = st.selectbox(
            "정렬",
            ["최신순", "긍정순", "부정순"],
            key=f"sort_option_{product.name}",
        )

    # 필터링
    polarity_map = {1: "긍정", 0: "중립", -1: "부정"}
    filtered_reviews = reviews

    if sentiment_filter != "전체":
        filtered_reviews = [
            r for r in reviews
            if polarity_map.get(r.general_polarity, "중립") == sentiment_filter
        ]

    # 정렬
    if sort_option == "긍정순":
        filtered_reviews.sort(key=lambda r: r.general_polarity, reverse=True)
    elif sort_option == "부정순":
        filtered_reviews.sort(key=lambda r: r.general_polarity)

    st.markdown(f"**{len(filtered_reviews)}개** 리뷰")
    st.markdown("---")

    # 리뷰 표시
    for i, review in enumerate(filtered_reviews[:20]):  # 최대 20개
        polarity_label = polarity_map.get(review.general_polarity, "중립")
        emoji = {"긍정": "😊", "중립": "😐", "부정": "😞"}.get(polarity_label, "❓")

        with st.expander(f"{emoji} 리뷰 {i+1}: {review.raw_text[:50]}...", expanded=False):
            st.markdown(review.raw_text)

            st.markdown("---")

            # 속성 정보
            if review.aspects:
                st.markdown("**언급된 속성:**")
                for aspect in review.aspects:
                    aspect_name = aspect.get("Aspect", "")
                    aspect_polarity = aspect.get("SentimentPolarity", 0)
                    aspect_text = aspect.get("SentimentText", "")

                    a_label = polarity_map.get(int(aspect_polarity) if isinstance(aspect_polarity, str) else aspect_polarity, "중립")
                    a_emoji = {"긍정": "👍", "중립": "➖", "부정": "👎"}.get(a_label, "❓")

                    st.markdown(f"- {a_emoji} **{aspect_name}**: {aspect_text[:100]}...")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 함수."""
    # 사이드바 렌더링
    api_available = render_sidebar()

    if not api_available:
        st.error("OpenAI API 키가 필요합니다.")
        st.stop()

    # 페이지 라우팅 - 컨테이너로 격리하여 렌더링 충돌 방지
    page_container = st.container()

    with page_container:
        if st.session_state.current_page == "product_list":
            render_product_list()
        elif st.session_state.current_page == "product_detail":
            render_product_detail()


if __name__ == "__main__":
    main()
