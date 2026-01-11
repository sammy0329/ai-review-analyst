"""
AI Review Analyst - Streamlit 대시보드.

리뷰 분석 및 RAG 기반 Q&A 인터페이스를 제공합니다.
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

from src.crawler.base import Review
from src.pipeline.preprocessor import create_default_preprocessor
from src.pipeline.embedder import create_embedder
from src.pipeline.aihub_loader import AIHubDataLoader
from src.pipeline.aspect_extractor import create_aspect_extractor, AspectResult
from src.chains.rag_chain import create_rag_chain


# =============================================================================
# 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="AI Review Analyst",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 세션 상태 초기화
# =============================================================================

def init_session_state():
    """세션 상태 초기화."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "reviews_loaded" not in st.session_state:
        st.session_state.reviews_loaded = False

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if "embedder" not in st.session_state:
        st.session_state.embedder = None

    if "review_stats" not in st.session_state:
        st.session_state.review_stats = None

    if "processed_reviews" not in st.session_state:
        st.session_state.processed_reviews = []

    # 속성 분석 관련
    if "aspect_results" not in st.session_state:
        st.session_state.aspect_results = []

    if "aspect_stats" not in st.session_state:
        st.session_state.aspect_stats = None

    if "raw_reviews" not in st.session_state:
        st.session_state.raw_reviews = []

    # 제품 비교용 데이터
    if "comparison_data" not in st.session_state:
        st.session_state.comparison_data = {}  # {category: {stats, results}}


init_session_state()


# =============================================================================
# 사이드바
# =============================================================================

def render_sidebar():
    """사이드바 렌더링."""
    with st.sidebar:
        st.title("🔍 AI Review Analyst")
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

        # 데이터 로드 섹션
        st.subheader("📊 데이터 로드")

        # 카테고리 선택
        categories = ["패션", "화장품", "가전", "IT기기", "생활용품"]
        selected_category = st.selectbox(
            "카테고리 선택",
            categories,
            index=0,
        )

        # 샘플 수 선택
        sample_size = st.slider(
            "샘플 수",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
        )

        # 로드 버튼
        if st.button("📥 데이터 로드", use_container_width=True):
            load_reviews(selected_category, sample_size)

        # 로드 상태 표시
        if st.session_state.reviews_loaded:
            stats = st.session_state.review_stats
            st.success(f"✅ {stats['total']}개 리뷰 로드됨")
            st.metric("평균 평점", f"{stats['avg_rating']:.1f}")

        st.markdown("---")

        # 설정 섹션
        st.subheader("⚙️ 설정")

        # 모델 선택
        model_name = st.selectbox(
            "LLM 모델",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )

        # Top-K 설정
        top_k = st.slider(
            "검색 문서 수 (Top-K)",
            min_value=1,
            max_value=10,
            value=5,
        )

        # 설정 적용
        if st.session_state.rag_chain is not None:
            st.session_state.rag_chain.update_config(
                model_name=model_name,
                top_k=top_k,
            )

        st.markdown("---")

        # 초기화 버튼
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        return True


# =============================================================================
# 데이터 로드
# =============================================================================

def load_reviews(category: str, sample_size: int):
    """리뷰 데이터 로드."""
    with st.spinner(f"📊 {category} 리뷰 로드 중..."):
        try:
            # AI Hub 데이터 로더
            loader = AIHubDataLoader(data_dir="data/aihub_data")

            # 리뷰 로드 (as_project_format=True로 Review 객체 반환)
            reviews = loader.load_reviews(
                category=category,
                limit=sample_size,
                as_project_format=True,
            )

            if not reviews:
                st.error("리뷰를 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
                reviews = _get_sample_reviews()

            # 원본 리뷰 저장 (속성 분석용)
            st.session_state.raw_reviews = reviews

            # 전처리
            with st.spinner("🔧 전처리 중..."):
                preprocessor = create_default_preprocessor(chunk_size=300)
                processed = preprocessor.process_batch(reviews)
                st.session_state.processed_reviews = processed

            # 벡터 DB에 저장
            with st.spinner("💾 벡터 DB 저장 중..."):
                embedder = create_embedder(
                    collection_name="streamlit_reviews",
                    persist_directory="./data/chroma_db_streamlit",
                )
                embedder.reset_collection()
                embedder.add_reviews(processed, show_progress=False)
                st.session_state.embedder = embedder

            # RAG Chain 생성
            with st.spinner("🔗 RAG Chain 초기화 중..."):
                rag_chain = create_rag_chain(
                    embedder=embedder,
                    model_name="gpt-4o-mini",
                    top_k=5,
                )
                st.session_state.rag_chain = rag_chain

            # 통계 계산
            ratings = [r.rating for r in reviews if r.rating]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0

            st.session_state.review_stats = {
                "total": len(reviews),
                "avg_rating": avg_rating,
                "category": category,
            }
            st.session_state.reviews_loaded = True

            st.success(f"✅ {len(reviews)}개 리뷰 로드 완료!")
            st.rerun()

        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            # 샘플 데이터로 폴백
            _load_sample_data()


def _get_sample_reviews():
    """샘플 리뷰 데이터 (Review 객체 리스트)."""
    return [
        Review(
            text="이 제품 정말 좋아요! 배송도 빠르고 품질도 훌륭합니다. 가격 대비 만족스럽습니다.",
            rating=5.0,
            date="2024-01-15",
        ),
        Review(
            text="배송은 빨랐는데 제품 품질이 기대에 못 미치네요. 가격이 좀 아깝습니다.",
            rating=2.0,
            date="2024-01-14",
        ),
        Review(
            text="무난한 제품입니다. 특별히 좋지도 나쁘지도 않아요. 그냥 평범합니다.",
            rating=3.0,
            date="2024-01-13",
        ),
        Review(
            text="배송이 정말 빨라서 놀랐어요! 주문 다음날 도착했습니다. 제품도 괜찮네요.",
            rating=4.0,
            date="2024-01-12",
        ),
        Review(
            text="사이즈가 생각보다 작아요. 교환하려니 배송비가 아까워서 그냥 씁니다.",
            rating=2.5,
            date="2024-01-11",
        ),
        Review(
            text="디자인이 예쁘고 색상도 마음에 들어요. 친구들한테 추천했습니다!",
            rating=5.0,
            date="2024-01-10",
        ),
        Review(
            text="가성비 최고입니다. 이 가격에 이 정도면 정말 만족스러워요.",
            rating=4.5,
            date="2024-01-09",
        ),
        Review(
            text="포장이 꼼꼼하게 되어 왔어요. 제품 상태도 완벽합니다.",
            rating=5.0,
            date="2024-01-08",
        ),
    ]


def _load_sample_data():
    """샘플 데이터로 초기화."""
    reviews = _get_sample_reviews()

    # 원본 리뷰 저장 (속성 분석용)
    st.session_state.raw_reviews = reviews

    # 전처리
    preprocessor = create_default_preprocessor(chunk_size=300)
    processed = preprocessor.process_batch(reviews)
    st.session_state.processed_reviews = processed

    # 벡터 DB
    embedder = create_embedder(
        collection_name="streamlit_sample",
        persist_directory="./data/chroma_db_streamlit",
    )
    embedder.reset_collection()
    embedder.add_reviews(processed, show_progress=False)
    st.session_state.embedder = embedder

    # RAG Chain
    rag_chain = create_rag_chain(embedder=embedder)
    st.session_state.rag_chain = rag_chain

    # 통계
    ratings = [r.rating for r in reviews if r.rating]
    st.session_state.review_stats = {
        "total": len(reviews),
        "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
        "category": "샘플",
    }
    st.session_state.reviews_loaded = True


# =============================================================================
# 메인 컨텐츠
# =============================================================================

def render_main_content():
    """메인 컨텐츠 렌더링."""
    # 헤더
    st.title("🔍 AI Review Analyst")
    st.markdown("리뷰 데이터를 분석하고 질문에 답변하는 AI 어시스턴트입니다.")

    if not st.session_state.reviews_loaded:
        # 데이터 로드 안내
        st.info("👈 왼쪽 사이드바에서 데이터를 먼저 로드해주세요.")

        # 샘플 로드 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 샘플 데이터로 시작하기", use_container_width=True):
                with st.spinner("샘플 데이터 로드 중..."):
                    _load_sample_data()
                    st.rerun()
        return

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 채팅", "📊 분석", "🏷️ 속성 분석", "⚖️ 제품 비교", "📋 리뷰 목록"
    ])

    with tab1:
        render_chat_interface()

    with tab2:
        render_analysis_tab()

    with tab3:
        render_aspect_analysis_tab()

    with tab4:
        render_product_comparison_tab()

    with tab5:
        render_reviews_tab()


def render_chat_interface():
    """채팅 인터페이스 렌더링."""
    st.subheader("💬 리뷰 기반 Q&A")

    # 예시 질문
    with st.expander("💡 예시 질문", expanded=False):
        example_questions = [
            "배송이 빠른가요?",
            "품질은 어떤가요?",
            "가격 대비 가치가 있나요?",
            "이 제품을 추천하시나요?",
            "주요 장점은 무엇인가요?",
            "주요 단점은 무엇인가요?",
        ]
        cols = st.columns(3)
        for i, q in enumerate(example_questions):
            with cols[i % 3]:
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()

    # 메시지 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 출처 표시 (assistant 메시지에만)
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 참조 리뷰"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**[{i}]** (평점: {source.get('rating', 'N/A')})")
                        st.markdown(f"> {source['text'][:200]}...")
                        st.markdown("---")

    # 사용자 입력
    if prompt := st.chat_input("리뷰에 대해 질문하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    rag_chain = st.session_state.rag_chain
                    result = rag_chain.query_with_sources(prompt)

                    # 스트리밍 효과
                    response_placeholder = st.empty()
                    response_placeholder.markdown(result["answer"])

                    # 출처 표시
                    if result["sources"]:
                        with st.expander("📚 참조 리뷰"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"**[{i}]** (평점: {source.get('rating', 'N/A')})")
                                st.markdown(f"> {source['text'][:200]}...")
                                st.markdown("---")

                    # 메시지 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })

                except Exception as e:
                    st.error(f"오류 발생: {e}")


def render_analysis_tab():
    """분석 탭 렌더링."""
    st.subheader("📊 리뷰 분석")

    stats = st.session_state.review_stats
    if not stats:
        st.warning("데이터를 먼저 로드해주세요.")
        return

    # 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("총 리뷰 수", f"{stats['total']}개")

    with col2:
        st.metric("평균 평점", f"{stats['avg_rating']:.1f}점")

    with col3:
        st.metric("카테고리", stats['category'])

    with col4:
        # 긍정 비율 계산
        processed = st.session_state.processed_reviews
        positive = sum(1 for r in processed if r.metadata.get('rating', 0) >= 4)
        positive_rate = (positive / len(processed) * 100) if processed else 0
        st.metric("긍정 비율", f"{positive_rate:.0f}%")

    st.markdown("---")

    # 평점 분포 차트
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("평점 분포")
        processed = st.session_state.processed_reviews
        if processed:
            ratings = [r.metadata.get('rating', 0) for r in processed if r.metadata.get('rating')]

            # 평점별 카운트
            rating_counts = {}
            for r in ratings:
                key = f"{int(r)}점" if r == int(r) else f"{r:.1f}점"
                rating_counts[key] = rating_counts.get(key, 0) + 1

            st.bar_chart(rating_counts)

    with col2:
        st.subheader("감성 분포")
        if processed:
            # 감성 분류 (평점 기반)
            sentiment_counts = {"긍정": 0, "중립": 0, "부정": 0}
            for r in processed:
                rating = r.metadata.get('rating', 3)
                if rating >= 4:
                    sentiment_counts["긍정"] += 1
                elif rating >= 3:
                    sentiment_counts["중립"] += 1
                else:
                    sentiment_counts["부정"] += 1

            # 파이 차트 대신 바 차트 사용
            st.bar_chart(sentiment_counts)


def render_aspect_analysis_tab():
    """속성 분석 탭 렌더링."""
    st.subheader("🏷️ 속성 분석 (Aspect-based Analysis)")
    st.markdown("LLM을 활용하여 리뷰에서 속성(가격, 디자인, 품질 등)을 자동 추출하고 감정을 분석합니다.")

    raw_reviews = st.session_state.raw_reviews
    if not raw_reviews:
        st.warning("데이터를 먼저 로드해주세요.")
        return

    # 속성 분석 실행 섹션
    st.markdown("---")
    col1, col2 = st.columns([3, 1])

    with col1:
        analysis_count = st.slider(
            "분석할 리뷰 수",
            min_value=5,
            max_value=min(50, len(raw_reviews)),
            value=min(10, len(raw_reviews)),
            step=5,
            help="LLM API 호출 비용을 고려하여 적절한 수를 선택하세요."
        )

    with col2:
        analyze_button = st.button("🔍 속성 분석 실행", use_container_width=True)

    # 속성 분석 실행
    if analyze_button:
        with st.spinner(f"🔍 {analysis_count}개 리뷰 속성 분석 중..."):
            try:
                extractor = create_aspect_extractor(use_cache=True)

                # 리뷰 텍스트 추출
                review_texts = [
                    {"text": r.text, "metadata": {"rating": r.rating, "date": r.date}}
                    for r in raw_reviews[:analysis_count]
                ]

                # 진행 상황 표시
                progress_bar = st.progress(0)
                results = []

                for i, review_data in enumerate(review_texts):
                    result = extractor.extract(
                        review_data["text"],
                        metadata=review_data["metadata"]
                    )
                    results.append(result)
                    progress_bar.progress((i + 1) / len(review_texts))

                # 결과 저장
                st.session_state.aspect_results = results
                st.session_state.aspect_stats = extractor.get_aspect_statistics(results)

                st.success(f"✅ {len(results)}개 리뷰 속성 분석 완료!")
                st.rerun()

            except Exception as e:
                st.error(f"속성 분석 실패: {e}")

    # 결과 표시
    if st.session_state.aspect_stats:
        stats = st.session_state.aspect_stats
        results = st.session_state.aspect_results

        st.markdown("---")

        # 메트릭 카드
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("분석된 리뷰", f"{stats['total_reviews']}개")

        with col2:
            st.metric("평균 신뢰도", f"{stats['avg_confidence']:.0%}")

        with col3:
            positive = stats['overall_sentiment']['positive']
            total = stats['total_reviews']
            st.metric("긍정 비율", f"{positive/total*100:.0f}%" if total > 0 else "0%")

        with col4:
            st.metric("추출된 속성", f"{sum(stats['aspect_counts'].values())}개")

        st.markdown("---")

        # 차트 영역
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 속성별 언급 빈도")
            if stats['aspect_counts']:
                # 상위 8개만 표시
                aspect_data = dict(list(stats['aspect_counts'].items())[:8])
                st.bar_chart(aspect_data)
            else:
                st.info("추출된 속성이 없습니다.")

        with col2:
            st.subheader("🎭 속성별 감정 분포")
            if stats['aspect_sentiment']:
                # 속성별 긍정/부정 비율 차트
                sentiment_data = {}
                for category, sent in list(stats['aspect_sentiment'].items())[:6]:
                    total = sent['positive'] + sent['negative'] + sent['neutral']
                    if total > 0:
                        sentiment_data[category] = {
                            "긍정": sent['positive'],
                            "부정": sent['negative'],
                        }

                if sentiment_data:
                    import pandas as pd
                    df = pd.DataFrame(sentiment_data).T
                    st.bar_chart(df)
            else:
                st.info("감정 분석 데이터가 없습니다.")

        st.markdown("---")

        # 상세 결과 (필터링 가능)
        st.subheader("📋 상세 분석 결과")

        # 필터
        col1, col2 = st.columns(2)
        with col1:
            category_options = ["전체"] + list(stats['aspect_counts'].keys())
            selected_category = st.selectbox("속성 필터", category_options)

        with col2:
            sentiment_options = ["전체", "긍정", "부정", "중립"]
            selected_sentiment = st.selectbox("감정 필터", sentiment_options)

        # 결과 표시
        for i, result in enumerate(results[:20]):  # 최대 20개
            # 필터 적용
            filtered_aspects = result.aspects

            if selected_category != "전체":
                filtered_aspects = [a for a in filtered_aspects if a["category"] == selected_category]

            if selected_sentiment != "전체":
                sentiment_map = {"긍정": "positive", "부정": "negative", "중립": "neutral"}
                filtered_aspects = [a for a in filtered_aspects if a["sentiment"] == sentiment_map[selected_sentiment]]

            if not filtered_aspects and selected_category != "전체":
                continue

            with st.expander(f"리뷰 {i+1}: {result.review_text[:50]}...", expanded=False):
                col1, col2 = st.columns([1, 3])

                with col1:
                    sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                    st.markdown(f"**전체 감정:** {sentiment_emoji.get(result.overall_sentiment.value, '❓')} {result.overall_sentiment.value}")
                    st.markdown(f"**신뢰도:** {result.confidence:.0%}")
                    if result.metadata.get("rating"):
                        st.markdown(f"**평점:** ⭐ {result.metadata['rating']}")

                with col2:
                    st.markdown("**원문:**")
                    st.markdown(f"> {result.review_text[:200]}{'...' if len(result.review_text) > 200 else ''}")

                if filtered_aspects:
                    st.markdown("**추출된 속성:**")
                    for asp in filtered_aspects:
                        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(asp["sentiment"], "❓")
                        st.markdown(f"- {emoji} **[{asp['category']}]** {asp['sentiment']}")
                        st.markdown(f"  > \"{asp['text']}\"")


def render_product_comparison_tab():
    """제품 비교 탭 렌더링."""
    st.subheader("⚖️ 제품/카테고리 비교")
    st.markdown("여러 카테고리의 리뷰 속성을 비교 분석합니다.")

    # 카테고리 목록
    categories = ["패션", "화장품", "가전", "IT기기", "생활용품"]

    st.markdown("---")

    # 비교할 카테고리 선택
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**카테고리 A**")
        category_a = st.selectbox(
            "첫 번째 카테고리",
            categories,
            key="compare_cat_a",
            index=0,
        )
        sample_a = st.slider(
            "분석할 리뷰 수 (A)",
            min_value=5,
            max_value=30,
            value=10,
            key="sample_a",
        )

    with col2:
        st.markdown("**카테고리 B**")
        category_b = st.selectbox(
            "두 번째 카테고리",
            categories,
            key="compare_cat_b",
            index=1 if len(categories) > 1 else 0,
        )
        sample_b = st.slider(
            "분석할 리뷰 수 (B)",
            min_value=5,
            max_value=30,
            value=10,
            key="sample_b",
        )

    # 비교 분석 실행
    if st.button("🔍 비교 분석 실행", use_container_width=True):
        if category_a == category_b:
            st.warning("서로 다른 카테고리를 선택해주세요.")
            return

        with st.spinner("비교 분석 중..."):
            try:
                from src.pipeline.aihub_loader import AIHubDataLoader

                loader = AIHubDataLoader(data_dir="data/aihub_data")
                extractor = create_aspect_extractor(use_cache=True)

                comparison_data = {}

                # 카테고리 A 분석
                progress = st.progress(0, text=f"{category_a} 리뷰 로드 중...")
                reviews_a = loader.load_reviews(
                    category=category_a,
                    limit=sample_a,
                    as_project_format=True,
                )

                if reviews_a:
                    progress.progress(25, text=f"{category_a} 속성 추출 중...")
                    results_a = []
                    for i, r in enumerate(reviews_a):
                        result = extractor.extract(r.text, {"rating": r.rating})
                        results_a.append(result)
                        progress.progress(25 + int(25 * (i + 1) / len(reviews_a)))

                    comparison_data[category_a] = {
                        "results": results_a,
                        "stats": extractor.get_aspect_statistics(results_a),
                    }

                # 카테고리 B 분석
                progress.progress(50, text=f"{category_b} 리뷰 로드 중...")
                reviews_b = loader.load_reviews(
                    category=category_b,
                    limit=sample_b,
                    as_project_format=True,
                )

                if reviews_b:
                    progress.progress(75, text=f"{category_b} 속성 추출 중...")
                    results_b = []
                    for i, r in enumerate(reviews_b):
                        result = extractor.extract(r.text, {"rating": r.rating})
                        results_b.append(result)
                        progress.progress(75 + int(25 * (i + 1) / len(reviews_b)))

                    comparison_data[category_b] = {
                        "results": results_b,
                        "stats": extractor.get_aspect_statistics(results_b),
                    }

                progress.progress(100, text="완료!")

                # 결과 저장
                st.session_state.comparison_data = comparison_data
                st.success("비교 분석 완료!")
                st.rerun()

            except Exception as e:
                st.error(f"비교 분석 실패: {e}")

    # 비교 결과 표시
    if st.session_state.comparison_data and len(st.session_state.comparison_data) >= 2:
        comparison_data = st.session_state.comparison_data
        cats = list(comparison_data.keys())

        st.markdown("---")
        st.subheader("📊 비교 분석 결과")

        # 메트릭 비교
        col1, col2 = st.columns(2)

        with col1:
            cat = cats[0]
            stats = comparison_data[cat]["stats"]
            st.markdown(f"### {cat}")
            st.metric("분석 리뷰", f"{stats['total_reviews']}개")
            positive = stats['overall_sentiment']['positive']
            total = stats['total_reviews']
            st.metric("긍정 비율", f"{positive/total*100:.0f}%" if total > 0 else "0%")
            st.metric("평균 신뢰도", f"{stats['avg_confidence']:.0%}")

        with col2:
            cat = cats[1]
            stats = comparison_data[cat]["stats"]
            st.markdown(f"### {cat}")
            st.metric("분석 리뷰", f"{stats['total_reviews']}개")
            positive = stats['overall_sentiment']['positive']
            total = stats['total_reviews']
            st.metric("긍정 비율", f"{positive/total*100:.0f}%" if total > 0 else "0%")
            st.metric("평균 신뢰도", f"{stats['avg_confidence']:.0%}")

        st.markdown("---")

        # 속성 빈도 비교 차트
        st.subheader("📊 속성별 언급 빈도 비교")

        import pandas as pd

        # 모든 속성 수집
        all_aspects = set()
        for cat in cats:
            all_aspects.update(comparison_data[cat]["stats"]["aspect_counts"].keys())

        # 데이터 구성
        chart_data = {}
        for aspect in all_aspects:
            chart_data[aspect] = {}
            for cat in cats:
                chart_data[aspect][cat] = comparison_data[cat]["stats"]["aspect_counts"].get(aspect, 0)

        if chart_data:
            df = pd.DataFrame(chart_data).T
            st.bar_chart(df)

        st.markdown("---")

        # 감정 분포 비교
        st.subheader("🎭 감정 분포 비교")

        col1, col2 = st.columns(2)

        with col1:
            cat = cats[0]
            sent = comparison_data[cat]["stats"]["overall_sentiment"]
            st.markdown(f"**{cat}**")
            st.bar_chart({"긍정": sent["positive"], "부정": sent["negative"], "중립": sent["neutral"]})

        with col2:
            cat = cats[1]
            sent = comparison_data[cat]["stats"]["overall_sentiment"]
            st.markdown(f"**{cat}**")
            st.bar_chart({"긍정": sent["positive"], "부정": sent["negative"], "중립": sent["neutral"]})

        st.markdown("---")

        # 주요 인사이트
        st.subheader("💡 주요 인사이트")

        # 각 카테고리의 상위 속성 비교
        for cat in cats:
            stats = comparison_data[cat]["stats"]
            top_aspects = list(stats["aspect_counts"].items())[:3]
            if top_aspects:
                aspects_str = ", ".join([f"{a[0]}({a[1]}회)" for a in top_aspects])
                st.markdown(f"- **{cat}** 주요 속성: {aspects_str}")

            # 긍정/부정 비율이 높은 속성
            aspect_sent = stats.get("aspect_sentiment", {})
            for aspect, sent in list(aspect_sent.items())[:2]:
                total = sent["positive"] + sent["negative"] + sent["neutral"]
                if total > 0:
                    pos_rate = sent["positive"] / total * 100
                    neg_rate = sent["negative"] / total * 100
                    if pos_rate > 70:
                        st.markdown(f"  - ✅ **{aspect}**: 긍정 비율 {pos_rate:.0f}%로 높음")
                    elif neg_rate > 50:
                        st.markdown(f"  - ⚠️ **{aspect}**: 부정 비율 {neg_rate:.0f}%로 주의 필요")


def render_reviews_tab():
    """리뷰 목록 탭 렌더링."""
    st.subheader("📋 리뷰 목록")

    processed = st.session_state.processed_reviews
    if not processed:
        st.warning("데이터를 먼저 로드해주세요.")
        return

    # 필터
    col1, col2 = st.columns(2)
    with col1:
        min_rating = st.slider("최소 평점", 1.0, 5.0, 1.0, 0.5)
    with col2:
        sort_order = st.selectbox("정렬", ["평점 높은순", "평점 낮은순"])

    # 필터링 및 정렬
    filtered = [
        r for r in processed
        if r.metadata.get('rating', 0) >= min_rating
    ]

    if sort_order == "평점 높은순":
        filtered.sort(key=lambda x: x.metadata.get('rating', 0), reverse=True)
    else:
        filtered.sort(key=lambda x: x.metadata.get('rating', 0))

    st.markdown(f"**{len(filtered)}개** 리뷰 표시 중")

    # 리뷰 표시
    for i, review in enumerate(filtered[:20]):  # 최대 20개만 표시
        rating = review.metadata.get('rating', 'N/A')
        date = review.metadata.get('date', 'N/A')

        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**⭐ {rating}**")
                st.caption(date)
            with col2:
                st.markdown(review.cleaned_text[:300] + "..." if len(review.cleaned_text) > 300 else review.cleaned_text)
            st.markdown("---")


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

    # 메인 컨텐츠 렌더링
    render_main_content()


if __name__ == "__main__":
    main()
