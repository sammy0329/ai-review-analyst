"""
AI Review Analyst - Streamlit 대시보드 (쇼핑몰 스타일).

제품 목록 → 제품 상세 → 리뷰 분석/Q&A 형태의 UI를 제공합니다.
"""

import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

from src.core.logging import get_logger, setup_logging
from src.core.exceptions import ReviewAnalystError, RateLimitError, AuthenticationError
from src.pipeline.aihub_loader import AIHubDataLoader, Product
from src.pipeline.aspect_extractor import create_aspect_extractor
from src.pipeline.preprocessor import create_default_preprocessor
from src.pipeline.embedder import create_embedder
from src.chains.rag_chain import create_rag_chain
from src.pipeline.user_review_store import UserReview, create_user_review_store

# 로깅 초기화
setup_logging(level="INFO")
logger = get_logger(__name__)


# =============================================================================
# 다운로드 헬퍼 함수
# =============================================================================

def get_product_summary_json(product: Product) -> str:
    """제품 요약 정보를 JSON으로 변환."""
    sentiment_ratio = product.get_sentiment_ratio()

    summary = {
        "product_name": product.name,
        "category": product.category,
        "main_category": product.main_category,
        "avg_rating": round(product.avg_rating, 2),
        "review_count": product.review_count,
        "sentiment_ratio": {
            "positive": round(sentiment_ratio["긍정"], 1),
            "neutral": round(sentiment_ratio["중립"], 1),
            "negative": round(sentiment_ratio["부정"], 1),
        },
        "top_aspects": product.top_aspects[:10],
        "exported_at": datetime.now().isoformat(),
    }

    return json.dumps(summary, ensure_ascii=False, indent=2)


def get_reviews_csv(product: Product) -> str:
    """리뷰 목록을 CSV로 변환."""
    reviews_data = []

    for review in product.reviews:
        polarity_map = {1: "긍정", 0: "중립", -1: "부정"}
        # AIHubReview 속성 사용
        rating = review.review_score / 20 if review.review_score >= 0 else None
        # aspects는 dict 리스트, Aspect 키 추출
        aspect_names = [asp.get("Aspect", "") for asp in review.aspects if asp.get("Aspect")]

        reviews_data.append({
            "텍스트": review.raw_text,
            "평점": rating,
            "감정": polarity_map.get(review.general_polarity, "알 수 없음"),
            "날짜": review.date or "",
            "속성": ", ".join(aspect_names),
        })

    df = pd.DataFrame(reviews_data)
    return df.to_csv(index=False, encoding="utf-8-sig")


def get_aspects_json(product: Product) -> str:
    """속성 분석 결과를 JSON으로 변환."""
    aspect_sentiments = {}

    for review in product.reviews:
        if not review.aspects:
            continue

        # AIHubReview의 aspects는 dict 리스트: [{"Aspect": "배송", "SentimentPolarity": 1}, ...]
        for asp_data in review.aspects:
            aspect_name = asp_data.get("Aspect", "")
            if not aspect_name:
                continue

            # 속성별 감정 (SentimentPolarity 사용, 없으면 리뷰 전체 감정 사용)
            polarity = int(asp_data.get("SentimentPolarity", review.general_polarity))
            polarity_map = {1: "positive", 0: "neutral", -1: "negative"}
            sentiment = polarity_map.get(polarity, "neutral")

            if aspect_name not in aspect_sentiments:
                aspect_sentiments[aspect_name] = {"positive": 0, "neutral": 0, "negative": 0, "total": 0}

            aspect_sentiments[aspect_name][sentiment] += 1
            aspect_sentiments[aspect_name]["total"] += 1

    # 정렬 (total 기준)
    sorted_aspects = sorted(
        aspect_sentiments.items(),
        key=lambda x: x[1]["total"],
        reverse=True,
    )

    result = {
        "product_name": product.name,
        "aspects": [
            {
                "name": aspect,
                "positive": data["positive"],
                "neutral": data["neutral"],
                "negative": data["negative"],
                "total": data["total"],
            }
            for aspect, data in sorted_aspects
        ],
        "exported_at": datetime.now().isoformat(),
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


def get_user_friendly_error(error: Exception) -> tuple[str, str]:
    """에러를 사용자 친화적 메시지로 변환.

    Returns:
        (에러 메시지, 해결 방법) 튜플
    """
    # 커스텀 예외 클래스 처리
    if isinstance(error, ReviewAnalystError):
        msg = f"⚠️ {error.message}"
        solution = error.suggestion or f"상세: {error.details or str(error)[:100]}"
        return (msg, solution)

    error_str = str(error).lower()

    # API 키 관련
    if "api key" in error_str or "authentication" in error_str or "401" in error_str:
        return (
            "🔑 API 인증에 실패했습니다.",
            "`.env` 파일의 `OPENAI_API_KEY`가 올바른지 확인해주세요.",
        )

    # Rate limit
    if "rate limit" in error_str or "429" in error_str:
        return (
            "⏳ API 요청 한도를 초과했습니다.",
            "잠시 후 다시 시도해주세요. (약 1분 대기)",
        )

    # 네트워크 오류
    if "connection" in error_str or "timeout" in error_str or "network" in error_str:
        return (
            "🌐 네트워크 연결에 문제가 있습니다.",
            "인터넷 연결을 확인하고 다시 시도해주세요.",
        )

    # 파일 관련
    if "file not found" in error_str or "no such file" in error_str:
        return (
            "📁 파일을 찾을 수 없습니다.",
            "데이터 파일 경로를 확인해주세요.",
        )

    # 메모리 관련
    if "memory" in error_str or "oom" in error_str:
        return (
            "💾 메모리가 부족합니다.",
            "다른 프로그램을 종료하거나 데이터 크기를 줄여주세요.",
        )

    # 기본 메시지
    return (
        "⚠️ 오류가 발생했습니다.",
        f"상세: {str(error)[:100]}",
    )


def show_error(error: Exception, context: str = ""):
    """사용자 친화적 에러 표시 + 로깅."""
    # 로그에 에러 기록
    logger.error(f"{context}: {type(error).__name__}: {error}", exc_info=True)

    msg, solution = get_user_friendly_error(error)
    if context:
        msg = f"{context}: {msg}"
    st.error(msg)
    st.caption(solution)


# =============================================================================
# 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="AI Review Analyst",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# 반응형 CSS
# =============================================================================

st.markdown("""
<style>
/* 모바일 반응형 (768px 이하) */
@media (max-width: 768px) {
    /* 메인 컨테이너 패딩 조정 */
    .main .block-container {
        padding: 1rem 0.5rem;
    }

    /* 제목 크기 조정 */
    h1 {
        font-size: 1.5rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
    }
    h3 {
        font-size: 1.1rem !important;
    }

    /* 메트릭 카드 크기 조정 */
    [data-testid="stMetric"] {
        padding: 0.5rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }

    /* 버튼 크기 조정 */
    .stButton > button {
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
    }

    /* 탭 (라디오 버튼) 스크롤 가능하게 */
    [data-testid="stHorizontalBlock"] {
        overflow-x: auto;
        flex-wrap: nowrap !important;
    }

    /* 제품 카드 1열로 */
    [data-testid="column"] {
        min-width: 100% !important;
    }

    /* expander 헤더 크기 */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
    }

    /* 채팅 입력창 */
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important; /* iOS 줌 방지 */
    }
}

/* 태블릿 반응형 (769px ~ 1024px) */
@media (min-width: 769px) and (max-width: 1024px) {
    .main .block-container {
        padding: 1rem 1rem;
    }

    h1 {
        font-size: 1.75rem !important;
    }
}

/* 제품 카드 스타일 개선 */
.product-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    transition: box-shadow 0.2s;
}
.product-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 세션 상태 초기화
# =============================================================================

def init_session_state():
    """세션 상태 초기화."""
    # 뷰 모드: "b2b" (기업) 또는 "b2c" (소비자)
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "b2c"  # 기본값: 소비자 모드

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

    # 사용자 리뷰 저장소
    if "user_review_store" not in st.session_state:
        st.session_state.user_review_store = create_user_review_store()

    # 새로 추가된 리뷰 ID (자동 확장용)
    if "newly_added_review_id" not in st.session_state:
        st.session_state.newly_added_review_id = None

    # 비교할 제품 목록 (최대 4개)
    if "compare_products" not in st.session_state:
        st.session_state.compare_products = []


init_session_state()


# =============================================================================
# 모드 토글 UI
# =============================================================================

def render_mode_toggle():
    """뷰 모드 토글 렌더링."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # 현재 모드에 따른 라벨
        mode_labels = {
            "b2c": "🛒 소비자 모드",
            "b2b": "📊 기업 모드"
        }

        # 토글 버튼 스타일
        toggle_cols = st.columns([1, 1])

        with toggle_cols[0]:
            if st.button(
                "🛒 소비자",
                use_container_width=True,
                type="primary" if st.session_state.view_mode == "b2c" else "secondary",
                help="구매 결정에 도움이 되는 간단한 정보"
            ):
                st.session_state.view_mode = "b2c"
                st.rerun()

        with toggle_cols[1]:
            if st.button(
                "📊 기업",
                use_container_width=True,
                type="primary" if st.session_state.view_mode == "b2b" else "secondary",
                help="상세 분석 및 인사이트 대시보드"
            ):
                st.session_state.view_mode = "b2b"
                st.rerun()

    st.markdown("---")


def get_mode_description() -> str:
    """현재 모드 설명 반환."""
    if st.session_state.view_mode == "b2c":
        return "💡 **소비자 모드**: 구매 결정에 필요한 핵심 정보만 보여드려요"
    else:
        return "💼 **기업 모드**: 상세 분석과 데이터 기반 인사이트를 제공해요"


def get_or_create_product_rag_chain(product: Product):
    """제품별 RAG Chain 생성 또는 캐시된 것 반환."""
    product_name = product.name

    # 이미 생성된 RAG Chain이 있으면 반환
    if (st.session_state.product_rag_chain is not None and
        st.session_state.get("current_rag_product") == product_name):
        return st.session_state.product_rag_chain

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

        return rag_chain

    except Exception as e:
        logger.error(f"RAG Chain 생성 오류: {e}")
        return None


# =============================================================================
# API 체크
# =============================================================================

def check_api_key():
    """API 키 확인."""
    api_key = os.getenv("OPENAI_API_KEY")
    return bool(api_key)


# =============================================================================
# 제품 로드
# =============================================================================

def get_data_dir() -> str:
    """데이터 디렉토리 경로 반환 (병합 폴더 우선)."""
    from pathlib import Path
    merged_dir = Path("data/aihub_merged")
    if merged_dir.exists():
        return str(merged_dir)
    return "data/aihub_data"


def load_products(category: str):
    """제품 목록 로드."""
    with st.spinner("📦 제품 로드 중..."):
        try:
            loader = AIHubDataLoader(data_dir=get_data_dir())

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
            show_error(e, "제품 로드")


# =============================================================================
# 제품 목록 페이지
# =============================================================================

def render_product_list():
    """제품 목록 페이지 렌더링."""
    st.title("🛒 AI Review Analyst")

    # 모드 토글
    render_mode_toggle()

    products = st.session_state.products

    # 상단 필터 (대분류 + 소분류 + 검색 + 정렬 + 비교)
    col_cat, col_subcat, col_search, col_sort, col_compare = st.columns([1, 1.2, 2, 1, 1])

    with col_cat:
        categories = ["전체", "패션", "화장품", "가전", "IT기기", "생활용품"]
        selected_category = st.selectbox(
            "대분류",
            categories,
            key="category_filter",
        )

    with col_subcat:
        # 소분류 목록 (제품에서 추출)
        if products and selected_category != "전체":
            subcategories = sorted(set(
                p.main_category for p in products
                if p.category == selected_category and p.main_category
            ))
            subcategories = ["전체"] + subcategories
        else:
            subcategories = ["전체"]

        selected_subcategory = st.selectbox(
            "소분류",
            subcategories,
            key="subcategory_filter",
        )

    with col_search:
        search_query = st.text_input(
            "검색",
            placeholder="제품명 검색...",
        )

    with col_sort:
        sort_option = st.selectbox(
            "정렬",
            ["리뷰 많은순", "평점 높은순", "평점 낮은순"],
        )

    with col_compare:
        # 빈 레이블로 높이 맞춤
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        compare_count = len(st.session_state.compare_products)
        if compare_count >= 2:
            if st.button(f"📊 비교 ({compare_count})", use_container_width=True, type="primary"):
                st.session_state.current_page = "compare"
                st.rerun()
        else:
            if st.button(f"📊 비교 ({compare_count}/4)", use_container_width=True, disabled=True):
                pass

    # 카테고리 상태 초기화
    if "last_category" not in st.session_state:
        st.session_state.last_category = selected_category

    # 제품이 없거나 대분류 변경 시 자동 로드
    if not products:
        load_products(selected_category)
        return

    if st.session_state.last_category != selected_category:
        st.session_state.last_category = selected_category
        load_products(selected_category)
        return

    # 필터링 및 정렬
    filtered_products = products

    # 소분류 필터
    if selected_subcategory != "전체":
        filtered_products = [
            p for p in filtered_products
            if p.main_category == selected_subcategory
        ]

    # 검색 필터
    if search_query:
        filtered_products = [
            p for p in filtered_products
            if search_query.lower() in p.name.lower()
        ]

    # 정렬
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
    """제품 카드 렌더링 (모드에 따라 분기)."""
    if st.session_state.view_mode == "b2c":
        render_product_card_b2c(product)
    else:
        render_product_card_b2b(product)


def render_product_card_b2c(product: Product):
    """소비자 모드 - 제품 카드 (간단한 추천 여부 중심)."""
    sentiment_ratio = product.get_sentiment_ratio()
    positive_ratio = sentiment_ratio["긍정"]
    avg_rating = product.avg_rating

    # 추천 여부 결정
    if avg_rating >= 4.0 and positive_ratio >= 60:
        verdict = "👍 추천"
        verdict_style = "success"
    elif avg_rating >= 3.5 or positive_ratio >= 50:
        verdict = "🤔 보통"
        verdict_style = "info"
    else:
        verdict = "⚠️ 주의"
        verdict_style = "warning"

    # 제품 제목
    display_name = product.name[:25] + "..." if len(product.name) > 25 else product.name
    st.markdown(f"**{display_name}**")

    # 평점 & 추천
    st.markdown(f"⭐ **{avg_rating:.1f}** · {verdict}")

    # 리뷰 수
    st.caption(f"📝 리뷰 {product.review_count}개")

    # 상세 보기 버튼만 (비교는 B2B에서만)
    if st.button("상세 보기", key=f"view_b2c_{product.name}", use_container_width=True):
        st.session_state.selected_product = product
        st.session_state.current_page = "product_detail"
        st.rerun()

    st.markdown("---")


def render_product_card_b2b(product: Product):
    """기업 모드 - 제품 카드 (상세 정보)."""
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

    # 제품 제목 (2줄 고정)
    display_name = product.name[:28] + "..." if len(product.name) > 28 else product.name
    st.markdown(f"**📦 {display_name}**")

    # 카테고리
    st.caption(f"{product.category} > {product.main_category}")

    # 평점 & 리뷰 수 (한 줄)
    st.markdown(f"⭐ **{product.avg_rating:.1f}** · 📝 **{product.review_count}개**")

    # 감정 상태
    st.markdown(f"{sentiment_color} {sentiment_text} ({positive_ratio:.0f}% 긍정)")

    # 주요 속성 태그 (3개 고정, 없으면 빈 태그)
    aspects = product.top_aspects[:3] if product.top_aspects else ["-", "-", "-"]
    while len(aspects) < 3:
        aspects.append("-")
    tags = " ".join([f"`{a}`" for a in aspects])
    st.markdown(f"🏷️ {tags}")

    # 비교 체크박스 + 상세 보기 버튼
    col_compare, col_detail = st.columns([1, 2])

    with col_compare:
        is_in_compare = any(p.name == product.name for p in st.session_state.compare_products)
        compare_disabled = len(st.session_state.compare_products) >= 4 and not is_in_compare

        if st.checkbox(
            "비교",
            value=is_in_compare,
            key=f"compare_b2b_{product.name}",
            disabled=compare_disabled,
        ):
            if not is_in_compare:
                st.session_state.compare_products.append(product)
                st.rerun()
        else:
            if is_in_compare:
                st.session_state.compare_products = [
                    p for p in st.session_state.compare_products if p.name != product.name
                ]
                st.rerun()

    with col_detail:
        if st.button("상세 보기", key=f"view_b2b_{product.name}", use_container_width=True):
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

    # 상단 네비게이션
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← 목록으로", use_container_width=True):
            st.session_state.current_page = "product_list"
            st.session_state.selected_product = None
            st.rerun()

    # 헤더
    st.title(f"📦 {product.name}")
    st.caption(f"{product.category} > {product.main_category}")

    # 모드 토글
    render_mode_toggle()

    # 모드별 렌더링 분기
    if st.session_state.view_mode == "b2c":
        render_product_detail_b2c(product)
    else:
        render_product_detail_b2b(product)


def render_product_detail_b2c(product: Product):
    """소비자 모드 - 제품 상세 페이지 (간단한 구매 결정 도움)."""
    sentiment_ratio = product.get_sentiment_ratio()

    # 한눈에 보는 평가 카드
    st.subheader("📋 한눈에 보기")

    # 전체 평가 요약
    avg_rating = product.avg_rating
    pos_ratio = sentiment_ratio["긍정"]

    if avg_rating >= 4.0 and pos_ratio >= 60:
        verdict = "👍 추천해요!"
        verdict_color = "success"
        verdict_detail = "평점도 높고 긍정 리뷰가 많아요"
    elif avg_rating >= 3.5 or pos_ratio >= 50:
        verdict = "🤔 괜찮아요"
        verdict_color = "info"
        verdict_detail = "전반적으로 무난한 제품이에요"
    else:
        verdict = "⚠️ 신중히 고려하세요"
        verdict_color = "warning"
        verdict_detail = "부정적인 리뷰가 있어요"

    # 평가 카드
    eval_col1, eval_col2 = st.columns([1, 2])

    with eval_col1:
        st.metric("평균 평점", f"⭐ {avg_rating:.1f} / 5.0")
        st.metric("리뷰 수", f"📝 {product.review_count}개")

    with eval_col2:
        if verdict_color == "success":
            st.success(f"**{verdict}**\n\n{verdict_detail}")
        elif verdict_color == "warning":
            st.warning(f"**{verdict}**\n\n{verdict_detail}")
        else:
            st.info(f"**{verdict}**\n\n{verdict_detail}")

    st.markdown("---")

    # 장점/단점 TOP 3
    st.subheader("👍 장점 vs 👎 단점")

    # 속성별 감정 분석
    aspect_sentiment = {}
    for review in product.reviews:
        for aspect in review.aspects:
            aspect_name = aspect.get("Aspect", "")
            polarity = int(aspect.get("SentimentPolarity", 0))

            if aspect_name:
                if aspect_name not in aspect_sentiment:
                    aspect_sentiment[aspect_name] = {"positive": 0, "negative": 0}

                if polarity == 1:
                    aspect_sentiment[aspect_name]["positive"] += 1
                elif polarity == -1:
                    aspect_sentiment[aspect_name]["negative"] += 1

    # 긍정/부정 TOP 3 추출
    positive_aspects = sorted(
        [(k, v["positive"]) for k, v in aspect_sentiment.items() if v["positive"] > 0],
        key=lambda x: x[1],
        reverse=True
    )[:3]

    negative_aspects = sorted(
        [(k, v["negative"]) for k, v in aspect_sentiment.items() if v["negative"] > 0],
        key=lambda x: x[1],
        reverse=True
    )[:3]

    good_col, bad_col = st.columns(2)

    with good_col:
        st.markdown("#### 👍 이런 점이 좋아요")
        if positive_aspects:
            for aspect, count in positive_aspects:
                st.markdown(f"- **{aspect}** ({count}명 언급)")
        else:
            st.caption("긍정적인 속성 정보가 없어요")

    with bad_col:
        st.markdown("#### 👎 이런 점이 아쉬워요")
        if negative_aspects:
            for aspect, count in negative_aspects:
                st.markdown(f"- **{aspect}** ({count}명 언급)")
        else:
            st.caption("부정적인 속성 정보가 없어요")

    st.markdown("---")

    # 대표 리뷰
    st.subheader("💬 대표 리뷰")

    # 긍정/부정 대표 리뷰 각 2개
    positive_reviews = [r for r in product.reviews if r.general_polarity == 1][:2]
    negative_reviews = [r for r in product.reviews if r.general_polarity == -1][:2]

    review_col1, review_col2 = st.columns(2)

    with review_col1:
        st.markdown("**😊 긍정 리뷰**")
        if positive_reviews:
            for r in positive_reviews:
                text = r.raw_text[:150] + "..." if len(r.raw_text) > 150 else r.raw_text
                st.info(f'"{text}"')
        else:
            st.caption("긍정 리뷰가 없어요")

    with review_col2:
        st.markdown("**😞 부정 리뷰**")
        if negative_reviews:
            for r in negative_reviews:
                text = r.raw_text[:150] + "..." if len(r.raw_text) > 150 else r.raw_text
                st.warning(f'"{text}"')
        else:
            st.caption("부정 리뷰가 없어요")

    st.markdown("---")

    # 간단 Q&A
    st.subheader("💡 궁금한 점이 있으신가요?")
    st.caption("리뷰를 기반으로 AI가 답변해드려요")

    # 자주 묻는 질문 버튼
    faq_col1, faq_col2, faq_col3 = st.columns(3)

    with faq_col1:
        if st.button("📦 배송은 어때요?", use_container_width=True, key="faq_delivery"):
            st.session_state.b2c_question = "배송은 어떤가요? 빠른 편인가요?"

    with faq_col2:
        if st.button("💰 가성비 좋아요?", use_container_width=True, key="faq_value"):
            st.session_state.b2c_question = "가성비가 좋은 제품인가요?"

    with faq_col3:
        if st.button("⚠️ 단점은 뭐예요?", use_container_width=True, key="faq_cons"):
            st.session_state.b2c_question = "이 제품의 주요 단점이 뭔가요?"

    # 직접 질문 입력
    user_question = st.text_input(
        "직접 질문하기",
        placeholder="예: 사이즈가 작은 편인가요?",
        key="b2c_user_question"
    )

    # FAQ 버튼 또는 직접 입력 질문 처리
    question_to_ask = getattr(st.session_state, "b2c_question", None) or user_question

    if question_to_ask:
        if "b2c_question" in st.session_state:
            del st.session_state.b2c_question

        with st.spinner("🤖 AI가 리뷰를 분석하고 있어요..."):
            try:
                rag_chain = get_or_create_product_rag_chain(product)
                if rag_chain:
                    response = rag_chain.query(question_to_ask)
                    st.markdown("#### �� AI 답변")
                    st.success(response.answer)
            except Exception as e:
                st.error(f"답변 생성 중 오류: {e}")


def render_product_detail_b2b(product: Product):
    """기업 모드 - 제품 상세 페이지 (상세 분석 대시보드)."""
    sentiment_ratio = product.get_sentiment_ratio()

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

    # 다운로드 버튼
    with st.expander("📥 데이터 다운로드"):
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        # 안전한 파일명 생성
        safe_filename = "".join(c if c.isalnum() or c in "-_" else "_" for c in product.name[:30])

        with dl_col1:
            summary_json = get_product_summary_json(product)
            st.download_button(
                label="📊 요약 (JSON)",
                data=summary_json,
                file_name=f"{safe_filename}_summary.json",
                mime="application/json",
                use_container_width=True,
            )

        with dl_col2:
            reviews_csv = get_reviews_csv(product)
            st.download_button(
                label="📋 리뷰 (CSV)",
                data=reviews_csv,
                file_name=f"{safe_filename}_reviews.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with dl_col3:
            aspects_json = get_aspects_json(product)
            st.download_button(
                label="🏷️ 속성 분석 (JSON)",
                data=aspects_json,
                file_name=f"{safe_filename}_aspects.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown("---")

    # 사용자 리뷰 수 표시
    user_review_count = st.session_state.user_review_store.get_review_count(product.name)
    if user_review_count > 0:
        st.info(f"✍️ 사용자 추가 리뷰: {user_review_count}개")

    # 탭 (radio 버튼으로 상태 유지)
    tab_options = ["📊 요약", "🏷️ 속성 분석", "💬 Q&A", "📋 리뷰 목록", "✍️ 리뷰 추가"]
    # 제품명에서 안전한 키 생성 (특수문자 제거)
    safe_product_key = "".join(c if c.isalnum() else "_" for c in product.name[:30])
    tab_key = f"product_tab_{safe_product_key}"

    selected_tab = st.radio(
        "탭 선택",
        options=tab_options,
        horizontal=True,
        key=tab_key,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if selected_tab == "📊 요약":
        render_product_summary(product)
    elif selected_tab == "🏷️ 속성 분석":
        render_product_aspects(product)
    elif selected_tab == "💬 Q&A":
        render_product_qa(product)
    elif selected_tab == "📋 리뷰 목록":
        render_product_reviews(product)
    elif selected_tab == "✍️ 리뷰 추가":
        render_add_review(product)


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
                show_error(e, "Q&A 시스템 초기화")
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

        # AI 응답 생성 (스트리밍)
        with st.chat_message("assistant"):
            try:
                rag_chain = st.session_state.product_rag_chain

                # 스트리밍 + 출처 가져오기
                stream, sources = rag_chain.stream_with_sources(prompt)

                # 스트리밍 응답 표시
                answer = st.write_stream(stream)

                # 출처 표시
                if sources:
                    with st.expander("📚 참조 리뷰"):
                        for i, source in enumerate(sources, 1):
                            rating = source.get("rating", "N/A")
                            st.markdown(f"**[{i}]** ⭐ {rating}")
                            st.markdown(f"> {source['text'][:300]}...")
                            st.markdown("---")

                # 메시지 저장
                messages.append({
                    "role": "assistant",
                    "content": answer,
                })

            except Exception as e:
                show_error(e, "답변 생성")


def render_product_reviews(product: Product):
    """리뷰 목록 탭."""
    st.subheader("📋 리뷰 목록")

    # 사용자 리뷰 가져오기
    user_reviews = st.session_state.user_review_store.get_reviews(product.name)
    user_review_count = len(user_reviews)

    reviews = product.reviews

    # 필터
    col1, col2, col3 = st.columns(3)
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
    with col3:
        source_filter = st.selectbox(
            "소스",
            ["전체", "AI Hub", "사용자 추가"],
            key=f"source_filter_{product.name}",
        )

    # 필터링
    polarity_map = {1: "긍정", 0: "중립", -1: "부정"}
    sentiment_en_kr = {"positive": "긍정", "negative": "부정", "neutral": "중립"}

    # AI Hub 리뷰 필터링
    filtered_aihub = reviews if source_filter in ["전체", "AI Hub"] else []
    if sentiment_filter != "전체" and filtered_aihub:
        filtered_aihub = [
            r for r in filtered_aihub
            if polarity_map.get(r.general_polarity, "중립") == sentiment_filter
        ]

    # 사용자 리뷰 필터링
    filtered_user = user_reviews if source_filter in ["전체", "사용자 추가"] else []
    if sentiment_filter != "전체" and filtered_user:
        filtered_user = [
            r for r in filtered_user
            if sentiment_en_kr.get(r.overall_sentiment, "중립") == sentiment_filter
        ]

    # 정렬 (AI Hub만)
    if sort_option == "긍정순":
        filtered_aihub.sort(key=lambda r: r.general_polarity, reverse=True)
    elif sort_option == "부정순":
        filtered_aihub.sort(key=lambda r: r.general_polarity)

    total_count = len(filtered_aihub) + len(filtered_user)
    st.markdown(f"**{total_count}개** 리뷰 (AI Hub: {len(filtered_aihub)}, 사용자: {len(filtered_user)})")
    st.markdown("---")

    # 사용자 리뷰 먼저 표시 (최신순)
    if filtered_user:
        st.markdown("#### ✍️ 사용자 추가 리뷰")
        for i, review in enumerate(reversed(filtered_user)):
            sentiment_kr = sentiment_en_kr.get(review.overall_sentiment, "중립")
            emoji = {"긍정": "😊", "중립": "😐", "부정": "😞"}.get(sentiment_kr, "❓")

            # 별점 표시
            rating = getattr(review, 'rating', 0)
            stars_display = "⭐" * rating if rating > 0 else ""

            with st.expander(f"{stars_display} {emoji} [사용자] {review.text[:40]}...", expanded=False):
                if rating > 0:
                    full_stars = "⭐" * rating + "☆" * (5 - rating)
                    st.markdown(f"**별점:** {full_stars} ({rating}점)")
                st.markdown(review.text)
                st.caption(f"📅 {review.created_at[:10]} | 🤖 AI 분석 완료 (신뢰도: {review.confidence:.0%})")

                if review.aspects:
                    st.markdown("---")
                    st.markdown("**AI 추출 속성:**")
                    for aspect in review.aspects:
                        a_sentiment = aspect.get("sentiment", "neutral")
                        a_emoji = {"positive": "👍", "negative": "👎", "neutral": "➖"}.get(a_sentiment, "❓")
                        st.markdown(f"- {a_emoji} **{aspect.get('category', '')}**: {aspect.get('text', '')[:80]}...")

        if filtered_aihub:
            st.markdown("---")

    # AI Hub 리뷰 표시
    if filtered_aihub:
        st.markdown("#### 📦 AI Hub 리뷰")
        for i, review in enumerate(filtered_aihub[:20]):  # 최대 20개
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


def render_add_review(product: Product):
    """리뷰 추가 탭 - LLM 기반 속성 추출."""
    st.subheader("✍️ 리뷰 추가")
    st.markdown("직접 리뷰를 작성하면 **AI가 속성을 자동 분석**합니다.")

    # 별점 선택 UI
    st.markdown("**별점을 선택하세요**")
    rating_options = {
        "⭐": 1,
        "⭐⭐": 2,
        "⭐⭐⭐": 3,
        "⭐⭐⭐⭐": 4,
        "⭐⭐⭐⭐⭐": 5,
    }
    rating_key = f"star_rating_{product.name}"

    selected_stars = st.radio(
        "별점",
        options=list(rating_options.keys()),
        index=4,  # 기본 5점
        horizontal=True,
        key=rating_key,
        label_visibility="collapsed",
    )
    current_rating = rating_options[selected_stars]

    rating_text = {1: "매우 불만족", 2: "불만족", 3: "보통", 4: "만족", 5: "매우 만족"}
    st.caption(f"{current_rating}점 - {rating_text[current_rating]}")

    # 텍스트 영역 키 (제품명 안전 처리)
    safe_name = "".join(c if c.isalnum() else "_" for c in product.name[:30])
    text_key = f"review_text_{safe_name}"
    clear_flag_key = f"clear_review_text_{safe_name}"

    # 텍스트 초기화 플래그 처리
    if st.session_state.get(clear_flag_key, False):
        st.session_state[text_key] = ""
        st.session_state[clear_flag_key] = False

    # 리뷰 입력 (form 없이)
    review_text = st.text_area(
        "리뷰 내용",
        placeholder="이 제품에 대한 리뷰를 작성해주세요...\n예: 가격은 좀 비싸지만 품질이 정말 좋아요. 배송도 빨랐습니다.",
        height=150,
        key=text_key,
    )

    if st.button("🔍 AI 분석 후 저장", key=f"submit_review_{product.name}", use_container_width=True):
        if review_text.strip():
            with st.spinner("🤖 AI가 리뷰를 분석하고 있습니다..."):
                try:
                    # AspectExtractor로 분석
                    extractor = create_aspect_extractor(use_cache=True)
                    result = extractor.extract(review_text.strip())

                    # UserReview 생성 (별점 포함)
                    user_review = UserReview.create(
                        product_name=product.name,
                        text=review_text.strip(),
                        rating=current_rating,
                    )

                    # 분석 결과 업데이트
                    user_review.overall_sentiment = result.overall_sentiment.value
                    user_review.confidence = result.confidence
                    user_review.aspects = result.aspects
                    user_review.analyzed = True

                    # 저장
                    st.session_state.user_review_store.add_review(user_review)

                    # 새로 추가된 리뷰 ID 저장
                    st.session_state.newly_added_review_id = user_review.id

                    # 텍스트 초기화 플래그 설정
                    st.session_state[clear_flag_key] = True

                    # rerun으로 텍스트 초기화 (탭은 radio로 유지됨)
                    st.rerun()

                except Exception as e:
                    show_error(e, "리뷰 분석")
        else:
            st.warning("리뷰 내용을 입력해주세요.")

    # 기존 사용자 리뷰 표시
    st.markdown("---")
    st.markdown("### 📝 내가 추가한 리뷰")

    user_reviews = st.session_state.user_review_store.get_reviews(product.name)

    if not user_reviews:
        st.info("아직 추가한 리뷰가 없습니다.")
    else:
        # 새로 추가된 리뷰 ID 확인
        newly_added_id = st.session_state.newly_added_review_id

        for review in reversed(user_reviews):  # 최신순
            sentiment_emoji = {
                "positive": "😊",
                "negative": "😞",
                "neutral": "😐",
            }

            # 새로 추가된 리뷰는 자동 확장
            is_newly_added = review.id == newly_added_id

            # 별점 표시
            rating = getattr(review, 'rating', 0)
            stars_display = "⭐" * rating if rating > 0 else ""

            with st.expander(
                f"{stars_display} {sentiment_emoji.get(review.overall_sentiment, '❓')} {review.text[:40]}...",
                expanded=is_newly_added
            ):
                # 새로 추가된 리뷰 - ID 초기화 (다음 렌더링을 위해)
                if is_newly_added:
                    st.session_state.newly_added_review_id = None

                # 별점 표시
                if rating > 0:
                    full_stars = "⭐" * rating + "☆" * (5 - rating)
                    st.markdown(f"**별점:** {full_stars} ({rating}점)")

                st.markdown(f"**리뷰:** {review.text}")
                st.markdown(f"**작성일:** {review.created_at[:10]}")

                sentiment_kr = {"positive": "긍정", "negative": "부정", "neutral": "중립"}
                st.markdown(f"**AI 감정 분석:** {sentiment_kr.get(review.overall_sentiment, '중립')} (신뢰도: {review.confidence:.0%})")

                if review.aspects:
                    st.markdown("**🤖 AI 속성 분석:**")
                    sentiment_color = {
                        "positive": "#e3f2fd",
                        "negative": "#ffebee",
                        "neutral": "#e8f5e9",
                    }
                    for aspect in review.aspects:
                        a_sentiment = aspect.get("sentiment", "neutral")
                        a_emoji = {"positive": "👍", "negative": "👎", "neutral": "➖"}.get(a_sentiment, "❓")
                        a_sentiment_kr = {"positive": "긍정", "negative": "부정", "neutral": "중립"}.get(a_sentiment, "중립")

                        st.markdown(
                            f'<div style="background-color: {sentiment_color.get(a_sentiment, "#f5f5f5")}; '
                            f'padding: 8px 12px; border-radius: 5px; margin-bottom: 6px;">'
                            f'{a_emoji} <b>{aspect.get("category", "")}</b>: {a_sentiment_kr}<br>'
                            f'<span style="color: #666; font-size: 0.9em;">"{aspect.get("text", "")}"</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # 삭제 버튼
                if st.button("🗑️ 삭제", key=f"delete_{review.id}"):
                    st.session_state.user_review_store.delete_review(product.name, review.id)
                    st.rerun()


# =============================================================================
# 제품 비교 페이지
# =============================================================================

def render_compare_products():
    """제품 비교 페이지 렌더링."""
    from collections import Counter

    products = st.session_state.compare_products

    # 상단 네비게이션
    col_back, col_clear, col_spacer = st.columns([1, 1, 4])
    with col_back:
        if st.button("← 목록으로", use_container_width=True):
            st.session_state.current_page = "product_list"
            st.rerun()
    with col_clear:
        if st.button("🗑️ 비교 초기화", use_container_width=True):
            st.session_state.compare_products = []
            st.session_state.current_page = "product_list"
            st.rerun()

    if len(products) < 2:
        st.warning("비교하려면 최소 2개 제품을 선택하세요.")
        return

    st.title("📊 제품 비교")
    st.markdown(f"**{len(products)}개 제품** 비교 분석")
    st.markdown("---")

    # 1. 제품 요약 비교 테이블
    st.subheader("📋 제품 요약 비교")

    # 테이블 헤더
    cols = st.columns(len(products) + 1)
    cols[0].markdown("**항목**")
    for i, product in enumerate(products):
        cols[i + 1].markdown(f"**{product.name[:15]}...**")

    # 평점
    cols = st.columns(len(products) + 1)
    cols[0].markdown("⭐ 평점")
    for i, product in enumerate(products):
        cols[i + 1].markdown(f"**{product.avg_rating:.1f}**")

    # 리뷰 수
    cols = st.columns(len(products) + 1)
    cols[0].markdown("📝 리뷰 수")
    for i, product in enumerate(products):
        cols[i + 1].markdown(f"**{product.review_count}개**")

    # 긍정 비율
    cols = st.columns(len(products) + 1)
    cols[0].markdown("😊 긍정 비율")
    for i, product in enumerate(products):
        ratio = product.get_sentiment_ratio()
        cols[i + 1].markdown(f"**{ratio['긍정']:.0f}%**")

    # 부정 비율
    cols = st.columns(len(products) + 1)
    cols[0].markdown("😞 부정 비율")
    for i, product in enumerate(products):
        ratio = product.get_sentiment_ratio()
        cols[i + 1].markdown(f"**{ratio['부정']:.0f}%**")

    st.markdown("---")

    # 2. 감정 분포 비교 차트
    st.subheader("📈 감정 분포 비교")

    chart_data = {}
    for product in products:
        ratio = product.get_sentiment_ratio()
        short_name = product.name[:12] + "..." if len(product.name) > 12 else product.name
        chart_data[short_name] = {
            "긍정": ratio["긍정"],
            "중립": ratio["중립"],
            "부정": ratio["부정"],
        }

    # DataFrame으로 변환
    import pandas as pd
    df = pd.DataFrame(chart_data).T
    st.bar_chart(df)

    st.markdown("---")

    # 3. 속성별 감정 비교
    st.subheader("🏷️ 속성별 감정 비교")

    # 모든 제품에서 언급된 속성 수집
    all_aspects = set()
    product_aspect_data = {}

    for product in products:
        aspect_counter: Counter = Counter()
        aspect_sentiment: dict = {}

        for review in product.reviews:
            for aspect in review.aspects:
                aspect_name = aspect.get("Aspect", "")
                if aspect_name:
                    all_aspects.add(aspect_name)
                    aspect_counter[aspect_name] += 1

                    # 감정별 집계 (AI Hub 데이터는 문자열로 저장)
                    polarity = int(aspect.get("SentimentPolarity", 0))
                    if aspect_name not in aspect_sentiment:
                        aspect_sentiment[aspect_name] = {"긍정": 0, "중립": 0, "부정": 0}

                    if polarity == 1:
                        aspect_sentiment[aspect_name]["긍정"] += 1
                    elif polarity == -1:
                        aspect_sentiment[aspect_name]["부정"] += 1
                    else:
                        aspect_sentiment[aspect_name]["중립"] += 1

        product_aspect_data[product.name] = {
            "counter": aspect_counter,
            "sentiment": aspect_sentiment,
        }

    # 상위 속성만 표시 (전체에서 가장 많이 언급된 순)
    total_counter: Counter = Counter()
    for product in products:
        for review in product.reviews:
            for aspect in review.aspects:
                aspect_name = aspect.get("Aspect", "")
                if aspect_name:
                    total_counter[aspect_name] += 1

    top_aspects = [a for a, _ in total_counter.most_common(8)]

    if top_aspects:
        # 속성별 비교 테이블
        for aspect_name in top_aspects:
            st.markdown(f"#### 🏷️ {aspect_name}")

            cols = st.columns(len(products))
            for i, product in enumerate(products):
                with cols[i]:
                    data = product_aspect_data.get(product.name, {})
                    sentiment = data.get("sentiment", {}).get(aspect_name, {"긍정": 0, "중립": 0, "부정": 0})
                    total = sum(sentiment.values())

                    if total > 0:
                        pos_pct = sentiment["긍정"] / total * 100
                        neg_pct = sentiment["부정"] / total * 100

                        st.markdown(f"**{product.name[:10]}...**")
                        st.markdown(f"언급 {total}회")

                        # 감정 막대
                        if pos_pct >= 60:
                            st.success(f"😊 긍정 {pos_pct:.0f}%")
                        elif neg_pct >= 40:
                            st.error(f"😞 부정 {neg_pct:.0f}%")
                        else:
                            st.info(f"😐 혼재 (긍정 {pos_pct:.0f}%)")
                    else:
                        st.markdown(f"**{product.name[:10]}...**")
                        st.caption("언급 없음")

            st.markdown("---")
    else:
        st.info("속성 정보가 없습니다.")

    # 4. 추천 인사이트
    st.subheader("💡 비교 인사이트")

    # 최고 평점 제품
    best_rating = max(products, key=lambda p: p.avg_rating)
    st.success(f"⭐ **최고 평점:** {best_rating.name[:30]}... ({best_rating.avg_rating:.1f}점)")

    # 가장 긍정적인 제품
    best_positive = max(products, key=lambda p: p.get_sentiment_ratio()["긍정"])
    pos_ratio = best_positive.get_sentiment_ratio()["긍정"]
    st.success(f"😊 **가장 긍정적:** {best_positive.name[:30]}... ({pos_ratio:.0f}% 긍정)")

    # 가장 리뷰 많은 제품
    most_reviews = max(products, key=lambda p: p.review_count)
    st.info(f"📝 **리뷰 가장 많음:** {most_reviews.name[:30]}... ({most_reviews.review_count}개)")

    # 주의 필요 제품 (부정 비율 높은 경우)
    for product in products:
        ratio = product.get_sentiment_ratio()
        if ratio["부정"] >= 40:
            st.warning(f"⚠️ **주의 필요:** {product.name[:30]}... (부정 {ratio['부정']:.0f}%)")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 함수."""
    # API 키 확인
    if not check_api_key():
        st.error("🔑 OpenAI API 키가 필요합니다")
        st.markdown("""
        **설정 방법:**
        1. 프로젝트 루트에 `.env` 파일 생성
        2. 다음 내용 추가: `OPENAI_API_KEY=sk-your-api-key`
        3. 앱 재시작

        API 키는 [OpenAI 대시보드](https://platform.openai.com/api-keys)에서 발급받을 수 있습니다.
        """)
        st.stop()

    # 페이지 라우팅
    if st.session_state.current_page == "product_list":
        render_product_list()
    elif st.session_state.current_page == "product_detail":
        render_product_detail()
    elif st.session_state.current_page == "compare":
        render_compare_products()


if __name__ == "__main__":
    main()
