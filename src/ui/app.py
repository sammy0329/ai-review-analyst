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
from src.database import (
    init_db, add_review, get_reviews_by_product, migrate_aihub_product,
    get_or_create_product, delete_review, get_review_aspects_by_text,
    get_product_by_name, get_all_products as db_get_products,
    get_review_count, save_qa_feedback, save_qa_log, get_qa_metrics,
)
from src.pipeline.aihub_loader import AIHubDataLoader, Product, AIHubReview
from src.pipeline.aspect_extractor import create_aspect_extractor
from src.pipeline.preprocessor import create_default_preprocessor
from src.pipeline.embedder import create_embedder
from src.chains.rag_chain import create_rag_chain
from src.pipeline.fake_review_filter import check_review_text
from src.pipeline.semantic_cache import get_semantic_cache, CacheResult

# 로깅 초기화
setup_logging(level="INFO")
logger = get_logger(__name__)


# =============================================================================
# 속성 분석 헬퍼 함수
# =============================================================================

def analyze_product_aspects(product: Product) -> dict:
    """제품 리뷰에서 속성별 감정 분석.

    Args:
        product: 분석할 제품

    Returns:
        {
            "strengths": [(속성명, 긍정수, 긍정비율), ...],  # 강점 (긍정 많은 속성)
            "weaknesses": [(속성명, 부정수, 부정비율), ...],  # 약점 (부정 많은 속성)
            "aspect_stats": {속성명: {"긍정": n, "부정": n, "중립": n}, ...}
        }
    """
    from collections import defaultdict

    # 속성별 감정 집계
    aspect_stats = defaultdict(lambda: {"긍정": 0, "부정": 0, "중립": 0})
    polarity_map = {1: "긍정", 0: "중립", -1: "부정"}

    for review in product.reviews:
        for aspect in review.aspects:
            aspect_name = aspect.get("Aspect", "")
            polarity_raw = aspect.get("SentimentPolarity", 0)
            # SentimentPolarity가 문자열일 수 있으므로 정수로 변환
            try:
                polarity = int(polarity_raw)
            except (ValueError, TypeError):
                polarity = 0

            if aspect_name:
                label = polarity_map.get(polarity, "중립")
                aspect_stats[aspect_name][label] += 1

    # 강점 추출 (긍정 비율 높고, 언급 횟수 5회 이상)
    strengths = []
    weaknesses = []

    for aspect_name, counts in aspect_stats.items():
        total = counts["긍정"] + counts["부정"] + counts["중립"]
        if total < 3:  # 언급 너무 적으면 제외
            continue

        pos_ratio = counts["긍정"] / total * 100 if total > 0 else 0
        neg_ratio = counts["부정"] / total * 100 if total > 0 else 0

        if pos_ratio >= 60 and counts["긍정"] >= 3:
            strengths.append((aspect_name, counts["긍정"], pos_ratio))

        if neg_ratio >= 40 and counts["부정"] >= 2:
            weaknesses.append((aspect_name, counts["부정"], neg_ratio))

    # 긍정/부정 비율 기준 정렬
    strengths.sort(key=lambda x: (-x[2], -x[1]))  # 비율 높은 순, 개수 많은 순
    weaknesses.sort(key=lambda x: (-x[2], -x[1]))

    return {
        "strengths": strengths[:5],  # 상위 5개
        "weaknesses": weaknesses[:3],  # 상위 3개
        "aspect_stats": dict(aspect_stats),
    }


def generate_verdict_reasons(product: Product, analysis: dict) -> tuple[str, str]:
    """분석 결과를 기반으로 추천 판단과 이유 생성.

    Args:
        product: 제품 정보
        analysis: analyze_product_aspects() 결과

    Returns:
        (verdict, verdict_detail) 튜플
    """
    sentiment_ratio = product.get_sentiment_ratio()
    avg_rating = product.avg_rating
    pos_ratio = sentiment_ratio["긍정"]
    neg_ratio = sentiment_ratio["부정"]

    strengths = analysis["strengths"]
    weaknesses = analysis["weaknesses"]

    # 강점/약점 텍스트 생성
    strength_texts = []
    for name, count, ratio in strengths[:3]:
        strength_texts.append(f"**{name}** 만족도 높음 ({ratio:.0f}%)")

    weakness_texts = []
    for name, count, ratio in weaknesses[:2]:
        weakness_texts.append(f"**{name}** 불만 있음 ({ratio:.0f}%)")

    # 판정 기준
    if avg_rating >= 4.0 and pos_ratio >= 60:
        verdict = "👍 추천해요!"
        verdict_color = "success"

        if strength_texts:
            detail = "✅ " + " | ".join(strength_texts)
        else:
            detail = f"✅ 긍정 리뷰 {pos_ratio:.0f}%, 평점 {avg_rating:.1f}점"

        if weakness_texts:
            detail += "\n\n⚠️ 참고: " + ", ".join(weakness_texts)

    elif avg_rating >= 3.5 or pos_ratio >= 50:
        verdict = "🤔 괜찮아요"
        verdict_color = "info"

        details = []
        if strength_texts:
            details.append("✅ " + " | ".join(strength_texts[:2]))
        if weakness_texts:
            details.append("⚠️ " + " | ".join(weakness_texts[:2]))

        if details:
            detail = "\n\n".join(details)
        else:
            detail = f"긍정 {pos_ratio:.0f}% / 부정 {neg_ratio:.0f}%로 평가가 엇갈려요"

    else:
        verdict = "⚠️ 신중히 고려하세요"
        verdict_color = "warning"

        if weakness_texts:
            detail = "❌ " + " | ".join(weakness_texts)
        else:
            detail = f"부정 리뷰 {neg_ratio:.0f}%로 불만이 많아요"

        if strength_texts:
            detail += "\n\n✅ 그래도: " + strength_texts[0]

    return verdict, verdict_color, detail


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
/* 화면 깜빡임 및 투명도 변경 방지 */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #ffffff !important;
}
.stApp {
    background-color: #ffffff !important;
}
/* rerun 시 깜빡임 최소화 */
[data-testid="stAppViewContainer"] > section {
    background-color: #ffffff !important;
}
.main .block-container {
    background-color: #ffffff !important;
}
/* rerun 중 투명도 변경 방지 */
.stApp * {
    opacity: 1 !important;
    transition: none !important;
}
/* 요소 숨김 시에도 레이아웃 유지 */
[data-stale="true"] {
    opacity: 1 !important;
}
/* fragment 업데이트 시 다른 영역 투명도 유지 */
.element-container, .stMarkdown, .stExpander, [data-testid="stVerticalBlock"] {
    opacity: 1 !important;
}

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

/* 페이지네이션 버튼 스타일 */
.stButton > button {
    min-width: 40px;
    height: 40px;
    border-radius: 8px !important;
    font-weight: 500;
}

/* expander 내부 페이지네이션 버튼 간격 */
[data-testid="stExpander"] .stButton > button {
    margin: 2px;
    padding: 8px 12px;
}
</style>
""", unsafe_allow_html=True)


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

    # Q&A 피드백 상태 (메시지별)
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}  # {f"{product}_{idx}": True}

init_session_state()


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

def load_products(category: str):
    """제품 목록 로드 (SQLite DB에서)."""
    with st.spinner("📦 제품 로드 중..."):
        try:
            # DB 초기화
            init_db()

            cat_filter = None if category == "전체" else category

            # SQLite에서 전체 제품 목록 조회
            product_records = db_get_products()

            # ProductRecord를 Product 객체로 변환
            products = []
            for p in product_records:
                # 카테고리 필터 적용 (대분류 기준)
                if cat_filter and p.category != cat_filter:
                    continue

                # 리뷰 3개 이상인 제품만 포함
                if p.review_count >= 3:
                    # 실제 감정 분포 조회
                    sentiment_stats = get_review_count(p.name)
                    sentiment_dist = {
                        "긍정": sentiment_stats.get("긍정", 0),
                        "중립": sentiment_stats.get("중립", 0),
                        "부정": sentiment_stats.get("부정", 0),
                    }

                    product = Product(
                        name=p.name,
                        category=p.category,  # 대분류 (가전, 패션 등)
                        main_category=p.main_category,  # 소분류 (영상/음향가전 등)
                        review_count=p.review_count,
                        avg_rating=p.avg_rating,
                        sentiment_distribution=sentiment_dist,
                        top_aspects=[],
                        reviews=[],
                    )
                    products.append(product)

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

    products = st.session_state.products

    # 상단 필터 (대분류 + 소분류 + 검색 + 정렬)
    col_cat, col_subcat, col_search, col_sort = st.columns([1, 1.2, 2, 1])

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
            ["리뷰 많은순", "리뷰 적은순", "평점 높은순", "평점 낮은순"],
        )

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
    elif sort_option == "리뷰 적은순":
        filtered_products.sort(key=lambda p: p.review_count)
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
    sentiment_ratio = product.get_sentiment_ratio()
    positive_ratio = sentiment_ratio["긍정"]
    avg_rating = product.avg_rating

    # 상세 페이지와 동일한 verdict 로직
    if avg_rating >= 4.0 and positive_ratio >= 60:
        verdict = "👍 추천해요!"
    elif avg_rating >= 3.5 or positive_ratio >= 50:
        verdict = "🤔 괜찮아요"
    else:
        verdict = "⚠️ 신중히 고려하세요"

    # 제품 제목
    display_name = product.name[:28] + "..." if len(product.name) > 28 else product.name
    st.markdown(f"**📦 {display_name}**")

    # 카테고리
    st.caption(f"{product.category} > {product.main_category}")

    # 평점 & 리뷰 수
    st.markdown(f"⭐ **{avg_rating:.1f}** · 📝 **{product.review_count}개**")

    # 추천 판단 (상세 페이지와 동일 형식)
    st.markdown(verdict)

    # 주요 속성 태그
    aspects = product.top_aspects[:3] if product.top_aspects else []
    if aspects:
        tags = " ".join([f"`{a}`" for a in aspects])
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

    # 리뷰가 로드되지 않았으면 DB에서 로드
    if not product.reviews:
        db_reviews = get_reviews_by_product(product.name)
        for r in db_reviews:
            # sentiment → general_polarity 변환
            polarity_map = {"긍정": 1, "중립": 0, "부정": -1}
            polarity = polarity_map.get(r["sentiment"], 0)

            # rating → review_score 변환 (5점 → 100점)
            score = int((r["rating"] or 3) * 20)

            review = AIHubReview(
                index=str(r["id"]),
                raw_text=r["text"],
                source="쇼핑몰",
                domain=product.main_category,
                main_category=product.category,
                product_name=product.name,
                review_score=score,
                general_polarity=polarity,
                aspects=r["aspects"] or [],
                date=r["created_at"],
            )
            product.reviews.append(review)

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

    # 제품 상세 렌더링
    render_product_detail_content(product)


def render_product_detail_content(product: Product):
    """소비자 모드 - 제품 상세 페이지 (간단한 구매 결정 도움)."""
    # 한눈에 보는 평가 카드
    st.subheader("📋 한눈에 보기")

    # 속성 기반 분석
    analysis = analyze_product_aspects(product)
    verdict, verdict_color, verdict_detail = generate_verdict_reasons(product, analysis)

    avg_rating = product.avg_rating

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

    # 리뷰 작성하기
    render_add_review(product)

    st.markdown("---")

    # 카카오톡 스타일 Q&A 채팅
    st.subheader("💬 AI에게 물어보세요")

    # Q&A 사용 통계 표시 (제품별)
    qa_metrics = get_qa_metrics(product_name=product.name)

    # 캐시 통계 가져오기
    try:
        cache = get_semantic_cache()
        cache_stats = cache.get_stats()
    except Exception:
        cache_stats = None

    if qa_metrics["total_questions"] > 0:
        avg_time = qa_metrics["avg_response_time_ms"]
        avg_time_str = f"{avg_time / 1000:.1f}초" if avg_time else "-"

        # 인기 질문 Top 3 표시
        top_kws = qa_metrics.get("top_keywords", [])
        if top_kws:
            kw_parts = [f"{kw['keyword']}({kw['count']})" for kw in top_kws]
            kw_str = f" · 인기: {', '.join(kw_parts)}"
        else:
            kw_str = ""

        # 캐시 히트율 표시
        cache_str = ""
        if cache_stats and cache_stats.total_hits > 0:
            cache_str = f" · ⚡ 캐시 히트 {cache_stats.hit_rate}%"
            if cache_stats.estimated_savings_usd > 0:
                cache_str += f" (${cache_stats.estimated_savings_usd:.3f} 절감)"

        st.caption(
            f"📊 이 제품 **{qa_metrics['total_questions']}개** 질문 · "
            f"평균 응답 **{avg_time_str}**{kw_str}{cache_str}"
        )
    else:
        st.caption("💡 세션이 종료되면 대화 내용이 사라져요!")

    @st.fragment
    def render_qa_fragment():
        """Q&A 섹션 - 카카오톡 스타일 채팅 인터페이스."""
        # 대화 기록 초기화 (제품별, 세션별로 독립)
        chat_key = f"chat_history_{product.name}"
        pending_key = f"pending_answer_{product.name}"

        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        chat_history = st.session_state.get(chat_key, [])

        # 채팅 영역 (고정 높이, 스크롤 가능)
        chat_container = st.container(height=300)

        with chat_container:
            if not chat_history:
                st.info("💬 리뷰에 대해 궁금한 점을 물어보세요!")
            else:
                for chat in chat_history:
                    # 사용자 질문
                    with st.chat_message("user"):
                        st.write(chat['question'])

                    # AI 답변
                    with st.chat_message("assistant"):
                        if chat['answer'] == "💭 답변 준비중...":
                            # 로딩 애니메이션 표시 (텍스트 뒤에 스피너, 세로 가운데 정렬)
                            st.markdown(
                                """
                                <style>
                                .loading-container {
                                    display: inline-flex;
                                    align-items: center;
                                    gap: 8px;
                                }
                                .loading-spinner {
                                    width: 14px;
                                    height: 14px;
                                    border: 2px solid #e0e0e0;
                                    border-top: 2px solid #1565c0;
                                    border-radius: 50%;
                                    animation: spin 0.8s linear infinite;
                                }
                                .loading-label {
                                    color: #555;
                                    font-size: 0.95em;
                                }
                                @keyframes spin {
                                    0% { transform: rotate(0deg); }
                                    100% { transform: rotate(360deg); }
                                }
                                </style>
                                <div class="loading-container">
                                    <span class="loading-label">리뷰 분석중</span>
                                    <div class="loading-spinner"></div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.write(chat['answer'])

                        # 근거 리뷰 토글 + 피드백 버튼 (같은 줄)
                        sources = chat.get("sources", [])
                        if chat['answer'] != "💭 답변 준비중...":
                            chat_idx = chat_history.index(chat)
                            feedback_key = f"{product.name}_qa_{chat_idx}"

                        if sources and chat['answer'] != "💭 답변 준비중...":
                            # 근거 리뷰 팝오버 + 피드백 버튼 (인라인)
                            feedback_value = st.session_state.feedback_given.get(feedback_key)
                            if feedback_value is not None:
                                # 피드백 완료 상태 - 비활성화된 버튼 표시
                                btn_cols = st.columns([2.5, 0.5, 0.5, 5.5])
                                with btn_cols[0]:
                                    with st.popover(f"📚 근거 리뷰 ({len(sources)}개)"):
                                        st.caption("💡 AI가 답변을 생성할 때 참고한 리뷰들입니다")

                                        # AI 응답에서 인용 문구 추출 (하이라이트용)
                                        import re
                                        answer = chat['answer']
                                        question = chat['question']

                                        # 1. AI 응답에서 따옴표 안의 문구 추출
                                        quoted_phrases = re.findall(r'["""]([^"""]+)["""]', answer)
                                        # 짧은 문구만 필터 (3자 이상, 50자 이하)
                                        quoted_phrases = [p.strip() for p in quoted_phrases if 3 <= len(p.strip()) <= 50]

                                        # 2. 질문에서 키워드도 추출 (fallback용)
                                        stopwords = {"이", "가", "은", "는", "을", "를", "의", "에", "에서", "로", "으로", "와", "과", "도", "만", "이나", "나", "고", "하고", "해서", "어떤", "어떻", "뭐", "뭔", "좀", "잘", "더", "많이", "정말", "진짜", "너무", "아주", "매우", "제품", "상품", "이거", "저거", "그거", "있", "없", "하", "되", "같", "인가요", "인가", "예요", "에요", "나요", "까요"}
                                        keywords = [w for w in re.findall(r'[가-힣]+', question) if len(w) >= 2 and w not in stopwords]

                                        def highlight_text_fb(text: str, phrases: list, keywords: list) -> str:
                                            """인용 문구 또는 키워드를 하이라이트."""
                                            result = text
                                            highlighted_any = False

                                            # 1. 인용 문구 정확히 하이라이트
                                            for phrase in phrases:
                                                if phrase in result:
                                                    result = result.replace(
                                                        phrase,
                                                        f'<mark style="background-color: #fff3cd; padding: 2px 4px; border-radius: 4px;">{phrase}</mark>',
                                                        1
                                                    )
                                                    highlighted_any = True

                                            # 2. 인용 문구로 하이라이트 안 됐으면 키워드로 시도
                                            if not highlighted_any and keywords:
                                                for kw in keywords:
                                                    if kw in result:
                                                        result = result.replace(
                                                            kw,
                                                            f'<mark style="background-color: #e7f3ff; padding: 2px 4px; border-radius: 4px;">{kw}</mark>',
                                                            1
                                                        )
                                                        highlighted_any = True
                                                        break

                                            return result

                                        for j, src in enumerate(sources, 1):
                                            content = src.get("text", src.get("content", "내용 없음"))
                                            rating = src.get("rating", "N/A")

                                            # 하이라이트 적용
                                            highlighted_content = highlight_text_fb(content, quoted_phrases, keywords)

                                            # 별점 이모지
                                            try:
                                                rating_int = int(rating)
                                                stars = "⭐" * rating_int
                                            except (ValueError, TypeError):
                                                stars = f"별점: {rating}"

                                            # 하이라이트가 포함된 경우 HTML로 렌더링
                                            if '<mark' in highlighted_content:
                                                st.markdown(f"**{j}. {stars}**", unsafe_allow_html=True)
                                                st.markdown(f'<div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 10px;">{highlighted_content}</div>', unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"**{j}. {stars}**")
                                                st.info(content)
                                with btn_cols[1]:
                                    # 👍 선택됨 - 핑크 배경으로 강조
                                    if feedback_value == 1:
                                        st.markdown(
                                            '<span style="display: inline-block; background-color: #FCE4EC; padding: 4px 10px; border-radius: 8px;">👍</span>',
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(
                                            '<span style="display: inline-block; background-color: #f5f5f5; padding: 4px 10px; border-radius: 8px; opacity: 0.4;">👍</span>',
                                            unsafe_allow_html=True
                                        )
                                with btn_cols[2]:
                                    # 👎 선택됨 - 핑크 배경으로 강조
                                    if feedback_value == -1:
                                        st.markdown(
                                            '<span style="display: inline-block; background-color: #FCE4EC; padding: 4px 10px; border-radius: 8px;">👎</span>',
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(
                                            '<span style="display: inline-block; background-color: #f5f5f5; padding: 4px 10px; border-radius: 8px; opacity: 0.4;">👎</span>',
                                            unsafe_allow_html=True
                                        )
                            else:
                                # 피드백 대기 상태
                                btn_cols = st.columns([2.5, 0.4, 0.4, 5.7])
                                with btn_cols[0]:
                                    with st.popover(f"📚 근거 리뷰 ({len(sources)}개)"):
                                        st.caption("💡 AI가 답변을 생성할 때 참고한 리뷰들입니다")

                                        # AI 응답에서 인용 문구 추출 (하이라이트용)
                                        import re
                                        answer = chat['answer']
                                        question = chat['question']

                                        # 1. AI 응답에서 따옴표 안의 문구 추출
                                        quoted_phrases = re.findall(r'["""]([^"""]+)["""]', answer)
                                        # 짧은 문구만 필터 (3자 이상, 50자 이하)
                                        quoted_phrases = [p.strip() for p in quoted_phrases if 3 <= len(p.strip()) <= 50]

                                        # 2. 질문에서 키워드도 추출 (fallback용)
                                        stopwords = {"이", "가", "은", "는", "을", "를", "의", "에", "에서", "로", "으로", "와", "과", "도", "만", "이나", "나", "고", "하고", "해서", "어떤", "어떻", "뭐", "뭔", "좀", "잘", "더", "많이", "정말", "진짜", "너무", "아주", "매우", "제품", "상품", "이거", "저거", "그거", "있", "없", "하", "되", "같", "인가요", "인가", "예요", "에요", "나요", "까요"}
                                        keywords = [w for w in re.findall(r'[가-힣]+', question) if len(w) >= 2 and w not in stopwords]

                                        def highlight_text_nf(text: str, phrases: list, keywords: list) -> str:
                                            """인용 문구 또는 키워드를 하이라이트."""
                                            result = text
                                            highlighted_any = False

                                            # 1. 인용 문구 정확히 하이라이트
                                            for phrase in phrases:
                                                if phrase in result:
                                                    result = result.replace(
                                                        phrase,
                                                        f'<mark style="background-color: #fff3cd; padding: 2px 4px; border-radius: 4px;">{phrase}</mark>',
                                                        1
                                                    )
                                                    highlighted_any = True

                                            # 2. 인용 문구로 하이라이트 안 됐으면 키워드로 시도
                                            if not highlighted_any and keywords:
                                                for kw in keywords:
                                                    if kw in result:
                                                        result = result.replace(
                                                            kw,
                                                            f'<mark style="background-color: #e7f3ff; padding: 2px 4px; border-radius: 4px;">{kw}</mark>',
                                                            1
                                                        )
                                                        highlighted_any = True
                                                        break

                                            return result

                                        for j, src in enumerate(sources, 1):
                                            content = src.get("text", src.get("content", "내용 없음"))
                                            rating = src.get("rating", "N/A")

                                            # 하이라이트 적용
                                            highlighted_content = highlight_text_nf(content, quoted_phrases, keywords)

                                            # 별점 이모지
                                            try:
                                                rating_int = int(rating)
                                                stars = "⭐" * rating_int
                                            except (ValueError, TypeError):
                                                stars = f"별점: {rating}"

                                            # 하이라이트가 포함된 경우 HTML로 렌더링
                                            if '<mark' in highlighted_content:
                                                st.markdown(f"**{j}. {stars}**", unsafe_allow_html=True)
                                                st.markdown(f'<div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 10px;">{highlighted_content}</div>', unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"**{j}. {stars}**")
                                                st.info(content)
                                with btn_cols[1]:
                                    if st.button("👍", key=f"helpful_{feedback_key}", help="도움이 됐어요"):
                                        save_qa_feedback(product.name, chat['question'], chat['answer'], 1)
                                        st.session_state.feedback_given[feedback_key] = 1  # 어떤 피드백인지 저장
                                        st.toast("✅ 피드백 감사합니다!", icon="👍")
                                        st.rerun(scope="app")  # fragment 내부이므로 전체 앱 리런
                                with btn_cols[2]:
                                    if st.button("👎", key=f"not_helpful_{feedback_key}", help="도움이 안 됐어요"):
                                        save_qa_feedback(product.name, chat['question'], chat['answer'], -1)
                                        st.session_state.feedback_given[feedback_key] = -1  # 어떤 피드백인지 저장
                                        st.toast("✅ 피드백 감사합니다!", icon="👎")
                                        st.rerun(scope="app")  # fragment 내부이므로 전체 앱 리런
                        elif sources and chat['answer'] == "💭 답변 준비중...":
                            # 답변 준비중일 때는 팝오버만 표시 (버튼 없음)
                            with st.popover(f"📚 근거 리뷰 ({len(sources)}개)"):
                                st.caption("💡 AI가 답변을 생성할 때 참고한 리뷰들입니다")
                                for j, src in enumerate(sources, 1):
                                    content = src.get("text", src.get("content", "내용 없음"))
                                    rating = src.get("rating", "N/A")
                                    try:
                                        rating_int = int(rating)
                                        stars = "⭐" * rating_int
                                    except (ValueError, TypeError):
                                        stars = f"별점: {rating}"
                                    st.markdown(f"**{j}. {stars}**")
                                    st.info(content)

        # 자주 묻는 질문 버튼
        faq_col1, faq_col2, faq_col3 = st.columns(3)
        with faq_col1:
            if st.button("📦 배송", use_container_width=True, key="faq_delivery"):
                st.session_state.b2c_question = "배송은 어떤가요? 빠른 편인가요?"
        with faq_col2:
            if st.button("👍 장점", use_container_width=True, key="faq_pros"):
                st.session_state.b2c_question = "이 제품의 장점은 무엇인가요?"
        with faq_col3:
            if st.button("⚠️ 단점", use_container_width=True, key="faq_cons"):
                st.session_state.b2c_question = "이 제품의 주요 단점이 뭔가요?"

        # 질문 입력 (하단) - 동적 key로 입력창 초기화
        input_col, btn_col = st.columns([5, 1])
        with input_col:
            user_question = st.text_input(
                "질문",
                placeholder="궁금한 점을 입력하세요...",
                key=f"b2c_user_question_{len(chat_history)}",
                label_visibility="collapsed"
            )
        with btn_col:
            send_clicked = st.button("전송", use_container_width=True, type="primary")

        # FAQ 버튼 또는 전송 버튼 클릭 처리
        question_to_ask = getattr(st.session_state, "b2c_question", None)
        if not question_to_ask and send_clicked and user_question:
            question_to_ask = user_question

        # 1단계: 새 질문 접수 → 로딩 상태로 먼저 표시
        is_new_question = (
            question_to_ask
            and question_to_ask != st.session_state.get("b2c_last_question")
            and not st.session_state.get(pending_key)
        )

        if is_new_question:
            if "b2c_question" in st.session_state:
                del st.session_state.b2c_question

            # 질문 즉시 추가 (로딩 상태)
            st.session_state[chat_key].append({
                "question": question_to_ask,
                "answer": "💭 답변 준비중...",
                "sources": []
            })
            st.session_state.b2c_last_question = question_to_ask
            st.session_state[pending_key] = question_to_ask
            st.rerun()  # 로딩 상태 먼저 표시

        # 2단계: 로딩 상태에서 실제 AI 응답 생성
        if st.session_state.get(pending_key):
            pending_question = st.session_state[pending_key]

            try:
                import time
                start_time = time.time()

                rag_chain = get_or_create_product_rag_chain(product)
                if rag_chain:
                    response = rag_chain.query_with_sources(pending_question)
                    answer = response["answer"]
                    sources = response.get("sources", [])

                    # 응답 시간 계산 (ms) 및 로그 저장
                    response_time_ms = int((time.time() - start_time) * 1000)
                    save_qa_log(product.name, pending_question, response_time_ms)
                else:
                    answer = "RAG 시스템을 초기화할 수 없습니다."
                    sources = []
            except Exception as e:
                answer = f"오류가 발생했습니다: {e}"
                sources = []

            # 마지막 대화 업데이트
            if st.session_state[chat_key]:
                st.session_state[chat_key][-1] = {
                    "question": pending_question,
                    "answer": answer,
                    "sources": sources
                }

            # 로딩 상태 해제
            del st.session_state[pending_key]
            st.rerun()  # 완료된 답변 표시

    # Q&A fragment 실행
    render_qa_fragment()

    st.markdown("---")

    # 속성별 상세 리뷰 (B2B와 동일한 형식)
    st.subheader("🏷️ 속성별 상세 리뷰")

    from collections import defaultdict

    # 속성별 감정 분석 및 리뷰 수집
    aspect_sentiment: dict[str, dict[str, int]] = defaultdict(lambda: {"긍정": 0, "부정": 0, "중립": 0})
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
                        "all_aspects": review.aspects,  # 전체 속성 분석용
                        "review_score": review.review_score,  # 별점용
                    })

    # 데이터프레임 생성 (언급 횟수 기준 정렬)
    aspect_data = []
    for aspect_name, sentiments in aspect_sentiment.items():
        total = sum(sentiments.values())
        aspect_data.append({
            "속성": aspect_name,
            "긍정": sentiments["긍정"],
            "부정": sentiments["부정"],
            "중립": sentiments["중립"],
            "총합": total,
        })

    if aspect_data:
        import pandas as pd
        df = pd.DataFrame(aspect_data).sort_values("총합", ascending=False)

        # 속성별 expander (각각 내부에 토글)
        for idx, row in df.iterrows():
            aspect = row["속성"]
            total = row["총합"]
            pos_ratio = row["긍정"] / total * 100 if total > 0 else 0
            neg_ratio = row["부정"] / total * 100 if total > 0 else 0
            neu_ratio = row["중립"] / total * 100 if total > 0 else 0

            with st.expander(f"**{aspect}** ({total}회) - 긍정 {pos_ratio:.0f}% / 부정 {neg_ratio:.0f}% / 중립 {neu_ratio:.0f}%"):
                all_reviews = aspect_reviews.get(aspect, [])

                if not all_reviews:
                    st.caption("리뷰 텍스트가 없어요")
                    continue

                # 감정 필터 토글 (속성별 독립) + 색상 레전드
                filter_cols = st.columns(3)
                with filter_cols[0]:
                    st.markdown('<span style="background-color: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 3px; font-weight: bold;">😊 긍정</span>', unsafe_allow_html=True)
                    show_pos = st.toggle("긍정 표시", value=True, key=f"pos_{aspect}", label_visibility="collapsed")
                with filter_cols[1]:
                    st.markdown('<span style="background-color: #ffebee; color: #c62828; padding: 2px 8px; border-radius: 3px; font-weight: bold;">😞 부정</span>', unsafe_allow_html=True)
                    show_neg = st.toggle("부정 표시", value=True, key=f"neg_{aspect}", label_visibility="collapsed")
                with filter_cols[2]:
                    st.markdown('<span style="background-color: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 3px; font-weight: bold;">😐 중립</span>', unsafe_allow_html=True)
                    show_neu = st.toggle("중립 표시", value=True, key=f"neu_{aspect}", label_visibility="collapsed")

                # 선택된 감정 필터링
                selected_sentiments = []
                if show_pos:
                    selected_sentiments.append("긍정")
                if show_neg:
                    selected_sentiments.append("부정")
                if show_neu:
                    selected_sentiments.append("중립")

                filtered_reviews = [r for r in all_reviews if r["sentiment"] in selected_sentiments]

                if not filtered_reviews:
                    st.info("선택한 감정의 리뷰가 없어요")
                    continue

                # 페이지네이션 설정
                reviews_per_page = 5
                total_reviews = len(filtered_reviews)
                total_pages = (total_reviews + reviews_per_page - 1) // reviews_per_page

                # 페이지 상태 키
                page_key = f"aspect_page_{aspect}"
                if page_key not in st.session_state:
                    st.session_state[page_key] = 0

                current_page = st.session_state[page_key]

                st.caption(f"총 {total_reviews}개 리뷰")

                # 현재 페이지 리뷰
                start_idx = current_page * reviews_per_page
                end_idx = min(start_idx + reviews_per_page, total_reviews)
                page_reviews = filtered_reviews[start_idx:end_idx]

                for rv_idx, review_data in enumerate(page_reviews):
                    highlighted_html = highlight_aspect_in_text(
                        review_data["full_text"],
                        review_data["aspect_text"],
                        review_data["sentiment"]
                    )

                    # 감정별 색상
                    sentiment_color = {"긍정": "#1976D2", "중립": "#388E3C", "부정": "#D32F2F"}.get(review_data["sentiment"], "#666")
                    border_color = {"긍정": "#bbdefb", "중립": "#c8e6c9", "부정": "#ffcdd2"}.get(review_data["sentiment"], "#ddd")

                    # 별점 (100점 → 5점) - "⭐ 5.0" 형태로 통일
                    r_score = review_data.get("review_score", 0)
                    if r_score > 0:
                        star_count = min(5, max(1, round(r_score / 20)))
                        stars_str = f"⭐ {star_count} "
                    else:
                        stars_str = ""

                    # 신뢰도 검사
                    trust_result = check_review_text(review_data["full_text"], star_count if r_score > 0 else None)
                    trust_label = ' <span style="color: #F57C00; font-weight: bold;">[의심]</span>' if trust_result.is_suspicious else ""

                    # 의심 리뷰 경고 HTML
                    warning_html = ""
                    if trust_result.is_suspicious:
                        reason_map = {
                            "excessive_praise": "과도한 칭찬",
                            "spam_keywords": "스팸/광고",
                            "too_short": "너무 짧음",
                            "repetitive_pattern": "반복 문구",
                            "no_specifics": "구체성 부족",
                            "extreme_rating": "평점-내용 불일치",
                        }
                        reasons = [reason_map.get(r.value, r.value) for r in trust_result.reasons]
                        warning_html = f'<div style="background-color: #FFF3E0; padding: 8px 12px; border-radius: 5px; margin-bottom: 10px; color: #E65100;">⚠️ 의심 사유: {", ".join(reasons)}</div>'

                    # 미리보기 (속성 텍스트에 하이라이트 적용)
                    preview_text = review_data["aspect_text"] if review_data["aspect_text"] else review_data["full_text"]
                    preview_raw = preview_text[:50] + "..." if len(preview_text) > 50 else preview_text

                    # 미리보기에 감정별 배경색 하이라이트 적용
                    highlight_bg = {"긍정": "#e3f2fd", "중립": "#e8f5e9", "부정": "#ffebee"}.get(review_data["sentiment"], "#f5f5f5")
                    preview = f'<span style="background-color: {highlight_bg}; padding: 2px 6px; border-radius: 4px;">{preview_raw}</span>'

                    # 속성 분석 HTML 생성
                    aspects_html = ""
                    all_aspects = review_data.get("all_aspects", [])
                    if all_aspects:
                        aspects_html = '<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;"><b>🏷️ 속성 분석</b>'
                        for asp in all_aspects:
                            a_name = asp.get("Aspect", "")
                            a_polarity = asp.get("SentimentPolarity", 0)
                            a_text = asp.get("SentimentText", "")
                            if not a_name:
                                continue
                            a_label = polarity_map.get(a_polarity, polarity_map.get(str(a_polarity), "중립"))
                            a_emoji = {"긍정": "👍", "중립": "➖", "부정": "👎"}.get(a_label, "")
                            a_bg = {"긍정": "#e3f2fd", "중립": "#f5f5f5", "부정": "#ffebee"}.get(a_label, "#f5f5f5")
                            aspects_html += f'<div style="background-color: {a_bg}; padding: 6px 10px; border-radius: 5px; margin-top: 6px;">{a_emoji} <b>{a_name}</b>: {a_text}</div>'
                        aspects_html += '</div>'

                    # 페이지+인덱스 기반 고유 ID로 페이지 변경 시 토글 상태 리셋
                    detail_id = f"aspect_{aspect}_p{current_page}_i{rv_idx}"
                    st.markdown(f'''
<details id="{detail_id}" style="margin-bottom: 8px; border: 1px solid {border_color}; border-radius: 8px;">
<summary style="padding: 10px; cursor: pointer; color: {sentiment_color}; font-weight: 500; list-style: none;">
{stars_str}{preview}{trust_label}
</summary>
<div style="padding: 12px; background-color: #f8f9fa;">
{warning_html}
{highlighted_html}
{aspects_html}
</div>
</details>
''', unsafe_allow_html=True)

                # 페이지네이션 UI (10개 버튼 그룹)
                if total_pages > 1:
                    pages_per_group = 10
                    current_group = current_page // pages_per_group
                    group_start = current_group * pages_per_group
                    group_end = min(group_start + pages_per_group, total_pages)

                    # 버튼 컬럼 계산: < [페이지들] >
                    num_page_buttons = group_end - group_start
                    cols = st.columns([1] + [1] * num_page_buttons + [1])

                    # < 이전 그룹 버튼
                    with cols[0]:
                        if current_group > 0:
                            if st.button("◀", key=f"prev_group_{aspect}"):
                                st.session_state[page_key] = group_start - 1
                                st.rerun()

                    # 페이지 번호 버튼들
                    for i, page_num in enumerate(range(group_start, group_end)):
                        with cols[i + 1]:
                            is_current = page_num == current_page
                            if is_current:
                                st.button(f"**{page_num + 1}**", key=f"page_{aspect}_{page_num}", disabled=True)
                            else:
                                if st.button(str(page_num + 1), key=f"page_{aspect}_{page_num}"):
                                    st.session_state[page_key] = page_num
                                    st.rerun()

                    # > 다음 그룹 버튼
                    with cols[-1]:
                        if group_end < total_pages:
                            if st.button("▶", key=f"next_group_{aspect}"):
                                st.session_state[page_key] = group_end
                                st.rerun()
    else:
        st.info("속성 분석 데이터가 없어요")

    st.markdown("---")

    # 대표 리뷰
    st.subheader("💬 대표 리뷰")

    # 신뢰도 높은 리뷰만 필터링 후 긍정/부정 각 2개
    def is_trusted_review(r) -> bool:
        result = check_review_text(r.raw_text, None)
        return not result.is_suspicious

    trusted_positive = [r for r in product.reviews if r.general_polarity == 1 and is_trusted_review(r)]
    trusted_negative = [r for r in product.reviews if r.general_polarity == -1 and is_trusted_review(r)]

    positive_reviews = trusted_positive[:2]
    negative_reviews = trusted_negative[:2]

    # 별점 변환 헬퍼 (100점 → 5점) - "⭐ 5" 형태로 통일
    def get_stars_from_score(score: int) -> str:
        if score <= 0:
            return ""
        star_count = min(5, max(1, round(score / 20)))
        return f"⭐ {star_count} "

    # 속성 분석 HTML 생성 헬퍼
    def build_aspects_html(aspects: list) -> str:
        if not aspects:
            return ""
        polarity_map = {"1": "긍정", "0": "중립", "-1": "부정", 1: "긍정", 0: "중립", -1: "부정"}
        html = '<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;"><b>🏷️ 속성 분석</b>'
        for asp in aspects:
            name = asp.get("Aspect", "")
            polarity = asp.get("SentimentPolarity", 0)
            text = asp.get("SentimentText", "")
            if not name:
                continue
            label = polarity_map.get(polarity, polarity_map.get(str(polarity), "중립"))
            emoji = {"긍정": "👍", "중립": "➖", "부정": "👎"}.get(label, "")
            bg = {"긍정": "#e3f2fd", "중립": "#f5f5f5", "부정": "#ffebee"}.get(label, "#f5f5f5")
            html += f'<div style="background-color: {bg}; padding: 6px 10px; border-radius: 5px; margin-top: 6px;">{emoji} <b>{name}</b>: {text}</div>'
        html += '</div>'
        return html

    review_col1, review_col2 = st.columns(2)

    with review_col1:
        st.markdown('<span style="background-color: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 3px; font-weight: bold;">긍정 리뷰</span>', unsafe_allow_html=True)
        if positive_reviews:
            for r in positive_reviews:
                stars = get_stars_from_score(r.review_score)
                preview = r.raw_text[:35] + "..." if len(r.raw_text) > 35 else r.raw_text
                aspects_html = build_aspects_html(r.aspects)
                st.markdown(f'''
<details style="margin-bottom: 8px; border: 1px solid #bbdefb; border-radius: 8px;">
<summary style="padding: 10px; cursor: pointer; color: #1976D2; font-weight: 500; list-style: none;">
{stars}{preview}
</summary>
<div style="padding: 12px; background-color: #fff;">
"{r.raw_text}"
{aspects_html}
</div>
</details>
''', unsafe_allow_html=True)
        else:
            st.caption("긍정 리뷰가 없어요")

    with review_col2:
        st.markdown('<span style="background-color: #ffebee; color: #c62828; padding: 2px 8px; border-radius: 3px; font-weight: bold;">부정 리뷰</span>', unsafe_allow_html=True)
        if negative_reviews:
            for r in negative_reviews:
                stars = get_stars_from_score(r.review_score)
                preview = r.raw_text[:35] + "..." if len(r.raw_text) > 35 else r.raw_text
                aspects_html = build_aspects_html(r.aspects)
                st.markdown(f'''
<details style="margin-bottom: 8px; border: 1px solid #ffcdd2; border-radius: 8px;">
<summary style="padding: 10px; cursor: pointer; color: #D32F2F; font-weight: 500; list-style: none;">
{stars}{preview}
</summary>
<div style="padding: 12px; background-color: #fff;">
"{r.raw_text}"
{aspects_html}
</div>
</details>
''', unsafe_allow_html=True)
        else:
            st.caption("부정 리뷰가 없어요")

    st.markdown("---")

    # 전체 리뷰 보기
    st.subheader("📋 전체 리뷰")
    st.markdown('<span style="color: #1976D2;">■ 긍정</span> | <span style="color: #388E3C;">■ 중립</span> | <span style="color: #D32F2F;">■ 부정</span>', unsafe_allow_html=True)
    st.caption("💡 [의심] 표시는 과도한 칭찬, 광고성 문구 등이 감지된 리뷰입니다.")
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
    import re

    # 감정별 스타일
    styles = {
        "긍정": "background-color: #e3f2fd; color: #1565c0; font-weight: bold; padding: 2px 4px; border-radius: 3px;",
        "부정": "background-color: #ffebee; color: #c62828; font-weight: bold; padding: 2px 4px; border-radius: 3px;",
        "중립": "background-color: #e8f5e9; color: #2e7d32; font-weight: bold; padding: 2px 4px; border-radius: 3px;",
    }

    style = styles.get(sentiment, styles["중립"])

    if not aspect_text:
        return html.escape(full_text)

    # 공백 정규화 (연속 공백 → 단일 공백)
    normalized_full = re.sub(r'\s+', ' ', full_text.strip())
    normalized_aspect = re.sub(r'\s+', ' ', aspect_text.strip())

    # HTML 이스케이프
    escaped_full = html.escape(normalized_full)
    escaped_aspect = html.escape(normalized_aspect)

    # 1. 정확한 매칭 시도
    if escaped_aspect in escaped_full:
        highlighted = escaped_full.replace(
            escaped_aspect,
            f'<span style="{style}">{escaped_aspect}</span>',
            1
        )
        return highlighted

    # 2. 공백 무시 유연 매칭 (aspect의 공백을 \s*로 변환)
    pattern_str = r'\s*'.join(re.escape(c) for c in normalized_aspect if c.strip())
    try:
        match = re.search(pattern_str, escaped_full, re.IGNORECASE)
        if match:
            matched_text = match.group()
            highlighted = escaped_full[:match.start()] + f'<span style="{style}">{matched_text}</span>' + escaped_full[match.end():]
            return highlighted
    except re.error:
        pass

    # 3. 핵심 키워드 매칭 (aspect에서 2자 이상 단어 추출하여 매칭)
    keywords = [w for w in re.findall(r'[가-힣]{2,}', normalized_aspect)]
    if keywords:
        # 가장 긴 키워드부터 시도
        keywords.sort(key=len, reverse=True)
        for kw in keywords[:3]:  # 최대 3개
            if kw in escaped_full:
                # 키워드가 포함된 절/구 찾기
                pattern = f'([^.!?]*{re.escape(kw)}[^.!?]*)'
                match = re.search(pattern, escaped_full)
                if match:
                    matched_text = match.group(1).strip()
                    if len(matched_text) <= 100:  # 너무 긴 매칭 방지
                        highlighted = escaped_full.replace(
                            matched_text,
                            f'<span style="{style}">{matched_text}</span>',
                            1
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


def render_qa_sources(sources: list[dict], key_prefix: str = "current"):
    """Q&A 근거 리뷰 표시 (개선된 버전 + 속성 분석).

    Args:
        sources: 출처 리뷰 목록
        key_prefix: expander 키 중복 방지용 접두사
    """
    # 출처 개수 표시
    if not sources:
        st.info("🔍 관련 리뷰를 찾지 못했습니다. 다른 질문을 시도해보세요.")
        return

    # 참고한 리뷰 개수 표시
    st.caption(f"📚 {len(sources)}개 리뷰 참고")

    # 감정 색상 매핑
    sentiment_colors = {
        "긍정": "#1565c0",
        "부정": "#c62828",
        "중립": "#2e7d32",
    }

    # 속성 감정 색상 (태그용)
    aspect_sentiment_colors = {
        "1": "#1565c0",   # 긍정 - 파랑
        1: "#1565c0",
        "-1": "#c62828",  # 부정 - 빨강
        -1: "#c62828",
        "0": "#666",      # 중립 - 회색
        0: "#666",
    }

    with st.expander(f"📚 근거 리뷰 ({len(sources)}개)", expanded=False):
        st.caption("💡 AI가 답변을 생성할 때 참고한 리뷰들입니다")

        for i, source in enumerate(sources, 1):
            text = source.get("text", "")
            rating = source.get("rating")

            # 가짜 리뷰 검사
            fake_result = check_review_text(text, rating)
            is_suspicious = fake_result.is_suspicious

            # DB에서 속성 분석 조회
            aspects = get_review_aspects_by_text(text)

            # 감정 추정 (별점 기반)
            if rating:
                if rating >= 4:
                    sentiment = "긍정"
                    emoji = "😊"
                elif rating <= 2:
                    sentiment = "부정"
                    emoji = "😞"
                else:
                    sentiment = "중립"
                    emoji = "😐"
            else:
                sentiment = "중립"
                emoji = "😐"

            color = sentiment_colors.get(sentiment, "#666")

            # 별점 표시
            rating_display = f"⭐ {rating}" if rating else "평점 없음"

            # 의심 라벨
            suspicious_label = " <span style='color: orange; font-weight: bold;'>[의심]</span>" if is_suspicious else ""

            # 속성 태그 HTML 생성
            aspect_tags_html = ""
            if aspects:
                tags = []
                for asp in aspects[:5]:  # 최대 5개까지만 표시
                    asp_name = asp.get("Aspect", "")
                    asp_polarity = asp.get("SentimentPolarity", 0)
                    asp_color = aspect_sentiment_colors.get(asp_polarity, "#666")
                    if asp_name:
                        tags.append(
                            f'<span style="display: inline-block; padding: 2px 8px; margin: 2px; '
                            f'border-radius: 12px; background-color: {asp_color}; color: white; '
                            f'font-size: 0.75em;">{asp_name}</span>'
                        )
                if tags:
                    aspect_tags_html = f'<div style="margin-top: 8px;">{"".join(tags)}</div>'

            # HTML 렌더링
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: bold; color: #333;">[{i}] {emoji} {sentiment}</span>
                        <span style="font-size: 0.85em; color: #666;">{rating_display}{suspicious_label}</span>
                    </div>
                    <div style="line-height: 1.6; color: #444;">
                        {text}
                    </div>
                    {aspect_tags_html}
                </div>
                """,
                unsafe_allow_html=True
            )

            # 의심 사유 표시
            if is_suspicious and fake_result.reasons:
                reason_text = ", ".join([r.value for r in fake_result.reasons])
                st.caption(f"⚠️ 의심 사유: {reason_text}")


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

    # 메시지 히스토리 표시 (출처 포함)
    messages = st.session_state.product_messages[product_name]
    for msg_idx, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 이전 대화의 출처도 표시
            if message["role"] == "assistant" and message.get("sources"):
                render_qa_sources(message["sources"], key_prefix=f"history_{msg_idx}")

            # assistant 메시지에 피드백 버튼 추가
            if message["role"] == "assistant":
                feedback_key = f"{product_name}_{msg_idx}"
                feedback_value = st.session_state.feedback_given.get(feedback_key)

                # 이전 user 메시지(질문) 찾기
                question = ""
                if msg_idx > 0 and messages[msg_idx - 1]["role"] == "user":
                    question = messages[msg_idx - 1]["content"]

                if feedback_value is not None:
                    # 피드백 완료 - 핑크 배경으로 선택 상태 표시
                    col1, col2, col3 = st.columns([1, 1, 6])
                    with col1:
                        if feedback_value == 1:
                            st.markdown(
                                '<span style="display: inline-block; background-color: #FCE4EC; padding: 4px 10px; border-radius: 8px;">👍</span>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<span style="display: inline-block; background-color: #f5f5f5; padding: 4px 10px; border-radius: 8px; opacity: 0.4;">👍</span>',
                                unsafe_allow_html=True
                            )
                    with col2:
                        if feedback_value == -1:
                            st.markdown(
                                '<span style="display: inline-block; background-color: #FCE4EC; padding: 4px 10px; border-radius: 8px;">👎</span>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<span style="display: inline-block; background-color: #f5f5f5; padding: 4px 10px; border-radius: 8px; opacity: 0.4;">👎</span>',
                                unsafe_allow_html=True
                            )
                else:
                    # 피드백 대기
                    col1, col2, col3 = st.columns([1, 1, 6])
                    with col1:
                        if st.button("👍", key=f"helpful_{feedback_key}", help="도움이 됐어요"):
                            save_qa_feedback(product_name, question, message["content"], 1)
                            st.session_state.feedback_given[feedback_key] = 1
                            st.toast("✅ 피드백 감사합니다!", icon="👍")
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"not_helpful_{feedback_key}", help="도움이 안됐어요"):
                            save_qa_feedback(product_name, question, message["content"], -1)
                            st.session_state.feedback_given[feedback_key] = -1
                            st.toast("✅ 피드백 감사합니다!", icon="👎")
                            st.rerun()

    # 사용자 입력 (메시지 수 기반 key로 입력창 리셋)
    if prompt := st.chat_input("이 제품에 대해 질문하세요...", key=f"qa_input_{product_name}_{len(messages)}"):
        # 사용자 메시지 추가
        messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성 (캐시 + 스트리밍)
        with st.chat_message("assistant"):
            try:
                import time
                start_time = time.time()

                # 시맨틱 캐시 조회
                cache = get_semantic_cache()
                cache_result = cache.lookup(prompt, product_name)

                if cache_result.hit:
                    # 캐시 히트! 즉시 답변 표시
                    answer = cache_result.answer
                    sources = cache_result.sources
                    st.markdown(answer)

                    # 캐시 히트 표시
                    st.caption(f"⚡ 캐시에서 응답 (유사도 {cache_result.similarity:.1%})")

                    response_time_ms = int((time.time() - start_time) * 1000)
                else:
                    # 캐시 미스 - RAG 호출
                    rag_chain = st.session_state.product_rag_chain

                    # 스트리밍 + 출처 가져오기
                    stream, sources = rag_chain.stream_with_sources(prompt)

                    # 스트리밍 응답 표시
                    answer = st.write_stream(stream)

                    # 응답 시간 계산 (ms)
                    response_time_ms = int((time.time() - start_time) * 1000)

                    # 캐시에 저장
                    cache.store(prompt, answer, sources, product_name)

                # Q&A 로그 저장
                save_qa_log(product_name, prompt, response_time_ms)

                # 출처 표시 (개선된 버전) - 빈 결과도 표시
                render_qa_sources(sources)

                # 메시지 저장 (출처 포함)
                messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,  # 출처도 저장
                    "from_cache": cache_result.hit,  # 캐시 히트 여부
                })

                # 입력창 초기화를 위해 rerun
                st.rerun()

            except Exception as e:
                show_error(e, "답변 생성")


def render_product_reviews(product: Product):
    """리뷰 목록 - DB 조회 + 페이지네이션."""
    # DB에서 리뷰 조회
    db_reviews = get_reviews_by_product(product.name)

    # dict 형태로 변환
    # db_reviews는 이미 dict 리스트
    all_reviews = db_reviews

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
    if sentiment_filter != "전체":
        all_reviews = [r for r in all_reviews if r["sentiment"] == sentiment_filter]

    # 정렬
    sentiment_order = {"긍정": 1, "중립": 0, "부정": -1}
    if sort_option == "긍정순":
        all_reviews.sort(key=lambda r: sentiment_order.get(r["sentiment"], 0), reverse=True)
    elif sort_option == "부정순":
        all_reviews.sort(key=lambda r: sentiment_order.get(r["sentiment"], 0))

    total_count = len(all_reviews)
    st.markdown(f"**총 {total_count}개** 리뷰")

    if not all_reviews:
        st.info("표시할 리뷰가 없어요")
        return

    # 페이지네이션
    reviews_per_page = 5
    total_pages = (total_count + reviews_per_page - 1) // reviews_per_page
    page_key = f"review_page_{product.name}"

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    current_page = st.session_state[page_key]
    start_idx = current_page * reviews_per_page
    end_idx = min(start_idx + reviews_per_page, total_count)
    page_reviews = all_reviews[start_idx:end_idx]

    # 새 리뷰 확인 (자동 스크롤/열기용)
    new_review_product = st.session_state.get("new_review_added")
    new_review_id = st.session_state.get("new_review_id")
    is_new_review_page = (new_review_product == product.name)

    # 새 리뷰가 추가되면 첫 페이지로 이동 (최신순 정렬이므로)
    if is_new_review_page and current_page != 0:
        st.session_state[page_key] = 0
        st.rerun()

    # 리뷰 표시 (클릭하면 속성 분석 표시)
    polarity_map = {"1": "긍정", "0": "중립", "-1": "부정", 1: "긍정", 0: "중립", -1: "부정"}

    for i, review in enumerate(page_reviews):
        # 감정별 글씨 색상 (하늘=긍정, 빨강=부정, 초록=중립)
        sentiment_color = {"긍정": "#1976D2", "중립": "#388E3C", "부정": "#D32F2F"}.get(review["sentiment"], "#666")

        # 별점 표시 - "⭐ 5" 형태로 통일
        rating = review.get("rating", 0)
        stars = f"⭐ {rating} " if rating and rating > 0 else ""

        # 신뢰도 검사
        trust_result = check_review_text(review["text"], rating)
        trust_label = ' <span style="color: #F57C00; font-weight: bold;">[의심]</span>' if trust_result.is_suspicious else ""

        # 미리보기 텍스트
        preview = review["text"][:50] + "..." if len(review["text"]) > 50 else review["text"]

        # 의심 리뷰 상세 내용
        warning_html = ""
        if trust_result.is_suspicious:
            reason_map = {
                "excessive_praise": "과도한 칭찬",
                "spam_keywords": "스팸/광고",
                "too_short": "너무 짧음",
                "repetitive_pattern": "반복 문구",
                "no_specifics": "구체성 부족",
                "extreme_rating": "평점-내용 불일치",
            }
            reasons = [reason_map.get(r.value, r.value) for r in trust_result.reasons]
            warning_html = f'<div style="background-color: #FFF3E0; padding: 8px 12px; border-radius: 5px; margin-bottom: 10px; color: #E65100;">⚠️ 의심 사유: {", ".join(reasons)}</div>'

        # 속성 분석 HTML
        aspects_html = ""
        aspects = review.get("aspects", [])
        if aspects:
            aspects_html = '<hr style="margin: 10px 0;"><div style="font-weight: bold; margin-bottom: 8px;">🏷️ 속성 분석</div>'
            for aspect in aspects:
                aspect_name = aspect.get("Aspect", aspect.get("category", ""))
                aspect_polarity = aspect.get("SentimentPolarity", aspect.get("sentiment", "0"))
                aspect_text = aspect.get("SentimentText", aspect.get("text", ""))

                # 감정 변환
                if isinstance(aspect_polarity, str) and aspect_polarity in ["positive", "negative", "neutral"]:
                    a_label = {"positive": "긍정", "negative": "부정", "neutral": "중립"}.get(aspect_polarity, "중립")
                else:
                    a_label = polarity_map.get(aspect_polarity, polarity_map.get(str(aspect_polarity), "중립"))

                a_emoji = {"긍정": "👍", "중립": "➖", "부정": "👎"}.get(a_label, "❓")
                bg = {"긍정": "#e3f2fd", "중립": "#f5f5f5", "부정": "#ffebee"}.get(a_label, "#f5f5f5")
                aspects_html += f'<div style="background-color: {bg}; padding: 8px 12px; border-radius: 5px; margin-bottom: 4px;">{a_emoji} <b>{aspect_name}</b>: {aspect_text}</div>'

        # 새 리뷰인지 확인
        is_this_new_review = is_new_review_page and review.get("id") == new_review_id
        open_attr = "open" if is_this_new_review else ""
        new_review_class = "new-review-highlight" if is_this_new_review else ""

        # HTML details/summary로 접기/펼치기 구현
        # 페이지+인덱스 기반 고유 ID로 페이지 변경 시 토글 상태 리셋
        detail_id = f"review_p{current_page}_i{i}"
        st.markdown(f'''
<details id="{detail_id}" class="{new_review_class}" {open_attr} style="margin-bottom: 8px; border: 1px solid #ddd; border-radius: 8px; padding: 0;">
<summary style="padding: 12px; cursor: pointer; color: {sentiment_color}; font-weight: 500; list-style: none;">
{stars}{preview}{trust_label}
</summary>
<div style="padding: 12px; border-top: 1px solid #eee;">
{warning_html}
<div style="font-style: italic; color: #333;">"{review["text"]}"</div>
{aspects_html}
</div>
</details>
''', unsafe_allow_html=True)

    # 새 리뷰로 자동 스크롤 (JavaScript)
    if is_new_review_page and new_review_id:
        import streamlit.components.v1 as components
        # class="new-review-highlight"로 새 리뷰 찾아서 스크롤
        components.html('''
<script>
(function tryScroll(attempts) {
    if (attempts <= 0) return;
    try {
        var doc = window.parent.document;
        var newReview = doc.querySelector('.new-review-highlight');
        if (newReview) {
            newReview.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            setTimeout(function() { tryScroll(attempts - 1); }, 200);
        }
    } catch(e) {
        setTimeout(function() { tryScroll(attempts - 1); }, 200);
    }
})(10);
</script>
''', height=0)
        # 플래그 초기화
        del st.session_state["new_review_added"]
        del st.session_state["new_review_id"]

    # 페이지네이션 UI (10개 버튼 그룹)
    if total_pages > 1:
        st.markdown("---")
        pages_per_group = 10
        current_group = current_page // pages_per_group
        group_start = current_group * pages_per_group
        group_end = min(group_start + pages_per_group, total_pages)

        num_page_buttons = group_end - group_start
        cols = st.columns([1] + [1] * num_page_buttons + [1])

        with cols[0]:
            if current_group > 0:
                if st.button("◀", key=f"prev_review_group_{product.name}"):
                    st.session_state[page_key] = group_start - 1
                    st.rerun()

        for i, page_num in enumerate(range(group_start, group_end)):
            with cols[i + 1]:
                is_current = page_num == current_page
                if is_current:
                    st.button(f"**{page_num + 1}**", key=f"review_page_btn_{product.name}_{page_num}", disabled=True)
                else:
                    if st.button(str(page_num + 1), key=f"review_page_btn_{product.name}_{page_num}"):
                        st.session_state[page_key] = page_num
                        st.rerun()

        with cols[-1]:
            if group_end < total_pages:
                if st.button("▶", key=f"next_review_group_{product.name}"):
                    st.session_state[page_key] = group_end
                    st.rerun()


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
        placeholder="이 제품에 대한 리뷰를 작성해주세요... (최소 10글자)\n예: 가격은 좀 비싸지만 품질이 정말 좋아요. 배송도 빨랐습니다.",
        height=150,
        max_chars=200,
        key=text_key,
    )

    # 글자 수 카운터 (우측 정렬)
    char_count = len(review_text)
    st.markdown(
        f'<p style="text-align: right; color: {"#ff4b4b" if char_count > 200 else "#888"}; margin-top: -10px; font-size: 0.85em;">{char_count}/200</p>',
        unsafe_allow_html=True
    )

    # 제출 상태 관리
    submit_key = f"submitting_{safe_name}"
    is_submitting = st.session_state.get(submit_key, False)

    # 버튼 텍스트 및 상태
    button_text = "⏳ 저장 중..." if is_submitting else "✍️ 리뷰 작성"

    if st.button(button_text, key=f"submit_review_{product.name}", use_container_width=True, disabled=is_submitting):
        if len(review_text.strip()) < 10:
            st.warning("리뷰는 최소 10자 이상 작성해주세요.")
        elif review_text.strip():
            # 제출 시작
            st.session_state[submit_key] = True
            st.rerun()
        else:
            st.warning("리뷰 내용을 입력해주세요.")

    # 제출 처리 (버튼 클릭 후 rerun 시 실행)
    if is_submitting and review_text.strip():
        try:
            # AspectExtractor로 분석
            extractor = create_aspect_extractor(use_cache=True)
            result = extractor.extract(review_text.strip())

            # 감정을 한글로 변환
            sentiment_map = {"positive": "긍정", "negative": "부정", "neutral": "중립"}
            sentiment_kr = sentiment_map.get(result.overall_sentiment.value, "중립")

            # 제품 ID 조회
            product_id = get_or_create_product(
                name=product.name,
                category=product.category,
                main_category=product.main_category
            )

            # DB에 저장
            new_review_id = add_review(
                product_id=product_id,
                text=review_text.strip(),
                sentiment=sentiment_kr,
                aspects=result.aspects,
                rating=current_rating
            )

            # DB에서 최신 평균 별점 조회하여 Product 객체 업데이트
            db_product = get_product_by_name(product.name)
            if db_product:
                product.avg_rating = db_product.avg_rating
                product.review_count = db_product.review_count

            # 상태 초기화
            st.session_state[submit_key] = False
            st.session_state[clear_flag_key] = True

            # 새 리뷰 추가 플래그 (자동 스크롤/열기용)
            st.session_state["new_review_added"] = product.name
            st.session_state["new_review_id"] = new_review_id

            st.success("✅ 리뷰가 저장되었습니다!")
            st.rerun()

        except Exception as e:
            st.session_state[submit_key] = False
            show_error(e, "리뷰 분석")

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


if __name__ == "__main__":
    main()
