"""
속성 추출기 (Aspect Extractor) 테스트.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.aspect_extractor import (
    AspectCategory,
    AspectExtractionResult,
    AspectExtractor,
    AspectResult,
    ExtractedAspect,
    Sentiment,
    create_aspect_extractor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_aspect_result():
    """샘플 AspectResult 객체."""
    return AspectResult(
        review_text="가격은 좀 비싸지만 소재가 정말 좋아요.",
        overall_sentiment=Sentiment.POSITIVE,
        confidence=0.85,
        aspects=[
            {
                "category": "가격/가성비",
                "sentiment": "negative",
                "text": "가격은 좀 비싸지만",
                "keywords": ["가격", "비싸다"],
            },
            {
                "category": "소재/품질",
                "sentiment": "positive",
                "text": "소재가 정말 좋아요",
                "keywords": ["소재", "좋다"],
            },
        ],
    )


@pytest.fixture
def sample_extraction_result():
    """샘플 AspectExtractionResult 객체."""
    return AspectExtractionResult(
        aspects=[
            ExtractedAspect(
                category="가격/가성비",
                sentiment="negative",
                text="가격은 좀 비싸지만",
                keywords=["가격", "비싸다"],
            ),
            ExtractedAspect(
                category="소재/품질",
                sentiment="positive",
                text="소재가 정말 좋아요",
                keywords=["소재", "좋다"],
            ),
        ],
        overall_sentiment="positive",
        confidence=0.85,
    )


@pytest.fixture
def mock_llm():
    """모킹된 LLM."""
    mock = MagicMock()
    mock.invoke.return_value = AspectExtractionResult(
        aspects=[
            ExtractedAspect(
                category="배송/포장",
                sentiment="positive",
                text="배송이 빨랐습니다",
                keywords=["배송", "빠르다"],
            ),
        ],
        overall_sentiment="positive",
        confidence=0.9,
    )
    return mock


# =============================================================================
# Sentiment Enum Tests
# =============================================================================


class TestSentiment:
    """Sentiment Enum 테스트."""

    def test_sentiment_values(self):
        """감정 값 테스트."""
        assert Sentiment.POSITIVE.value == "positive"
        assert Sentiment.NEGATIVE.value == "negative"
        assert Sentiment.NEUTRAL.value == "neutral"

    def test_sentiment_from_string(self):
        """문자열에서 감정 변환."""
        assert Sentiment("positive") == Sentiment.POSITIVE
        assert Sentiment("negative") == Sentiment.NEGATIVE
        assert Sentiment("neutral") == Sentiment.NEUTRAL


# =============================================================================
# AspectCategory Enum Tests
# =============================================================================


class TestAspectCategory:
    """AspectCategory Enum 테스트."""

    def test_category_values(self):
        """카테고리 값 테스트."""
        assert AspectCategory.PRICE.value == "가격/가성비"
        assert AspectCategory.DESIGN.value == "디자인/외관"
        assert AspectCategory.SIZE.value == "사이즈/치수"
        assert AspectCategory.QUALITY.value == "소재/품질"
        assert AspectCategory.DELIVERY.value == "배송/포장"

    def test_all_categories_have_values(self):
        """모든 카테고리가 값을 가지고 있는지 확인."""
        for category in AspectCategory:
            assert category.value is not None
            assert len(category.value) > 0


# =============================================================================
# AspectResult Tests
# =============================================================================


class TestAspectResult:
    """AspectResult 테스트."""

    def test_from_extraction_result(self, sample_extraction_result):
        """AspectExtractionResult에서 생성 테스트."""
        result = AspectResult.from_extraction_result(
            review_text="테스트 리뷰",
            result=sample_extraction_result,
        )

        assert result.review_text == "테스트 리뷰"
        assert result.overall_sentiment == Sentiment.POSITIVE
        assert result.confidence == 0.85
        assert len(result.aspects) == 2

    def test_from_extraction_result_with_metadata(self, sample_extraction_result):
        """메타데이터 포함 생성 테스트."""
        metadata = {"source": "test", "id": 123}
        result = AspectResult.from_extraction_result(
            review_text="테스트 리뷰",
            result=sample_extraction_result,
            metadata=metadata,
        )

        assert result.metadata["source"] == "test"
        assert result.metadata["id"] == 123

    def test_to_dict(self, sample_aspect_result):
        """딕셔너리 변환 테스트."""
        data = sample_aspect_result.to_dict()

        assert data["review_text"] == "가격은 좀 비싸지만 소재가 정말 좋아요."
        assert data["overall_sentiment"] == "positive"
        assert data["confidence"] == 0.85
        assert len(data["aspects"]) == 2

    def test_get_aspect_by_category(self, sample_aspect_result):
        """카테고리별 속성 조회."""
        price_aspects = sample_aspect_result.get_aspect_by_category("가격/가성비")
        assert len(price_aspects) == 1
        assert price_aspects[0]["sentiment"] == "negative"

        quality_aspects = sample_aspect_result.get_aspect_by_category("소재/품질")
        assert len(quality_aspects) == 1
        assert quality_aspects[0]["sentiment"] == "positive"

        # 없는 카테고리
        other_aspects = sample_aspect_result.get_aspect_by_category("기타")
        assert len(other_aspects) == 0

    def test_get_positive_aspects(self, sample_aspect_result):
        """긍정 속성 조회."""
        positive = sample_aspect_result.get_positive_aspects()
        assert len(positive) == 1
        assert positive[0]["category"] == "소재/품질"

    def test_get_negative_aspects(self, sample_aspect_result):
        """부정 속성 조회."""
        negative = sample_aspect_result.get_negative_aspects()
        assert len(negative) == 1
        assert negative[0]["category"] == "가격/가성비"


# =============================================================================
# AspectExtractor Tests
# =============================================================================


class TestAspectExtractor:
    """AspectExtractor 테스트."""

    def test_init_without_api_key(self):
        """API 키 없이 초기화 시 에러."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API 키가 필요합니다"):
                AspectExtractor()

    def test_init_with_api_key(self):
        """API 키로 초기화."""
        with patch("src.pipeline.aspect_extractor.ChatOpenAI"):
            extractor = AspectExtractor(openai_api_key="test-key")
            assert extractor.model_name == "gpt-4o-mini"
            assert extractor.temperature == 0.0

    def test_init_with_custom_settings(self):
        """커스텀 설정으로 초기화."""
        with patch("src.pipeline.aspect_extractor.ChatOpenAI"):
            extractor = AspectExtractor(
                model_name="gpt-4",
                temperature=0.5,
                openai_api_key="test-key",
                use_cache=False,
            )
            assert extractor.model_name == "gpt-4"
            assert extractor.temperature == 0.5
            assert extractor.use_cache is False

    def test_cache_key_generation(self):
        """캐시 키 생성 테스트."""
        with patch("src.pipeline.aspect_extractor.ChatOpenAI"):
            extractor = AspectExtractor(openai_api_key="test-key")

            key1 = extractor._get_cache_key("테스트 텍스트")
            key2 = extractor._get_cache_key("테스트 텍스트")
            key3 = extractor._get_cache_key("다른 텍스트")

            assert key1 == key2  # 같은 텍스트는 같은 키
            assert key1 != key3  # 다른 텍스트는 다른 키

    def test_cache_save_and_load(self, sample_aspect_result):
        """캐시 저장 및 로드 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.pipeline.aspect_extractor.ChatOpenAI"):
                extractor = AspectExtractor(
                    openai_api_key="test-key",
                    cache_dir=tmpdir,
                    use_cache=True,
                )

                # 캐시 저장
                extractor._save_to_cache(sample_aspect_result)

                # 캐시 로드
                loaded = extractor._get_from_cache(sample_aspect_result.review_text)

                assert loaded is not None
                assert loaded.review_text == sample_aspect_result.review_text
                assert loaded.overall_sentiment == sample_aspect_result.overall_sentiment

    def test_cache_disabled(self, sample_aspect_result):
        """캐시 비활성화 테스트."""
        with patch("src.pipeline.aspect_extractor.ChatOpenAI"):
            extractor = AspectExtractor(
                openai_api_key="test-key",
                use_cache=False,
            )

            # 캐시 비활성화 시 저장해도 조회 불가
            extractor._save_to_cache(sample_aspect_result)
            loaded = extractor._get_from_cache(sample_aspect_result.review_text)

            assert loaded is None

    def test_extract_with_mock(self, mock_llm):
        """추출 테스트 (모킹)."""
        with patch("src.pipeline.aspect_extractor.ChatOpenAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            extractor = AspectExtractor(openai_api_key="test-key", use_cache=False)
            extractor._llm = mock_llm

            result = extractor.extract("배송이 빨랐습니다")

            assert result.overall_sentiment == Sentiment.POSITIVE
            assert len(result.aspects) == 1
            assert result.aspects[0]["category"] == "배송/포장"

    def test_extract_batch(self, mock_llm):
        """배치 추출 테스트."""
        with patch("src.pipeline.aspect_extractor.ChatOpenAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            extractor = AspectExtractor(openai_api_key="test-key", use_cache=False)
            extractor._llm = mock_llm

            reviews = ["리뷰 1", "리뷰 2", "리뷰 3"]
            results = extractor.extract_batch(reviews, show_progress=False)

            assert len(results) == 3
            assert mock_llm.invoke.call_count == 3

    def test_extract_batch_with_metadata(self, mock_llm):
        """메타데이터 포함 배치 추출 테스트."""
        with patch("src.pipeline.aspect_extractor.ChatOpenAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            extractor = AspectExtractor(openai_api_key="test-key", use_cache=False)
            extractor._llm = mock_llm

            reviews = [
                {"text": "리뷰 1", "metadata": {"id": 1}},
                {"text": "리뷰 2", "metadata": {"id": 2}},
            ]
            results = extractor.extract_batch(reviews, show_progress=False)

            assert len(results) == 2

    def test_get_aspect_statistics(self):
        """통계 계산 테스트."""
        results = [
            AspectResult(
                review_text="리뷰 1",
                overall_sentiment=Sentiment.POSITIVE,
                confidence=0.9,
                aspects=[
                    {"category": "배송/포장", "sentiment": "positive", "text": "", "keywords": []},
                    {"category": "가격/가성비", "sentiment": "negative", "text": "", "keywords": []},
                ],
            ),
            AspectResult(
                review_text="리뷰 2",
                overall_sentiment=Sentiment.NEGATIVE,
                confidence=0.8,
                aspects=[
                    {"category": "소재/품질", "sentiment": "negative", "text": "", "keywords": []},
                ],
            ),
            AspectResult(
                review_text="리뷰 3",
                overall_sentiment=Sentiment.NEUTRAL,
                confidence=0.7,
                aspects=[
                    {"category": "배송/포장", "sentiment": "positive", "text": "", "keywords": []},
                ],
            ),
        ]

        with patch("src.pipeline.aspect_extractor.ChatOpenAI"):
            extractor = AspectExtractor(openai_api_key="test-key")

        stats = extractor.get_aspect_statistics(results)

        assert stats["total_reviews"] == 3
        assert stats["overall_sentiment"]["positive"] == 1
        assert stats["overall_sentiment"]["negative"] == 1
        assert stats["overall_sentiment"]["neutral"] == 1
        assert stats["aspect_counts"]["배송/포장"] == 2
        assert stats["aspect_counts"]["가격/가성비"] == 1
        assert stats["aspect_counts"]["소재/품질"] == 1
        assert stats["avg_confidence"] == pytest.approx(0.8, rel=0.01)

    def test_clear_cache(self):
        """캐시 삭제 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.pipeline.aspect_extractor.ChatOpenAI"):
                extractor = AspectExtractor(
                    openai_api_key="test-key",
                    cache_dir=tmpdir,
                    use_cache=True,
                )

                # 캐시 파일 생성
                cache_file = Path(tmpdir) / "test.json"
                cache_file.write_text("{}")

                count = extractor.clear_cache()
                assert count == 1
                assert not cache_file.exists()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateAspectExtractor:
    """create_aspect_extractor 함수 테스트."""

    def test_create_with_defaults(self):
        """기본값으로 생성."""
        with patch("src.pipeline.aspect_extractor.ChatOpenAI"):
            extractor = create_aspect_extractor(openai_api_key="test-key")

            assert isinstance(extractor, AspectExtractor)
            assert extractor.model_name == "gpt-4o-mini"
            assert extractor.temperature == 0.0

    def test_create_with_custom_settings(self):
        """커스텀 설정으로 생성."""
        with patch("src.pipeline.aspect_extractor.ChatOpenAI"):
            extractor = create_aspect_extractor(
                model_name="gpt-4",
                temperature=0.7,
                openai_api_key="test-key",
                use_cache=False,
            )

            assert extractor.model_name == "gpt-4"
            assert extractor.temperature == 0.7
            assert extractor.use_cache is False


# =============================================================================
# ExtractedAspect Pydantic Model Tests
# =============================================================================


class TestExtractedAspect:
    """ExtractedAspect Pydantic 모델 테스트."""

    def test_create_extracted_aspect(self):
        """ExtractedAspect 생성 테스트."""
        aspect = ExtractedAspect(
            category="배송/포장",
            sentiment="positive",
            text="배송이 빨랐습니다",
            keywords=["배송", "빠르다"],
        )

        assert aspect.category == "배송/포장"
        assert aspect.sentiment == "positive"
        assert aspect.text == "배송이 빨랐습니다"
        assert aspect.keywords == ["배송", "빠르다"]

    def test_extracted_aspect_default_keywords(self):
        """키워드 기본값 테스트."""
        aspect = ExtractedAspect(
            category="가격/가성비",
            sentiment="negative",
            text="가격이 비쌉니다",
        )

        assert aspect.keywords == []


# =============================================================================
# AspectExtractionResult Pydantic Model Tests
# =============================================================================


class TestAspectExtractionResult:
    """AspectExtractionResult Pydantic 모델 테스트."""

    def test_create_extraction_result(self):
        """AspectExtractionResult 생성 테스트."""
        result = AspectExtractionResult(
            aspects=[
                ExtractedAspect(
                    category="배송/포장",
                    sentiment="positive",
                    text="배송이 빨랐습니다",
                ),
            ],
            overall_sentiment="positive",
            confidence=0.9,
        )

        assert len(result.aspects) == 1
        assert result.overall_sentiment == "positive"
        assert result.confidence == 0.9

    def test_confidence_bounds(self):
        """신뢰도 범위 테스트."""
        # 유효한 범위
        result = AspectExtractionResult(
            aspects=[],
            overall_sentiment="neutral",
            confidence=0.5,
        )
        assert result.confidence == 0.5

        # 경계값
        result = AspectExtractionResult(
            aspects=[],
            overall_sentiment="neutral",
            confidence=0.0,
        )
        assert result.confidence == 0.0

        result = AspectExtractionResult(
            aspects=[],
            overall_sentiment="neutral",
            confidence=1.0,
        )
        assert result.confidence == 1.0
