"""
벡터 데이터베이스 임베딩 파이프라인 테스트.
"""

import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# playwright 의존성 없이 테스트할 수 있도록 모킹
sys.modules["playwright"] = MagicMock()
sys.modules["playwright.async_api"] = MagicMock()
sys.modules["playwright_stealth"] = MagicMock()

from src.crawler.base import Review
from src.pipeline.embedder import (
    CollectionManager,
    EmbedderConfig,
    ReviewEmbedder,
    SearchResult,
    create_embedder,
)
from src.pipeline.preprocessor import (
    ProcessedReview,
    TextChunk,
    create_default_preprocessor,
)


class TestEmbedderConfig:
    """EmbedderConfig 테스트."""

    def test_default_config(self):
        """기본 설정 테스트."""
        config = EmbedderConfig()
        assert config.persist_directory == "./data/chroma_db"
        assert config.collection_name == "reviews"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.default_top_k == 5
        assert config.batch_size == 100

    def test_custom_config(self):
        """커스텀 설정 테스트."""
        config = EmbedderConfig(
            persist_directory="/custom/path",
            collection_name="custom_reviews",
            default_top_k=10,
        )
        assert config.persist_directory == "/custom/path"
        assert config.collection_name == "custom_reviews"
        assert config.default_top_k == 10


class TestSearchResult:
    """SearchResult 데이터클래스 테스트."""

    def test_search_result_creation(self):
        """검색 결과 생성 테스트."""
        result = SearchResult(
            text="테스트 텍스트",
            score=0.95,
            metadata={"rating": 5.0},
            chunk_id="abc123",
        )
        assert result.text == "테스트 텍스트"
        assert result.score == 0.95
        assert result.metadata["rating"] == 5.0
        assert result.chunk_id == "abc123"

    def test_default_metadata(self):
        """기본 메타데이터 테스트."""
        result = SearchResult(text="텍스트", score=0.5)
        assert result.metadata == {}
        assert result.chunk_id is None


class TestCollectionManager:
    """CollectionManager 테스트."""

    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 픽스처."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_dir):
        """CollectionManager 픽스처."""
        return CollectionManager(persist_directory=temp_dir)

    def test_list_collections_empty(self, manager):
        """빈 컬렉션 리스트 테스트."""
        collections = manager.list_collections()
        assert isinstance(collections, list)

    def test_create_collection(self, manager):
        """컬렉션 생성 테스트."""
        result = manager.create_collection("test_collection")
        assert result is True

        info = manager.get_collection_info("test_collection")
        assert info is not None
        assert info["name"] == "test_collection"
        assert info["count"] == 0

    def test_collection_exists(self, manager):
        """컬렉션 존재 확인 테스트."""
        assert manager.collection_exists("nonexistent") is False

        manager.create_collection("existing")
        assert manager.collection_exists("existing") is True

    def test_delete_collection(self, manager):
        """컬렉션 삭제 테스트."""
        manager.create_collection("to_delete")
        assert manager.collection_exists("to_delete") is True

        result = manager.delete_collection("to_delete")
        assert result is True
        assert manager.collection_exists("to_delete") is False

    def test_delete_nonexistent_collection(self, manager):
        """존재하지 않는 컬렉션 삭제 테스트."""
        result = manager.delete_collection("nonexistent")
        assert result is False


class TestReviewEmbedderMocked:
    """ReviewEmbedder 모킹 테스트 (API 호출 없이)."""

    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 픽스처."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_no_api_key_error(self, temp_dir):
        """API 키 없을 때 에러 테스트."""
        # 환경변수 제거
        with patch.dict(os.environ, {}, clear=True):
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

            with pytest.raises(ValueError) as exc_info:
                config = EmbedderConfig(persist_directory=temp_dir)
                ReviewEmbedder(config=config, openai_api_key=None)

            assert "OpenAI API 키가 필요합니다" in str(exc_info.value)

    def test_config_initialization(self, temp_dir):
        """설정 초기화 테스트."""
        config = EmbedderConfig(
            persist_directory=temp_dir,
            collection_name="test_collection",
        )
        assert config.persist_directory == temp_dir
        assert config.collection_name == "test_collection"

    def test_build_filter_empty(self, temp_dir):
        """빈 필터 빌드 테스트."""
        with patch.object(ReviewEmbedder, "__init__", lambda x, **kwargs: None):
            embedder = ReviewEmbedder.__new__(ReviewEmbedder)
            result = embedder._build_filter()
            assert result is None

    def test_build_filter_rating_min(self, temp_dir):
        """최소 평점 필터 테스트."""
        with patch.object(ReviewEmbedder, "__init__", lambda x, **kwargs: None):
            embedder = ReviewEmbedder.__new__(ReviewEmbedder)
            result = embedder._build_filter(filter_rating_min=4.0)
            assert result == {"rating": {"$gte": 4.0}}

    def test_build_filter_rating_max(self, temp_dir):
        """최대 평점 필터 테스트."""
        with patch.object(ReviewEmbedder, "__init__", lambda x, **kwargs: None):
            embedder = ReviewEmbedder.__new__(ReviewEmbedder)
            result = embedder._build_filter(filter_rating_max=3.0)
            assert result == {"rating": {"$lte": 3.0}}

    def test_build_filter_combined(self, temp_dir):
        """복합 필터 테스트."""
        with patch.object(ReviewEmbedder, "__init__", lambda x, **kwargs: None):
            embedder = ReviewEmbedder.__new__(ReviewEmbedder)
            result = embedder._build_filter(
                filter_rating_min=3.0,
                filter_rating_max=5.0,
                filter_metadata={"verified": True},
            )
            assert "$and" in result
            assert len(result["$and"]) == 3


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestReviewEmbedderIntegration:
    """ReviewEmbedder 통합 테스트 (실제 API 사용)."""

    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 픽스처."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def embedder(self, temp_dir):
        """ReviewEmbedder 픽스처."""
        config = EmbedderConfig(
            persist_directory=temp_dir,
            collection_name="test_reviews",
        )
        emb = ReviewEmbedder(config=config)
        yield emb
        emb.delete_collection()

    @pytest.fixture
    def sample_processed_reviews(self):
        """샘플 전처리된 리뷰 픽스처."""
        preprocessor = create_default_preprocessor(chunk_size=200)
        reviews = [
            Review(
                text="이 제품 정말 좋아요! 배송도 빠르고 품질도 훌륭합니다.",
                rating=5.0,
                date="2024-01-15",
            ),
            Review(
                text="배송은 빨랐는데 품질이 기대에 못 미치네요.",
                rating=2.0,
                date="2024-01-14",
            ),
            Review(
                text="무난한 제품입니다. 가격 대비 괜찮아요.",
                rating=3.5,
                date="2024-01-13",
            ),
        ]
        return preprocessor.process_batch(reviews)

    def test_add_and_search_reviews(self, embedder, sample_processed_reviews):
        """리뷰 추가 및 검색 통합 테스트."""
        # 리뷰 추가
        added = embedder.add_reviews(sample_processed_reviews)
        assert added > 0

        # 통계 확인
        stats = embedder.get_collection_stats()
        assert stats["total_chunks"] == added

        # 검색 테스트
        results = embedder.search("배송이 빠른가요?", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_with_rating_filter(self, embedder, sample_processed_reviews):
        """평점 필터 검색 테스트."""
        embedder.add_reviews(sample_processed_reviews)

        # 4점 이상만 검색
        results = embedder.search("좋은 제품", top_k=5, filter_rating_min=4.0)

        # 결과가 있으면 모두 4점 이상이어야 함
        for result in results:
            if "rating" in result.metadata:
                assert result.metadata["rating"] >= 4.0

    def test_reset_collection(self, embedder, sample_processed_reviews):
        """컬렉션 리셋 테스트."""
        embedder.add_reviews(sample_processed_reviews)
        initial_stats = embedder.get_collection_stats()
        assert initial_stats["total_chunks"] > 0

        embedder.reset_collection()

        final_stats = embedder.get_collection_stats()
        assert final_stats["total_chunks"] == 0

    def test_get_retriever(self, embedder, sample_processed_reviews):
        """Retriever 생성 테스트."""
        embedder.add_reviews(sample_processed_reviews)

        retriever = embedder.get_retriever(top_k=3)
        assert retriever is not None

        # Retriever로 검색
        docs = retriever.invoke("배송 품질")
        assert len(docs) > 0


class TestCreateEmbedder:
    """create_embedder 헬퍼 함수 테스트."""

    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 픽스처."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_create_embedder_with_api_key(self, temp_dir):
        """API 키로 임베더 생성 테스트."""
        embedder = create_embedder(
            collection_name="test",
            persist_directory=temp_dir,
        )
        assert isinstance(embedder, ReviewEmbedder)
        embedder.delete_collection()

    def test_create_embedder_without_api_key(self, temp_dir):
        """API 키 없이 임베더 생성 시 에러 테스트."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                create_embedder(
                    collection_name="test",
                    persist_directory=temp_dir,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
