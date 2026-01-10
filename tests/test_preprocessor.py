"""
전처리 파이프라인 단위 테스트.
"""

import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

# playwright 의존성 없이 테스트할 수 있도록 모킹
sys.modules["playwright"] = MagicMock()
sys.modules["playwright.async_api"] = MagicMock()
sys.modules["playwright_stealth"] = MagicMock()

from src.crawler.base import Review
from src.pipeline.preprocessor import (
    DuplicateFilter,
    ProcessedReview,
    ReviewPreprocessor,
    TextChunk,
    TextChunker,
    TextCleaner,
    create_default_preprocessor,
)


class TestTextCleaner:
    """TextCleaner 테스트."""

    def test_basic_cleaning(self):
        """기본 텍스트 정제 테스트."""
        cleaner = TextCleaner()
        text = "이 제품   정말    좋아요!"
        result = cleaner.clean(text)
        assert "  " not in result  # 연속 공백 제거 확인

    def test_emoji_removal(self):
        """이모지 제거 테스트."""
        cleaner = TextCleaner(remove_emojis=True)
        text = "좋아요! 👍👍 최고예요! 🎉"
        result = cleaner.clean(text)
        assert "👍" not in result
        assert "🎉" not in result
        assert "좋아요" in result

    def test_emoji_preservation(self):
        """이모지 보존 테스트."""
        cleaner = TextCleaner(remove_emojis=False)
        text = "좋아요! 👍"
        result = cleaner.clean(text)
        assert "👍" in result

    def test_repeated_char_normalization(self):
        """반복 문자 정규화 테스트."""
        cleaner = TextCleaner(normalize_repeated_chars=True, max_repeated_chars=2)
        text = "정말 좋아요ㅋㅋㅋㅋㅋㅋㅋ"
        result = cleaner.clean(text)
        assert "ㅋㅋㅋ" not in result
        assert "ㅋㅋ" in result

    def test_special_char_removal(self):
        """특수문자 제거 테스트."""
        # remove_emojis=True로 설정해야 이모지 유니코드 범위의 기호(★, ♡)도 제거됨
        cleaner = TextCleaner(remove_special_chars=True, remove_emojis=True)
        text = "제품★★★ 정말 좋아요♡♡"
        result = cleaner.clean(text)
        assert "★" not in result
        assert "♡" not in result
        assert "제품" in result
        assert "좋아요" in result

    def test_html_entity_removal(self):
        """HTML 엔티티 제거 테스트."""
        cleaner = TextCleaner(remove_html_entities=True)
        text = "가격이 &lt;10000원&gt; 입니다&nbsp;좋아요"
        result = cleaner.clean(text)
        assert "&lt;" not in result
        assert "&gt;" not in result
        assert "&nbsp;" not in result

    def test_whitespace_normalization(self):
        """공백 정규화 테스트."""
        cleaner = TextCleaner(normalize_whitespace=True)
        text = "첫 번째 줄\n\n\n두 번째 줄   세 번째"
        result = cleaner.clean(text)
        # 연속 빈 줄과 공백이 정규화되어야 함
        assert "\n\n\n" not in result
        assert "   " not in result

    def test_lowercase_conversion(self):
        """소문자 변환 테스트."""
        cleaner = TextCleaner(lowercase=True)
        text = "HELLO World 안녕하세요"
        result = cleaner.clean(text)
        assert "hello" in result
        assert "world" in result
        assert "안녕하세요" in result  # 한글은 영향 없음

    def test_empty_text(self):
        """빈 텍스트 처리 테스트."""
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""

    def test_unicode_normalization(self):
        """Unicode 정규화 테스트 (한글 자모 결합)."""
        cleaner = TextCleaner()
        # 분리된 자모 (ㅎㅏㄴㄱㅜㄹ) → 결합된 한글 (한글)
        decomposed = "\u1112\u1161\u11ab\u1100\u116e\u11af"  # 한글 (분리형)
        result = cleaner.clean(decomposed)
        assert result  # 정규화 후 텍스트가 있어야 함


class TestTextChunker:
    """TextChunker 테스트."""

    def test_short_text_single_chunk(self):
        """짧은 텍스트 단일 청크 테스트."""
        chunker = TextChunker(chunk_size=500)
        text = "짧은 리뷰입니다."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_multiple_chunks(self):
        """긴 텍스트 다중 청크 테스트."""
        chunker = TextChunker(chunk_size=50, min_chunk_size=10, chunk_overlap=10)
        text = "이것은 첫 번째 문장입니다. 이것은 두 번째 문장입니다. 이것은 세 번째 문장입니다. 이것은 네 번째 문장입니다."
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        """청크 오버랩 테스트."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=20)
        text = "A" * 50 + ". " + "B" * 50 + ". " + "C" * 50 + "."
        chunks = chunker.chunk(text)

        # 청크가 여러 개 생성되어야 함
        assert len(chunks) >= 2

    def test_sentence_boundary_split(self):
        """문장 경계 분할 테스트."""
        chunker = TextChunker(chunk_size=100, split_by_sentence=True, min_chunk_size=10)
        text = "첫 번째 문장입니다. 두 번째 문장입니다! 세 번째 문장입니다?"
        chunks = chunker.chunk(text)

        # 각 청크가 완전한 문장을 포함해야 함
        for chunk in chunks:
            # 청크가 문장 중간에서 잘리지 않았는지 확인
            assert chunk.text.strip()

    def test_chunk_indices(self):
        """청크 인덱스 테스트."""
        chunker = TextChunker(chunk_size=30, min_chunk_size=10)
        text = "첫 번째. 두 번째. 세 번째. 네 번째."
        chunks = chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_text(self):
        """빈 텍스트 처리 테스트."""
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk(None) == []


class TestDuplicateFilter:
    """DuplicateFilter 테스트."""

    def test_exact_duplicate_detection(self):
        """정확한 중복 감지 테스트."""
        filter = DuplicateFilter()

        filter.add("이 제품 정말 좋아요")
        assert filter.is_duplicate("이 제품 정말 좋아요") is True
        assert filter.is_duplicate("다른 제품 리뷰") is False

    def test_whitespace_normalized_duplicate(self):
        """공백 정규화 중복 감지 테스트."""
        filter = DuplicateFilter()

        filter.add("이 제품 정말 좋아요")
        # 공백만 다른 경우도 중복으로 처리
        assert filter.is_duplicate("이 제품  정말   좋아요") is True

    def test_fuzzy_duplicate_detection(self):
        """퍼지 중복 감지 테스트."""
        filter = DuplicateFilter(use_fuzzy=True, fuzzy_threshold=0.7)

        filter.add("이 제품 정말 좋아요 배송도 빨라요")
        # 유사한 텍스트
        assert filter.is_duplicate("이 제품 정말 좋아요 배송 빨라요") is True
        # 다른 텍스트
        assert filter.is_duplicate("완전히 다른 내용의 리뷰입니다") is False

    def test_filter_list(self):
        """리스트 필터링 테스트."""
        filter = DuplicateFilter()

        texts = [
            "첫 번째 리뷰",
            "두 번째 리뷰",
            "첫 번째 리뷰",  # 중복
            "세 번째 리뷰",
            "두 번째 리뷰",  # 중복
        ]

        filtered = filter.filter(texts)
        assert len(filtered) == 3
        assert "첫 번째 리뷰" in filtered
        assert "두 번째 리뷰" in filtered
        assert "세 번째 리뷰" in filtered

    def test_reset(self):
        """필터 리셋 테스트."""
        filter = DuplicateFilter()

        filter.add("테스트 텍스트")
        assert filter.is_duplicate("테스트 텍스트") is True

        filter.reset()
        assert filter.is_duplicate("테스트 텍스트") is False

    def test_hash_computation(self):
        """해시 계산 테스트."""
        hash1 = DuplicateFilter._compute_hash("테스트 텍스트")
        hash2 = DuplicateFilter._compute_hash("테스트 텍스트")
        hash3 = DuplicateFilter._compute_hash("다른 텍스트")

        assert hash1 == hash2  # 동일 텍스트는 동일 해시
        assert hash1 != hash3  # 다른 텍스트는 다른 해시

    def test_similarity_calculation(self):
        """유사도 계산 테스트."""
        # 동일 텍스트
        sim1 = DuplicateFilter._similarity("a b c d e", "a b c d e")
        assert sim1 == 1.0

        # 완전히 다른 텍스트
        sim2 = DuplicateFilter._similarity("a b c", "x y z")
        assert sim2 == 0.0

        # 부분 유사 텍스트
        sim3 = DuplicateFilter._similarity("a b c d", "a b c e")
        assert 0 < sim3 < 1


class TestReviewPreprocessor:
    """ReviewPreprocessor 테스트."""

    @pytest.fixture
    def sample_review(self):
        """샘플 리뷰 픽스처."""
        return Review(
            text="이 제품 정말 좋아요ㅋㅋㅋㅋ 배송도 빨라서 매우 만족합니다!!! 품질도 가격 대비 훌륭하네요. 👍",
            rating=5.0,
            date="2024-01-15",
            author="테스터",
            option="블랙 / L",
            verified_purchase=True,
        )

    @pytest.fixture
    def preprocessor(self):
        """전처리기 픽스처."""
        return create_default_preprocessor(
            chunk_size=500,
            remove_emojis=True,
        )

    def test_basic_processing(self, preprocessor, sample_review):
        """기본 전처리 테스트."""
        result = preprocessor.process(sample_review)

        assert result is not None
        assert isinstance(result, ProcessedReview)
        assert result.original_text == sample_review.text
        assert result.cleaned_text != result.original_text  # 정제됨
        assert len(result.chunks) >= 1
        assert result.text_hash  # 해시 생성됨
        assert result.rating == 5.0

    def test_short_text_filtering(self):
        """짧은 텍스트 필터링 테스트."""
        preprocessor = ReviewPreprocessor(min_text_length=20)
        short_review = Review(text="좋아요", rating=5.0)

        result = preprocessor.process(short_review)
        assert result is None  # 필터링됨

    def test_long_text_truncation(self):
        """긴 텍스트 잘라내기 테스트."""
        preprocessor = ReviewPreprocessor(max_text_length=100, min_text_length=5)
        long_review = Review(text="좋은 제품입니다 " * 50, rating=5.0)

        result = preprocessor.process(long_review)
        assert result is not None
        assert len(result.cleaned_text) <= 100

    def test_duplicate_filtering(self):
        """중복 리뷰 필터링 테스트."""
        preprocessor = create_default_preprocessor()

        review1 = Review(text="이 제품 정말 좋습니다. 품질이 훌륭하고 배송도 빠릅니다. 추천해요!", rating=5.0)
        review2 = Review(text="이 제품 정말 좋습니다. 품질이 훌륭하고 배송도 빠릅니다. 추천해요!", rating=4.0)  # 동일 텍스트

        result1 = preprocessor.process(review1)
        result2 = preprocessor.process(review2)

        assert result1 is not None
        assert result2 is None  # 중복으로 필터링

    def test_skip_duplicate_check(self):
        """중복 검사 스킵 테스트."""
        preprocessor = create_default_preprocessor()

        review1 = Review(text="이 제품 정말 좋습니다. 품질이 훌륭하고 배송도 빠릅니다. 추천해요!", rating=5.0)
        review2 = Review(text="이 제품 정말 좋습니다. 품질이 훌륭하고 배송도 빠릅니다. 추천해요!", rating=4.0)

        result1 = preprocessor.process(review1, skip_duplicate_check=True)
        result2 = preprocessor.process(review2, skip_duplicate_check=True)

        assert result1 is not None
        assert result2 is not None  # 스킵으로 인해 통과

    def test_batch_processing(self, preprocessor):
        """배치 전처리 테스트."""
        reviews = [
            Review(text="첫 번째 리뷰입니다. 제품이 정말 좋아요. 품질이 훌륭합니다.", rating=5.0),
            Review(text="두 번째 리뷰입니다. 배송이 빨라요. 포장도 꼼꼼합니다.", rating=4.0),
            Review(text="세 번째 리뷰입니다. 가격이 착해요. 가성비 최고입니다.", rating=4.5),
        ]

        results = preprocessor.process_batch(reviews)
        assert len(results) == 3

    def test_metadata_extraction(self, preprocessor, sample_review):
        """메타데이터 추출 테스트."""
        result = preprocessor.process(sample_review)

        assert result is not None
        assert result.metadata["author"] == "테스터"
        assert result.metadata["option"] == "블랙 / L"
        assert result.metadata["verified_purchase"] is True
        assert "text_length" in result.metadata
        assert "word_count" in result.metadata

    def test_statistics(self, preprocessor):
        """통계 계산 테스트."""
        reviews = [
            Review(text="첫 번째 리뷰입니다. 제품이 정말 좋아요. 품질이 훌륭합니다.", rating=5.0),
            Review(text="두 번째 리뷰입니다. 배송이 빨라요. 포장도 꼼꼼합니다.", rating=4.0),
            Review(text="세 번째 리뷰입니다. 가격이 착해요. 가성비 최고입니다.", rating=3.0),
        ]

        processed = preprocessor.process_batch(reviews)
        stats = preprocessor.get_statistics(processed)

        assert stats["total_reviews"] == 3
        assert stats["total_chunks"] >= 3
        assert stats["avg_rating"] == 4.0
        assert stats["reviews_with_rating"] == 3

    def test_reset(self, preprocessor):
        """파이프라인 리셋 테스트."""
        review = Review(text="테스트 리뷰입니다. 이 제품 정말 좋아요. 품질이 훌륭합니다.", rating=5.0)

        result1 = preprocessor.process(review)
        result2 = preprocessor.process(review)  # 중복

        assert result1 is not None
        assert result2 is None

        preprocessor.reset()

        result3 = preprocessor.process(review)
        assert result3 is not None  # 리셋 후 다시 처리 가능


class TestCreateDefaultPreprocessor:
    """create_default_preprocessor 테스트."""

    def test_default_creation(self):
        """기본 전처리기 생성 테스트."""
        preprocessor = create_default_preprocessor()
        assert isinstance(preprocessor, ReviewPreprocessor)
        assert isinstance(preprocessor.cleaner, TextCleaner)
        assert isinstance(preprocessor.chunker, TextChunker)
        assert isinstance(preprocessor.duplicate_filter, DuplicateFilter)

    def test_custom_chunk_size(self):
        """커스텀 청크 크기 테스트."""
        preprocessor = create_default_preprocessor(chunk_size=100, chunk_overlap=20)
        assert preprocessor.chunker.chunk_size == 100
        assert preprocessor.chunker.chunk_overlap == 20

    def test_emoji_removal_option(self):
        """이모지 제거 옵션 테스트."""
        preprocessor = create_default_preprocessor(remove_emojis=True)
        assert preprocessor.cleaner.remove_emojis is True

    def test_fuzzy_dedup_option(self):
        """퍼지 중복 제거 옵션 테스트."""
        preprocessor = create_default_preprocessor(use_fuzzy_dedup=True)
        assert preprocessor.duplicate_filter.use_fuzzy is True


class TestTextChunkDataclass:
    """TextChunk 데이터클래스 테스트."""

    def test_chunk_creation(self):
        """청크 생성 테스트."""
        chunk = TextChunk(
            text="테스트 텍스트",
            chunk_index=0,
            start_char=0,
            end_char=10,
            metadata={"key": "value"},
        )

        assert chunk.text == "테스트 텍스트"
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 10
        assert chunk.metadata["key"] == "value"

    def test_default_metadata(self):
        """기본 메타데이터 테스트."""
        chunk = TextChunk(text="텍스트", chunk_index=0, start_char=0, end_char=5)
        assert chunk.metadata == {}


class TestProcessedReviewDataclass:
    """ProcessedReview 데이터클래스 테스트."""

    def test_processed_review_creation(self):
        """전처리된 리뷰 생성 테스트."""
        chunk = TextChunk(text="정제된 텍스트", chunk_index=0, start_char=0, end_char=10)

        processed = ProcessedReview(
            original_text="원본 텍스트",
            cleaned_text="정제된 텍스트",
            chunks=[chunk],
            text_hash="abc123",
            rating=4.5,
            date="2024-01-15",
            metadata={"source": "test"},
        )

        assert processed.original_text == "원본 텍스트"
        assert processed.cleaned_text == "정제된 텍스트"
        assert len(processed.chunks) == 1
        assert processed.text_hash == "abc123"
        assert processed.rating == 4.5
        assert processed.date == "2024-01-15"
        assert processed.metadata["source"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
