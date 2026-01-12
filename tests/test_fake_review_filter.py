"""
가짜 리뷰 필터 테스트.
"""

import pytest

from src.crawler.base import Review
from src.pipeline.fake_review_filter import (
    FakeReviewFilter,
    FakeReviewReason,
    FakeReviewResult,
    create_fake_review_filter,
)


class TestFakeReviewFilter:
    """FakeReviewFilter 테스트."""

    @pytest.fixture
    def filter(self):
        """LLM 없는 필터 인스턴스."""
        return FakeReviewFilter(use_llm=False)

    def test_normal_review(self, filter):
        """정상 리뷰 테스트."""
        review = Review(
            text="배송이 빠르고 품질도 좋아요. 색상이 사진과 같아서 만족합니다. 다만 사이즈가 조금 작네요.",
            rating=4.0,
        )
        result = filter.check_review(review)

        assert not result.is_suspicious
        assert result.weight >= 0.8

    def test_excessive_praise(self, filter):
        """과도한 칭찬 패턴 테스트."""
        review = Review(
            text="인생템이에요! 최고의 제품! 무조건 사세요! 두 번 사도 아깝지 않아요!",
            rating=5.0,
        )
        result = filter.check_review(review)

        assert result.is_suspicious
        assert FakeReviewReason.EXCESSIVE_PRAISE in result.reasons

    def test_spam_keywords(self, filter):
        """스팸 키워드 테스트."""
        review = Review(
            text="좋은 제품이에요. 문의 주시면 자세히 알려드릴게요. 카톡 문의 환영합니다.",
            rating=5.0,
        )
        result = filter.check_review(review)

        assert result.is_suspicious
        assert FakeReviewReason.SPAM_KEYWORDS in result.reasons
        assert result.weight < 0.5  # 스팸은 가중치 많이 감소

    def test_too_short(self, filter):
        """너무 짧은 리뷰 테스트."""
        review = Review(
            text="좋아요",
            rating=5.0,
        )
        result = filter.check_review(review)

        assert FakeReviewReason.TOO_SHORT in result.reasons

    def test_repetitive_pattern(self, filter):
        """반복 패턴 테스트."""
        review = Review(
            text="좋아요 좋아요 좋아요 정말 좋아요 너무 좋아요",
            rating=5.0,
        )
        result = filter.check_review(review)

        assert FakeReviewReason.REPETITIVE_PATTERN in result.reasons

    def test_rating_mismatch_high(self, filter):
        """평점-내용 불일치 (높은 평점, 부정적 내용) 테스트."""
        review = Review(
            text="별로예요. 실망했어요. 후회합니다.",
            rating=5.0,
        )
        result = filter.check_review(review)

        assert FakeReviewReason.EXTREME_RATING in result.reasons

    def test_rating_mismatch_low(self, filter):
        """평점-내용 불일치 (낮은 평점, 긍정적 내용) 테스트."""
        review = Review(
            text="너무 좋아요! 만족해요! 추천합니다!",
            rating=1.0,
        )
        result = filter.check_review(review)

        assert FakeReviewReason.EXTREME_RATING in result.reasons

    def test_no_specifics(self, filter):
        """구체성 부족 테스트."""
        review = Review(
            text="그냥 좋습니다. 마음에 들어요. 괜찮네요.",
            rating=4.0,
        )
        result = filter.check_review(review)

        assert FakeReviewReason.NO_SPECIFICS in result.reasons

    def test_filter_reviews_keep(self, filter):
        """리뷰 필터링 (유지) 테스트."""
        reviews = [
            Review(text="배송 빠르고 품질 좋아요. 사이즈 맞아요.", rating=4.0),
            Review(text="인생템! 최고의 제품! 무조건 사세요!", rating=5.0),  # 명확한 의심 리뷰
            Review(text="가격 대비 품질이 좋습니다. 배송도 빨랐어요.", rating=4.5),
        ]

        filtered, results = filter.filter_reviews(reviews, remove_suspicious=False)

        assert len(filtered) == 3  # 모두 유지
        assert any(r.is_suspicious for r in results)

    def test_filter_reviews_remove(self, filter):
        """리뷰 필터링 (제거) 테스트."""
        reviews = [
            Review(text="배송 빠르고 품질 좋아요. 사이즈 맞아요.", rating=4.0),
            Review(text="인생템! 최고! 무조건 사세요!", rating=5.0),
            Review(text="가격 대비 품질이 좋습니다. 배송도 빨랐어요.", rating=4.5),
        ]

        filtered, results = filter.filter_reviews(reviews, remove_suspicious=True)

        assert len(filtered) < 3  # 의심 리뷰 제거됨

    def test_get_statistics(self, filter):
        """통계 테스트."""
        reviews = [
            Review(text="배송 빠르고 품질 좋아요.", rating=4.0),
            Review(text="인생템! 최고의 제품! 무조건 사세요!", rating=5.0),  # 명확한 의심 리뷰
            Review(text="좋아요", rating=5.0),  # 너무 짧음 (confidence 0.4)
        ]

        _, results = filter.filter_reviews(reviews)
        stats = filter.get_statistics(results)

        assert stats["total"] == 3
        assert stats["suspicious"] >= 1
        assert "reason_counts" in stats

    def test_weight_calculation(self, filter):
        """가중치 계산 테스트."""
        # 정상 리뷰
        normal = Review(text="배송 빠르고 품질 좋아요. 색상 마음에 들어요.", rating=4.0)
        normal_result = filter.check_review(normal)

        # 의심 리뷰
        suspicious = Review(text="인생템! 최고의 제품! 무조건 사세요!", rating=5.0)
        suspicious_result = filter.check_review(suspicious)

        assert normal_result.weight > suspicious_result.weight


class TestCreateFakeReviewFilter:
    """create_fake_review_filter 헬퍼 테스트."""

    def test_create_default(self):
        """기본 생성 테스트."""
        filter = create_fake_review_filter()
        assert filter.use_llm is True
        assert filter.llm_threshold == 0.5

    def test_create_no_llm(self):
        """LLM 없이 생성 테스트."""
        filter = create_fake_review_filter(use_llm=False)
        assert filter.use_llm is False

    def test_create_custom_threshold(self):
        """커스텀 임계값 테스트."""
        filter = create_fake_review_filter(llm_threshold=0.7)
        assert filter.llm_threshold == 0.7
