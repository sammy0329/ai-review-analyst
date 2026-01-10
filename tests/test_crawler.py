"""
Unit tests for crawler module.
"""

import pytest

from src.crawler import (
    BaseCrawler,
    CoupangCrawler,
    AlwaysCrawler,
    CrawlResult,
    Review,
    get_crawler,
    USER_AGENTS,
)
from src.crawler.base import CrawlerError


class TestReviewDataClass:
    """Tests for Review dataclass."""

    def test_review_creation(self):
        """Test basic review creation."""
        review = Review(
            text="좋은 상품입니다!",
            rating=5,
            date="2024-01-10",
        )
        assert review.text == "좋은 상품입니다!"
        assert review.rating == 5
        assert review.date == "2024-01-10"

    def test_review_defaults(self):
        """Test review default values."""
        review = Review(text="테스트 리뷰")
        assert review.text == "테스트 리뷰"
        assert review.rating is None
        assert review.date is None
        assert review.author is None
        assert review.option is None
        assert review.helpful_count is None
        assert review.verified_purchase is False
        assert review.images == []
        assert review.metadata == {}


class TestCrawlResultDataClass:
    """Tests for CrawlResult dataclass."""

    def test_crawl_result_creation(self):
        """Test basic crawl result creation."""
        result = CrawlResult(
            url="https://www.coupang.com/products/123",
            product_name="테스트 상품",
            total_reviews=10,
        )
        assert result.url == "https://www.coupang.com/products/123"
        assert result.product_name == "테스트 상품"
        assert result.total_reviews == 10
        assert result.success is True

    def test_crawl_result_to_dict(self):
        """Test crawl result serialization."""
        review = Review(text="좋아요", rating=5)
        result = CrawlResult(
            url="https://example.com",
            reviews=[review],
        )
        data = result.to_dict()

        assert data["url"] == "https://example.com"
        assert len(data["reviews"]) == 1
        assert data["reviews"][0]["text"] == "좋아요"
        assert data["reviews"][0]["rating"] == 5

    def test_crawl_result_error(self):
        """Test crawl result with error."""
        result = CrawlResult(
            url="https://example.com",
            success=False,
            error_message="Connection failed",
        )
        assert result.success is False
        assert result.error_message == "Connection failed"


class TestCoupangCrawler:
    """Tests for CoupangCrawler."""

    def test_is_valid_url_valid(self):
        """Test valid Coupang URLs."""
        crawler = CoupangCrawler()

        valid_urls = [
            "https://www.coupang.com/products/123456",
            "https://www.coupang.com/products/123456?itemId=789",
            "https://www.coupang.com/products/7890123456",
        ]

        for url in valid_urls:
            assert crawler.is_valid_url(url) is True, f"Should be valid: {url}"

    def test_is_valid_url_invalid(self):
        """Test invalid Coupang URLs."""
        crawler = CoupangCrawler()

        invalid_urls = [
            "https://www.google.com",
            "https://www.amazon.com/products/123",
            "https://coupang.com/search?q=test",
            "https://www.coupang.com/categories/123",
            "",
            "not-a-url",
        ]

        for url in invalid_urls:
            assert crawler.is_valid_url(url) is False, f"Should be invalid: {url}"

    def test_extract_product_id(self):
        """Test product ID extraction."""
        crawler = CoupangCrawler()

        assert crawler._extract_product_id(
            "https://www.coupang.com/products/123456"
        ) == "123456"

        assert crawler._extract_product_id(
            "https://www.coupang.com/products/789?itemId=111"
        ) == "789"

        assert crawler._extract_product_id(
            "https://www.google.com"
        ) is None


class TestAlwaysCrawler:
    """Tests for AlwaysCrawler."""

    def test_is_valid_url_valid(self):
        """Test valid Always URLs."""
        crawler = AlwaysCrawler()

        valid_urls = [
            "https://always.co.kr/products/123",
            "https://www.always.co.kr/products/456",
            "https://m.always.co.kr/products/789",
        ]

        for url in valid_urls:
            assert crawler.is_valid_url(url) is True, f"Should be valid: {url}"

    def test_is_valid_url_invalid(self):
        """Test invalid Always URLs."""
        crawler = AlwaysCrawler()

        invalid_urls = [
            "https://www.google.com",
            "https://www.coupang.com/products/123",
            "",
            "not-a-url",
        ]

        for url in invalid_urls:
            assert crawler.is_valid_url(url) is False, f"Should be invalid: {url}"


class TestCrawlerFactory:
    """Tests for get_crawler factory function."""

    def test_get_coupang_crawler(self):
        """Test getting Coupang crawler."""
        crawler = get_crawler("https://www.coupang.com/products/123")
        assert isinstance(crawler, CoupangCrawler)

    def test_get_always_crawler(self):
        """Test getting Always crawler."""
        crawler = get_crawler("https://www.always.co.kr/products/123")
        assert isinstance(crawler, AlwaysCrawler)

    def test_get_crawler_with_kwargs(self):
        """Test getting crawler with custom settings."""
        crawler = get_crawler(
            "https://www.coupang.com/products/123",
            headless=False,
            timeout=60000,
        )
        assert isinstance(crawler, CoupangCrawler)
        assert crawler.headless is False
        assert crawler.timeout == 60000

    def test_get_crawler_invalid_url(self):
        """Test getting crawler for unsupported URL."""
        with pytest.raises(ValueError) as exc_info:
            get_crawler("https://www.amazon.com/products/123")

        assert "No crawler available" in str(exc_info.value)


class TestUserAgentRotation:
    """Tests for User-Agent rotation."""

    def test_user_agents_not_empty(self):
        """Test that USER_AGENTS list is not empty."""
        assert len(USER_AGENTS) > 0

    def test_user_agents_are_strings(self):
        """Test that all USER_AGENTS are strings."""
        for ua in USER_AGENTS:
            assert isinstance(ua, str)
            assert len(ua) > 0

    def test_get_random_user_agent(self):
        """Test random user agent selection."""
        crawler = CoupangCrawler()

        # Get multiple user agents and ensure they're valid
        user_agents = [crawler._get_random_user_agent() for _ in range(10)]

        for ua in user_agents:
            assert ua in USER_AGENTS


class TestBaseCrawlerConfig:
    """Tests for BaseCrawler configuration."""

    def test_default_config(self):
        """Test default crawler configuration."""
        crawler = CoupangCrawler()

        assert crawler.headless is True
        assert crawler.timeout == 30000
        assert crawler.max_retries == 3
        assert crawler.retry_delay == 1.0

    def test_custom_config(self):
        """Test custom crawler configuration."""
        crawler = CoupangCrawler(
            headless=False,
            timeout=60000,
            max_retries=5,
            retry_delay=2.0,
        )

        assert crawler.headless is False
        assert crawler.timeout == 60000
        assert crawler.max_retries == 5
        assert crawler.retry_delay == 2.0
