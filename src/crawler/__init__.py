# Crawler module

from .base import (
    BaseCrawler,
    CrawlerError,
    CrawlerTimeoutError,
    CrawlerBlockedError,
    CrawlResult,
    Review,
    USER_AGENTS,
)
from .coupang import CoupangCrawler
from .always import AlwaysCrawler


def get_crawler(url: str, **kwargs) -> BaseCrawler:
    """
    Factory function to get the appropriate crawler for a URL.

    Args:
        url: The product URL to crawl.
        **kwargs: Additional arguments to pass to the crawler.

    Returns:
        An instance of the appropriate crawler.

    Raises:
        ValueError: If no crawler supports the given URL.
    """
    crawlers = [
        CoupangCrawler(**kwargs),
        AlwaysCrawler(**kwargs),
    ]

    for crawler in crawlers:
        if crawler.is_valid_url(url):
            return crawler

    raise ValueError(f"No crawler available for URL: {url}")


__all__ = [
    # Base
    "BaseCrawler",
    "CrawlerError",
    "CrawlerTimeoutError",
    "CrawlerBlockedError",
    "CrawlResult",
    "Review",
    "USER_AGENTS",
    # Implementations
    "CoupangCrawler",
    "AlwaysCrawler",
    # Factory
    "get_crawler",
]
