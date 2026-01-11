"""
Base crawler class with common functionality.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

# playwright_stealth 호환성 처리
try:
    from playwright_stealth import stealth_async
except ImportError:
    from playwright_stealth import Stealth
    async def stealth_async(page: Page) -> None:
        """Fallback stealth wrapper."""
        await Stealth().apply(page)

# User-Agent rotation pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


@dataclass
class Review:
    """Data class representing a single review."""

    text: str
    rating: int | float | None = None
    date: str | None = None
    author: str | None = None
    option: str | None = None
    helpful_count: int | None = None
    verified_purchase: bool = False
    images: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawlResult:
    """Data class representing the result of a crawl operation."""

    url: str
    product_name: str | None = None
    product_price: str | None = None
    total_reviews: int = 0
    average_rating: float | None = None
    reviews: list[Review] = field(default_factory=list)
    crawled_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "product_name": self.product_name,
            "product_price": self.product_price,
            "total_reviews": self.total_reviews,
            "average_rating": self.average_rating,
            "reviews": [
                {
                    "text": r.text,
                    "rating": r.rating,
                    "date": r.date,
                    "author": r.author,
                    "option": r.option,
                    "helpful_count": r.helpful_count,
                    "verified_purchase": r.verified_purchase,
                    "images": r.images,
                    "metadata": r.metadata,
                }
                for r in self.reviews
            ],
            "crawled_at": self.crawled_at.isoformat(),
            "success": self.success,
            "error_message": self.error_message,
        }


class CrawlerError(Exception):
    """Base exception for crawler errors."""

    pass


class CrawlerTimeoutError(CrawlerError):
    """Raised when a crawl operation times out."""

    pass


class CrawlerBlockedError(CrawlerError):
    """Raised when the crawler is blocked by the target site."""

    pass


class BaseCrawler(ABC):
    """
    Abstract base class for web crawlers.

    Provides common functionality for Playwright-based crawling:
    - Browser lifecycle management
    - User-Agent rotation
    - Retry logic with exponential backoff
    - Error handling
    """

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the crawler.

        Args:
            headless: Whether to run browser in headless mode.
            timeout: Default timeout in milliseconds.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay between retries in seconds.
        """
        self.headless = headless
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._browser: Browser | None = None
        self._playwright = None

    def _get_random_user_agent(self) -> str:
        """Get a random User-Agent string."""
        return random.choice(USER_AGENTS)

    async def _init_browser(self) -> Browser:
        """Initialize the browser instance with anti-detection settings."""
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-infobars",
                    "--window-size=1920,1080",
                    "--disable-extensions",
                ],
            )
        return self._browser

    async def _create_context(self) -> BrowserContext:
        """Create a new browser context with anti-detection settings."""
        browser = await self._init_browser()

        # Randomize viewport slightly to avoid fingerprinting
        width = 1920 + random.randint(-100, 100)
        height = 1080 + random.randint(-50, 50)

        context = await browser.new_context(
            user_agent=self._get_random_user_agent(),
            viewport={"width": width, "height": height},
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            color_scheme="light",
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
            },
        )

        return context

    async def _create_page(self) -> Page:
        """Create a new page with anti-detection settings."""
        context = await self._create_context()
        page = await context.new_page()

        # Apply stealth settings to avoid bot detection
        await stealth_async(page)

        page.set_default_timeout(self.timeout)
        return page

    async def _human_like_delay(self, min_sec: float = 0.5, max_sec: float = 2.0) -> None:
        """Add random delay to simulate human behavior."""
        delay = random.uniform(min_sec, max_sec)
        await asyncio.sleep(delay)

    async def _human_like_scroll(self, page: Page) -> None:
        """Perform human-like scrolling behavior."""
        # Scroll down in small increments with random delays
        scroll_height = await page.evaluate("document.body.scrollHeight")
        current_position = 0
        viewport_height = 1080

        while current_position < scroll_height:
            # Random scroll amount (300-700 pixels)
            scroll_amount = random.randint(300, 700)
            current_position = min(current_position + scroll_amount, scroll_height)

            await page.evaluate(f"window.scrollTo(0, {current_position})")
            await self._human_like_delay(0.3, 0.8)

            # Occasionally scroll up a bit (like a human reading)
            if random.random() < 0.2:
                back_scroll = random.randint(50, 150)
                current_position = max(0, current_position - back_scroll)
                await page.evaluate(f"window.scrollTo(0, {current_position})")
                await self._human_like_delay(0.2, 0.5)

    async def _close_browser(self) -> None:
        """Close the browser and cleanup resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def _scroll_to_load(
        self,
        page: Page,
        scroll_count: int = 10,
        scroll_delay: float = 1.0,
    ) -> None:
        """
        Scroll the page to load dynamic content.

        Args:
            page: The page to scroll.
            scroll_count: Number of scroll operations.
            scroll_delay: Delay between scrolls in seconds.
        """
        for _ in range(scroll_count):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(scroll_delay)

    async def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs,
    ):
        """
        Execute a function with exponential backoff retry.

        Args:
            func: Async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function.

        Raises:
            CrawlerError: If all retries are exhausted.
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        raise CrawlerError(f"Failed after {self.max_retries} retries: {last_error}")

    @abstractmethod
    async def crawl(self, url: str) -> CrawlResult:
        """
        Crawl reviews from the given URL.

        Args:
            url: The product URL to crawl.

        Returns:
            CrawlResult containing the crawled data.
        """
        pass

    @abstractmethod
    def is_valid_url(self, url: str) -> bool:
        """
        Check if the URL is valid for this crawler.

        Args:
            url: The URL to validate.

        Returns:
            True if the URL is valid for this crawler.
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self._init_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_browser()
