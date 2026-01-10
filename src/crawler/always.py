"""
Always (올웨이즈) review crawler implementation.

Note: Always is primarily a mobile app-based service.
This crawler attempts to handle their web version if available.
"""

import re
from urllib.parse import urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from .base import (
    BaseCrawler,
    CrawlerBlockedError,
    CrawlerError,
    CrawlerTimeoutError,
    CrawlResult,
    Review,
)


class AlwaysCrawler(BaseCrawler):
    """
    Crawler for Always (올웨이즈) product reviews.

    Always is a discovery commerce platform focused on
    low-price products with social features.

    Note: The actual selectors may need to be updated
    based on the current site structure.
    """

    DOMAINS = ["always.co.kr", "www.always.co.kr", "m.always.co.kr"]
    REVIEW_SECTION_SELECTOR = ".review-section, .product-review, [data-testid='review']"
    REVIEW_ITEM_SELECTOR = ".review-item, .review-card, [data-testid='review-item']"

    def is_valid_url(self, url: str) -> bool:
        """Check if the URL is a valid Always product URL."""
        try:
            parsed = urlparse(url)
            return any(
                domain in parsed.netloc
                for domain in self.DOMAINS
            )
        except Exception:
            return False

    async def _navigate_to_product(self, page: Page, url: str) -> None:
        """상품 페이지로 이동합니다."""
        # 인간적인 딜레이 추가
        await self._human_like_delay(1.0, 2.0)

        await page.goto(url, wait_until="domcontentloaded")
        await self._human_like_delay(2.0, 4.0)

        # 모바일 앱 리다이렉트 또는 차단 확인
        content = await page.content()
        if "앱에서 확인" in content or "앱 설치" in content:
            raise CrawlerBlockedError(
                "올웨이즈가 모바일 앱으로 리다이렉트합니다. 웹 버전을 사용할 수 없습니다."
            )

    async def _get_product_info(self, page: Page) -> tuple[str | None, str | None, float | None]:
        """Extract product name, price, and average rating."""
        product_name = None
        product_price = None
        avg_rating = None

        try:
            # Try common selectors for product info
            name_selectors = [
                ".product-name",
                ".product-title",
                "h1.title",
                "[data-testid='product-name']",
            ]
            for selector in name_selectors:
                name_elem = await page.query_selector(selector)
                if name_elem:
                    product_name = await name_elem.inner_text()
                    product_name = product_name.strip()
                    break

            # Price
            price_selectors = [
                ".product-price",
                ".sale-price",
                ".price",
                "[data-testid='price']",
            ]
            for selector in price_selectors:
                price_elem = await page.query_selector(selector)
                if price_elem:
                    product_price = await price_elem.inner_text()
                    product_price = product_price.strip()
                    break

            # Rating
            rating_selectors = [
                ".rating-score",
                ".average-rating",
                "[data-testid='rating']",
            ]
            for selector in rating_selectors:
                rating_elem = await page.query_selector(selector)
                if rating_elem:
                    rating_text = await rating_elem.inner_text()
                    try:
                        avg_rating = float(re.search(r"[\d.]+", rating_text).group())
                    except (ValueError, AttributeError):
                        pass
                    break

        except Exception:
            pass

        return product_name, product_price, avg_rating

    async def _scroll_to_load_reviews(self, page: Page, max_scrolls: int = 10) -> None:
        """무한 스크롤을 처리하여 리뷰를 로드합니다."""
        previous_count = 0

        for _ in range(max_scrolls):
            # 현재 리뷰 수 확인
            reviews = await page.query_selector_all(self.REVIEW_ITEM_SELECTOR)
            current_count = len(reviews)

            # 새 리뷰가 로드되지 않으면 중지
            if current_count == previous_count:
                break

            previous_count = current_count

            # 인간적인 스크롤 동작
            await self._human_like_scroll(page)
            await self._human_like_delay(1.0, 2.0)

    async def _parse_reviews(self, page: Page) -> list[Review]:
        """Parse review elements from the page."""
        reviews = []

        # Try multiple selectors
        review_elements = await page.query_selector_all(self.REVIEW_ITEM_SELECTOR)

        # If no reviews found, try alternative selectors
        if not review_elements:
            alternative_selectors = [
                ".review",
                "[class*='review']",
                "[class*='Review']",
            ]
            for selector in alternative_selectors:
                review_elements = await page.query_selector_all(selector)
                if review_elements:
                    break

        for elem in review_elements:
            try:
                review = await self._parse_single_review(elem)
                if review and review.text:
                    reviews.append(review)
            except Exception:
                continue

        return reviews

    async def _parse_single_review(self, elem) -> Review | None:
        """Parse a single review element."""
        try:
            # Review text - try multiple selectors
            text = ""
            text_selectors = [
                ".review-content",
                ".review-text",
                ".content",
                "p",
            ]
            for selector in text_selectors:
                text_elem = await elem.query_selector(selector)
                if text_elem:
                    text = await text_elem.inner_text()
                    text = text.strip()
                    if text:
                        break

            if not text:
                # Try getting all text content
                text = await elem.inner_text()
                text = text.strip()

            if not text or len(text) < 5:
                return None

            # Rating
            rating = None
            rating_selectors = [
                ".rating",
                ".star-rating",
                "[class*='rating']",
            ]
            for selector in rating_selectors:
                rating_elem = await elem.query_selector(selector)
                if rating_elem:
                    rating_text = await rating_elem.get_attribute("data-rating")
                    if not rating_text:
                        rating_text = await rating_elem.inner_text()
                    if rating_text:
                        try:
                            rating = int(float(re.search(r"[\d.]+", rating_text).group()))
                        except (ValueError, AttributeError):
                            pass
                    break

            # Date
            date = None
            date_selectors = [
                ".review-date",
                ".date",
                "time",
                "[class*='date']",
            ]
            for selector in date_selectors:
                date_elem = await elem.query_selector(selector)
                if date_elem:
                    date = await date_elem.inner_text()
                    date = date.strip()
                    break

            # Author
            author = None
            author_selectors = [
                ".reviewer-name",
                ".author",
                ".user-name",
                "[class*='author']",
            ]
            for selector in author_selectors:
                author_elem = await elem.query_selector(selector)
                if author_elem:
                    author = await author_elem.inner_text()
                    author = author.strip()
                    break

            # Images
            images = []
            image_elems = await elem.query_selector_all("img")
            for img in image_elems:
                src = await img.get_attribute("src")
                if src and "review" in src.lower():
                    images.append(src)

            return Review(
                text=text,
                rating=rating,
                date=date,
                author=author,
                images=images,
            )

        except Exception:
            return None

    async def crawl(self, url: str, max_scrolls: int = 10) -> CrawlResult:
        """
        Crawl reviews from an Always product page.

        Args:
            url: The Always product URL.
            max_scrolls: Maximum number of scroll operations for loading reviews.

        Returns:
            CrawlResult containing the crawled data.
        """
        if not self.is_valid_url(url):
            return CrawlResult(
                url=url,
                success=False,
                error_message=f"Invalid Always URL: {url}",
            )

        page = None
        try:
            page = await self._create_page()

            # Navigate to product
            await self._retry_with_backoff(
                self._navigate_to_product,
                page,
                url,
            )

            # Get product info
            product_name, product_price, avg_rating = await self._get_product_info(page)

            # Load reviews with scroll
            await self._scroll_to_load_reviews(page, max_scrolls)

            # Parse reviews
            reviews = await self._parse_reviews(page)

            return CrawlResult(
                url=url,
                product_name=product_name,
                product_price=product_price,
                total_reviews=len(reviews),
                average_rating=avg_rating,
                reviews=reviews,
                success=True,
            )

        except PlaywrightTimeout:
            return CrawlResult(
                url=url,
                success=False,
                error_message="Timeout while loading page",
            )
        except CrawlerBlockedError as e:
            return CrawlResult(
                url=url,
                success=False,
                error_message=str(e),
            )
        except CrawlerError as e:
            return CrawlResult(
                url=url,
                success=False,
                error_message=str(e),
            )
        except Exception as e:
            return CrawlResult(
                url=url,
                success=False,
                error_message=f"Unexpected error: {e}",
            )
        finally:
            if page:
                await page.context.close()
