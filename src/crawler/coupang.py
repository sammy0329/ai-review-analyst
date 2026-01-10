"""
Coupang review crawler implementation.
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


class CoupangCrawler(BaseCrawler):
    """
    Crawler for Coupang product reviews.

    Handles:
    - Dynamic page loading with infinite scroll
    - Review pagination
    - Rating extraction
    - Purchase option extraction
    """

    DOMAIN = "www.coupang.com"
    REVIEW_TAB_SELECTOR = "a.tab-titles__btn[data-tab='review']"
    REVIEW_CONTAINER_SELECTOR = ".sdp-review__article__list"
    REVIEW_ITEM_SELECTOR = ".sdp-review__article__list__review"

    def is_valid_url(self, url: str) -> bool:
        """Check if the URL is a valid Coupang product URL."""
        try:
            parsed = urlparse(url)
            return (
                parsed.netloc == self.DOMAIN
                and "/products/" in parsed.path
            )
        except Exception:
            return False

    def _extract_product_id(self, url: str) -> str | None:
        """Extract product ID from Coupang URL."""
        match = re.search(r"/products/(\d+)", url)
        return match.group(1) if match else None

    async def _navigate_to_reviews(self, page: Page, url: str) -> None:
        """Navigate to the product page and click on reviews tab."""
        # Add human-like delay before navigation
        await self._human_like_delay(1.0, 2.0)

        await page.goto(url, wait_until="domcontentloaded")

        # Wait for page to load with random delay
        await self._human_like_delay(2.0, 4.0)

        # 봇 탐지/차단 여부 확인
        content = await page.content()
        page_title = await page.title()

        if (
            await page.query_selector(".captcha")
            or "access denied" in content.lower()
            or "access denied" in page_title.lower()
            or "blocked" in content.lower()
            or "reference #" in content.lower()
            or "edgesuite.net" in content.lower()
        ):
            raise CrawlerBlockedError(
                "쿠팡이 요청을 차단했습니다. (Access Denied) "
                "쿠팡은 강력한 봇 탐지 시스템을 사용합니다. "
                "프록시 서비스나 CAPTCHA 해결 서비스가 필요할 수 있습니다."
            )

        # Human-like scroll before clicking review tab
        await self._human_like_scroll(page)
        await self._human_like_delay(0.5, 1.5)

        # Click on review tab if exists
        review_tab = await page.query_selector(self.REVIEW_TAB_SELECTOR)
        if review_tab:
            await review_tab.click()
            await self._human_like_delay(1.0, 2.0)

    async def _get_product_info(self, page: Page) -> tuple[str | None, str | None, float | None]:
        """Extract product name, price, and average rating."""
        product_name = None
        product_price = None
        avg_rating = None

        try:
            # Product name
            name_elem = await page.query_selector("h1.prod-buy-header__title")
            if name_elem:
                product_name = await name_elem.inner_text()
                product_name = product_name.strip()

            # Price
            price_elem = await page.query_selector(".prod-sale-price .total-price strong")
            if price_elem:
                product_price = await price_elem.inner_text()
                product_price = product_price.strip()

            # Average rating
            rating_elem = await page.query_selector(".rds-rating-score")
            if rating_elem:
                rating_text = await rating_elem.inner_text()
                try:
                    avg_rating = float(rating_text.strip())
                except ValueError:
                    pass

        except Exception:
            pass

        return product_name, product_price, avg_rating

    async def _load_more_reviews(self, page: Page, max_pages: int = 5) -> None:
        """Load more reviews by clicking 'more' button or scrolling."""
        for _ in range(max_pages):
            try:
                # Try to find and click "more reviews" button
                more_btn = await page.query_selector(".sdp-review__article__list__nav__next")
                if more_btn:
                    is_disabled = await more_btn.get_attribute("disabled")
                    if not is_disabled:
                        await more_btn.click()
                        await self._human_like_delay(1.0, 2.5)
                    else:
                        break
                else:
                    # Scroll to load more with human-like behavior
                    await self._human_like_scroll(page)
                    break
            except Exception:
                break

    async def _parse_reviews(self, page: Page) -> list[Review]:
        """Parse review elements from the page."""
        reviews = []

        review_elements = await page.query_selector_all(self.REVIEW_ITEM_SELECTOR)

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
            # Review text
            text_elem = await elem.query_selector(".sdp-review__article__list__review__content")
            text = ""
            if text_elem:
                text = await text_elem.inner_text()
                text = text.strip()

            if not text:
                return None

            # Rating
            rating = None
            rating_elem = await elem.query_selector(".sdp-review__article__list__review__rating")
            if rating_elem:
                rating_class = await rating_elem.get_attribute("class")
                if rating_class:
                    match = re.search(r"rating(\d+)", rating_class)
                    if match:
                        rating = int(match.group(1))

            # Date
            date = None
            date_elem = await elem.query_selector(".sdp-review__article__list__review__date")
            if date_elem:
                date = await date_elem.inner_text()
                date = date.strip()

            # Author
            author = None
            author_elem = await elem.query_selector(".sdp-review__article__list__review__name")
            if author_elem:
                author = await author_elem.inner_text()
                author = author.strip()

            # Purchase option
            option = None
            option_elem = await elem.query_selector(".sdp-review__article__list__review__option")
            if option_elem:
                option = await option_elem.inner_text()
                option = option.strip()

            # Helpful count
            helpful_count = None
            helpful_elem = await elem.query_selector(".sdp-review__article__list__review__helpful__count")
            if helpful_elem:
                helpful_text = await helpful_elem.inner_text()
                try:
                    helpful_count = int(helpful_text.strip())
                except ValueError:
                    pass

            # Images
            images = []
            image_elems = await elem.query_selector_all(".sdp-review__article__list__review__content__image img")
            for img in image_elems:
                src = await img.get_attribute("src")
                if src:
                    images.append(src)

            return Review(
                text=text,
                rating=rating,
                date=date,
                author=author,
                option=option,
                helpful_count=helpful_count,
                images=images,
            )

        except Exception:
            return None

    async def crawl(self, url: str, max_pages: int = 5) -> CrawlResult:
        """
        Crawl reviews from a Coupang product page.

        Args:
            url: The Coupang product URL.
            max_pages: Maximum number of review pages to load.

        Returns:
            CrawlResult containing the crawled data.
        """
        if not self.is_valid_url(url):
            return CrawlResult(
                url=url,
                success=False,
                error_message=f"Invalid Coupang URL: {url}",
            )

        page = None
        try:
            page = await self._create_page()

            # Navigate to reviews
            await self._retry_with_backoff(
                self._navigate_to_reviews,
                page,
                url,
            )

            # Get product info
            product_name, product_price, avg_rating = await self._get_product_info(page)

            # Load more reviews
            await self._load_more_reviews(page, max_pages)

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
