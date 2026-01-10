"""
크롤러 디버그 스크립트 - HTML 구조 확인용
"""

import asyncio
import random
import sys
sys.path.insert(0, ".")

from playwright.async_api import async_playwright
from playwright_stealth import stealth_async


async def debug_page(url: str, headless: bool = False):
    """페이지 HTML 구조를 확인합니다."""
    print(f"디버깅: {url}")
    print(f"헤드리스 모드: {headless}\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-infobars",
                "--disable-extensions",
            ],
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            color_scheme="light",
        )
        page = await context.new_page()

        # stealth 설정 적용
        await stealth_async(page)

        # 자연스러운 브라우징을 위해 먼저 쿠팡 메인 페이지 방문
        print("쿠팡 메인 페이지 먼저 방문...")
        await page.goto("https://www.coupang.com", wait_until="networkidle")
        await asyncio.sleep(random.uniform(2.0, 4.0))

        # 인간적인 스크롤 동작
        for _ in range(3):
            await page.evaluate(f"window.scrollTo(0, {random.randint(200, 500)})")
            await asyncio.sleep(random.uniform(0.5, 1.0))

        print("상품 페이지로 이동 중...")
        await asyncio.sleep(random.uniform(1.0, 2.0))
        await page.goto(url, wait_until="networkidle")
        await asyncio.sleep(random.uniform(3.0, 5.0))

        # 스크린샷 저장
        await page.screenshot(path="data/debug_screenshot.png", full_page=False)
        print("📸 스크린샷 저장: data/debug_screenshot.png")

        # HTML 저장
        html = await page.content()
        with open("data/debug_page.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("📄 HTML 저장: data/debug_page.html")

        # 주요 요소 확인
        print("\n🔍 요소 탐색:\n")

        # 상품명 찾기
        selectors_to_try = {
            "상품명": [
                "h1.prod-buy-header__title",
                "h2.prod-buy-header__title",
                ".prod-buy-header__title",
                "h1[class*='title']",
                ".product-title",
                "h1",
            ],
            "가격": [
                ".prod-sale-price .total-price strong",
                ".prod-price .total-price",
                ".prod-coupon-price .total-price",
                "span.total-price strong",
                "[class*='price'] strong",
            ],
            "평점": [
                ".rds-rating-score",
                ".rating-star-num",
                "[class*='rating']",
                ".star-rating",
            ],
            "리뷰탭": [
                "a[data-tab='review']",
                ".tab-titles__btn[data-tab='review']",
                "a[href*='review']",
                "[class*='review-tab']",
            ],
            "리뷰 컨테이너": [
                ".sdp-review__article__list__review",
                ".js-review-article",
                "[class*='review-article']",
                ".review-list",
                "[class*='ReviewList']",
            ],
        }

        for name, selectors in selectors_to_try.items():
            print(f"  {name}:")
            found = False
            for selector in selectors:
                try:
                    elem = await page.query_selector(selector)
                    if elem:
                        text = await elem.inner_text()
                        text = text[:50].strip().replace('\n', ' ')
                        print(f"    ✅ {selector}")
                        print(f"       → \"{text}\"")
                        found = True
                        break
                except Exception as e:
                    pass
            if not found:
                print(f"    ❌ 찾지 못함")

        # 리뷰 탭 클릭 시도
        print("\n🔄 리뷰 탭 클릭 시도...")
        review_tab = await page.query_selector("a[data-tab='review']")
        if review_tab:
            await review_tab.click()
            await asyncio.sleep(2)
            print("   리뷰 탭 클릭 완료")

            # 리뷰 탭 클릭 후 스크린샷
            await page.screenshot(path="data/debug_screenshot_reviews.png", full_page=False)
            print("📸 리뷰 탭 스크린샷: data/debug_screenshot_reviews.png")

            # 리뷰 요소 다시 확인
            review_selectors = [
                ".sdp-review__article__list__review",
                ".js-review-article",
                "[class*='review']",
                "article[class*='review']",
            ]
            print("\n🔍 리뷰 요소 확인:")
            for selector in review_selectors:
                elems = await page.query_selector_all(selector)
                if elems:
                    print(f"   ✅ {selector}: {len(elems)}개 발견")

        await browser.close()
        print("\n완료!")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.coupang.com/vp/products/7335597976"
    # --headless 플래그로 헤드리스 모드 활성화
    headless = "--headless" in sys.argv
    asyncio.run(debug_page(url, headless=headless))
