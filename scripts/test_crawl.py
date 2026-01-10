"""
크롤러 테스트 스크립트

사용법:
    python scripts/test_crawl.py <URL>

예시:
    python scripts/test_crawl.py "https://www.coupang.com/vp/products/123456"
"""

import asyncio
import json
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, ".")

from src.crawler import get_crawler, CrawlerError


async def test_crawl(url: str):
    """URL을 크롤링하고 결과를 출력합니다."""
    print(f"\n{'='*60}")
    print(f"크롤링 테스트: {url}")
    print(f"{'='*60}\n")

    try:
        # URL에 맞는 크롤러 가져오기
        crawler = get_crawler(url, headless=True, timeout=30000)
        print(f"사용 크롤러: {crawler.__class__.__name__}")

        # 크롤링 실행
        print("크롤링 중...")
        async with crawler:
            result = await crawler.crawl(url, max_pages=3)

        # 결과 출력
        if result.success:
            print(f"\n✅ 크롤링 성공!")
            print(f"\n📦 상품 정보:")
            print(f"   - 상품명: {result.product_name or '(추출 실패)'}")
            print(f"   - 가격: {result.product_price or '(추출 실패)'}")
            print(f"   - 평균 평점: {result.average_rating or '(추출 실패)'}")
            print(f"   - 수집된 리뷰 수: {result.total_reviews}")

            if result.reviews:
                print(f"\n📝 리뷰 샘플 (최대 5개):")
                for i, review in enumerate(result.reviews[:5], 1):
                    print(f"\n   [{i}] ⭐ {review.rating or '?'}점")
                    print(f"       {review.text[:100]}{'...' if len(review.text) > 100 else ''}")
                    if review.date:
                        print(f"       📅 {review.date}")
                    if review.option:
                        print(f"       🏷️ {review.option}")

            # JSON으로 저장
            output_file = "data/crawl_result.json"
            import os
            os.makedirs("data", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\n💾 전체 결과 저장됨: {output_file}")

        else:
            print(f"\n❌ 크롤링 실패: {result.error_message}")

    except ValueError as e:
        print(f"\n❌ 지원하지 않는 URL: {e}")
    except CrawlerError as e:
        print(f"\n❌ 크롤러 에러: {e}")
    except Exception as e:
        print(f"\n❌ 예상치 못한 에러: {e}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n지원 플랫폼:")
        print("  - 쿠팡: https://www.coupang.com/vp/products/...")
        print("  - 올웨이즈: https://always.co.kr/products/...")
        sys.exit(1)

    url = sys.argv[1]
    asyncio.run(test_crawl(url))


if __name__ == "__main__":
    main()
