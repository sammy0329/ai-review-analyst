#!/usr/bin/env python
"""
가짜 리뷰 필터 테스트 스크립트.

실행:
    ./venv/bin/python scripts/test_fake_review_filter.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crawler.base import Review
from src.pipeline.fake_review_filter import (
    FakeReviewFilter,
    create_fake_review_filter,
)


def main():
    print("=" * 60)
    print("가짜 리뷰 필터 테스트")
    print("=" * 60)

    # LLM 없이 규칙 기반만 테스트 (API 키 불필요)
    filter = create_fake_review_filter(use_llm=False)

    # 테스트 리뷰들
    test_reviews = [
        # 정상 리뷰
        Review(
            text="배송이 빠르고 품질도 좋아요. 색상이 사진과 같아서 만족합니다. 다만 사이즈가 조금 작네요.",
            rating=4.0,
        ),
        # 과도한 칭찬
        Review(
            text="인생템이에요! 최고의 제품! 무조건 사세요! 두 번 사도 아깝지 않아요!",
            rating=5.0,
        ),
        # 스팸/광고
        Review(
            text="좋은 제품이에요. 문의 주시면 자세히 알려드릴게요. 카톡 문의 환영합니다.",
            rating=5.0,
        ),
        # 너무 짧은 리뷰
        Review(
            text="좋아요",
            rating=5.0,
        ),
        # 반복 패턴
        Review(
            text="좋아요 좋아요 좋아요 정말 좋아요 너무 좋아요",
            rating=5.0,
        ),
        # 평점-내용 불일치 (높은 평점, 부정적 내용)
        Review(
            text="별로예요. 실망했어요. 후회합니다.",
            rating=5.0,
        ),
        # 구체성 부족
        Review(
            text="그냥 좋습니다. 마음에 들어요. 괜찮네요.",
            rating=4.0,
        ),
    ]

    print("\n리뷰별 분석 결과:")
    print("-" * 60)

    for i, review in enumerate(test_reviews, 1):
        result = filter.check_review(review)

        print(f"\n[리뷰 {i}] 평점: {review.rating}")
        print(f"텍스트: {review.text[:50]}...")
        print(f"  - 의심 여부: {'의심' if result.is_suspicious else '정상'}")
        print(f"  - 신뢰도: {result.confidence:.2f}")
        print(f"  - 가중치: {result.weight:.2f}")
        if result.reasons:
            reasons_str = ", ".join(r.value for r in result.reasons)
            print(f"  - 사유: {reasons_str}")

    # 전체 통계
    print("\n" + "=" * 60)
    print("전체 통계")
    print("=" * 60)

    _, results = filter.filter_reviews(test_reviews)
    stats = filter.get_statistics(results)

    print(f"총 리뷰 수: {stats['total']}")
    print(f"의심 리뷰 수: {stats['suspicious']}")
    print(f"의심 비율: {stats['suspicious_rate']:.1%}")
    print(f"평균 가중치: {stats['avg_weight']:.2f}")
    print("\n사유별 카운트:")
    for reason, count in stats["reason_counts"].items():
        print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
