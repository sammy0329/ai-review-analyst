#!/usr/bin/env python
"""
속성 추출기 테스트 스크립트.

실제 AI Hub 데이터로 속성 추출을 테스트합니다.
"""

import os
import sys

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.pipeline.aihub_loader import AIHubDataLoader
from src.pipeline.aspect_extractor import create_aspect_extractor


def main():
    # 환경변수 로드
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    print("=" * 70)
    print("🔍 AI Hub 데이터 기반 속성 추출 테스트")
    print("=" * 70)

    # 1. AI Hub 데이터 로드
    print("\n📦 AI Hub 데이터 로드 중...")
    loader = AIHubDataLoader("data/aihub_data")

    # 패션 카테고리에서 5개 리뷰 로드
    reviews = loader.load_reviews(category="패션", limit=5, as_project_format=True)
    print(f"   → {len(reviews)}개 리뷰 로드 완료")

    # 2. 속성 추출기 생성
    print("\n🤖 속성 추출기 초기화...")
    extractor = create_aspect_extractor(use_cache=True)
    print("   → 완료")

    # 3. 각 리뷰에 대해 속성 추출
    print("\n" + "─" * 70)
    print("📝 속성 추출 결과")
    print("─" * 70)

    all_results = []
    for i, review in enumerate(reviews, 1):
        print(f"\n[리뷰 {i}]")
        print(f"📄 원문: {review.text[:100]}{'...' if len(review.text) > 100 else ''}")
        print(f"⭐ 평점: {review.rating:.1f}/5.0")

        # AI Hub 원본 라벨 (있으면)
        if review.metadata.get("aspects"):
            print(f"🏷️  AI Hub 라벨:")
            for asp in review.metadata["aspects"][:3]:  # 최대 3개만 표시
                print(f"    - {asp.get('aspect', 'N/A')}: {asp.get('polarity', 'N/A')}")

        # LLM 속성 추출
        result = extractor.extract(review.text)
        all_results.append(result)

        print(f"\n🤖 LLM 추출 결과:")
        print(f"   전체 감정: {result.overall_sentiment.value}")
        print(f"   신뢰도: {result.confidence:.2f}")

        if result.aspects:
            print(f"   추출된 속성 ({len(result.aspects)}개):")
            sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
            for asp in result.aspects:
                emoji = sentiment_emoji.get(asp["sentiment"], "❓")
                print(f"     {emoji} [{asp['category']}] {asp['sentiment']}")
                print(f"        → \"{asp['text'][:50]}{'...' if len(asp['text']) > 50 else ''}\"")
        else:
            print("   추출된 속성 없음")

    # 4. 통계 출력
    print("\n" + "─" * 70)
    print("📊 전체 통계")
    print("─" * 70)

    stats = extractor.get_aspect_statistics(all_results)

    print(f"\n📌 분석된 리뷰: {stats['total_reviews']}개")
    print(f"📌 평균 신뢰도: {stats['avg_confidence']:.2f}")

    print("\n🎭 전체 감정 분포:")
    for sentiment, count in stats["overall_sentiment"].items():
        pct = count / stats["total_reviews"] * 100 if stats["total_reviews"] > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"   {sentiment}: {count}개 ({pct:.1f}%) {bar}")

    if stats["aspect_counts"]:
        print("\n📋 속성별 언급 빈도 (TOP 5):")
        for category, count in list(stats["aspect_counts"].items())[:5]:
            print(f"   {category}: {count}회")
            if category in stats["aspect_sentiment"]:
                sent = stats["aspect_sentiment"][category]
                total = sent["positive"] + sent["negative"] + sent["neutral"]
                if total > 0:
                    pos_pct = sent["positive"] / total * 100
                    neg_pct = sent["negative"] / total * 100
                    print(f"      긍정 {pos_pct:.0f}% | 부정 {neg_pct:.0f}%")

    print("\n" + "=" * 70)
    print("✅ 테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
