#!/usr/bin/env python3
"""
벡터 DB 검증 스크립트.

사용법:
    ./venv/bin/python scripts/test_vector_db.py
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.crawler.base import Review
from src.pipeline.embedder import create_embedder
from src.pipeline.preprocessor import create_default_preprocessor


def main():
    # 환경변수 로드
    load_dotenv()

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("=" * 50)
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다!")
        print()
        print("설정 방법:")
        print('  echo "OPENAI_API_KEY=sk-..." > .env')
        print("=" * 50)
        return

    print("=" * 50)
    print("🔍 벡터 DB 검증 테스트")
    print("=" * 50)

    # 1. 샘플 리뷰 데이터
    sample_reviews = [
        Review(
            text="이 제품 정말 좋아요! 배송도 빠르고 품질도 훌륭합니다. 가격 대비 만족스럽습니다.",
            rating=5.0,
            date="2024-01-15",
        ),
        Review(
            text="배송은 빨랐는데 제품 품질이 기대에 못 미치네요. 가격이 좀 아깝습니다.",
            rating=2.0,
            date="2024-01-14",
        ),
        Review(
            text="무난한 제품입니다. 특별히 좋지도 나쁘지도 않아요. 그냥 평범합니다.",
            rating=3.0,
            date="2024-01-13",
        ),
        Review(
            text="배송이 정말 빨라서 놀랐어요! 주문 다음날 도착했습니다. 제품도 괜찮네요.",
            rating=4.0,
            date="2024-01-12",
        ),
        Review(
            text="사이즈가 생각보다 작아요. 교환하려니 배송비가 아까워서 그냥 씁니다.",
            rating=2.5,
            date="2024-01-11",
        ),
    ]

    # 2. 전처리
    print("\n📝 1단계: 리뷰 전처리")
    preprocessor = create_default_preprocessor(chunk_size=300)
    processed_reviews = preprocessor.process_batch(sample_reviews)
    print(f"   → {len(processed_reviews)}개 리뷰 전처리 완료")

    # 3. 벡터 DB 초기화
    print("\n🗄️  2단계: 벡터 DB 초기화")
    embedder = create_embedder(
        collection_name="verification_test",
        persist_directory="./data/chroma_db_test",
    )
    embedder.reset_collection()
    print("   → 테스트 컬렉션 생성 완료")

    # 4. 임베딩 및 저장
    print("\n📊 3단계: 임베딩 및 저장")
    added_count = embedder.add_reviews(processed_reviews, show_progress=True)
    print(f"   → {added_count}개 청크 저장 완료")

    # 5. 통계 확인
    stats = embedder.get_collection_stats()
    print("\n📈 4단계: 컬렉션 통계")
    print(f"   - 컬렉션 이름: {stats['collection_name']}")
    print(f"   - 총 청크 수: {stats['total_chunks']}")
    print(f"   - 임베딩 모델: {stats['embedding_model']}")

    # 6. 검색 테스트
    print("\n🔎 5단계: 검색 테스트")
    test_queries = [
        "배송이 빠른가요?",
        "품질이 좋은가요?",
        "가격 대비 어때요?",
        "사이즈가 맞을까요?",
    ]

    for query in test_queries:
        print(f"\n   쿼리: '{query}'")
        results = embedder.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            rating = result.metadata.get("rating", "N/A")
            print(f"   [{i}] (평점: {rating}, 점수: {result.score:.3f})")
            print(f"       {result.text[:60]}...")

    # 7. 평점 필터 테스트
    print("\n🏷️  6단계: 평점 필터 테스트")
    print("   조건: 4점 이상 리뷰에서 '좋은 제품' 검색")
    results = embedder.search("좋은 제품", top_k=3, filter_rating_min=4.0)
    if results:
        for i, result in enumerate(results, 1):
            rating = result.metadata.get("rating", "N/A")
            print(f"   [{i}] (평점: {rating}) {result.text[:50]}...")
    else:
        print("   → 조건에 맞는 결과 없음")

    # 8. 정리
    print("\n🧹 7단계: 테스트 데이터 정리")
    embedder.delete_collection()
    print("   → 테스트 컬렉션 삭제 완료")

    print("\n" + "=" * 50)
    print("✅ 벡터 DB 검증 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
