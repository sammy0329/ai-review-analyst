#!/usr/bin/env python3
"""
RAG Chain 검증 스크립트.

사용법:
    ./venv/bin/python scripts/test_rag_chain.py
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
from src.chains.rag_chain import create_rag_chain


def main():
    # 환경변수 로드
    load_dotenv()

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("=" * 60)
        print("OPENAI_API_KEY가 설정되지 않았습니다!")
        print()
        print("설정 방법:")
        print('  echo "OPENAI_API_KEY=sk-..." > .env')
        print("=" * 60)
        return

    print("=" * 60)
    print("RAG Chain 검증 테스트")
    print("=" * 60)

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
    print("\n1단계: 리뷰 전처리")
    preprocessor = create_default_preprocessor(chunk_size=300)
    processed_reviews = preprocessor.process_batch(sample_reviews)
    print(f"   {len(processed_reviews)}개 리뷰 전처리 완료")

    # 3. 벡터 DB에 저장
    print("\n2단계: 벡터 DB 저장")
    embedder = create_embedder(
        collection_name="rag_chain_test",
        persist_directory="./data/chroma_db_test",
    )
    embedder.reset_collection()
    added_count = embedder.add_reviews(processed_reviews, show_progress=True)
    print(f"   {added_count}개 청크 저장 완료")

    # 4. RAG Chain 생성
    print("\n3단계: RAG Chain 초기화")
    rag_chain = create_rag_chain(
        embedder=embedder,
        model_name="gpt-4o-mini",
        top_k=3,
    )
    print("   RAG Chain 생성 완료")

    # 5. 질의응답 테스트
    test_questions = [
        "배송이 빠른가요?",
        "품질은 어떤가요?",
        "가격 대비 가치가 있나요?",
    ]

    print("\n4단계: 질의응답 테스트")
    for question in test_questions:
        print(f"\n{'─' * 50}")
        print(f"질문: {question}")
        print(f"{'─' * 50}")

        result = rag_chain.query_with_sources(question)

        print(f"\n답변:\n{result['answer']}")

        print(f"\n참조 리뷰 ({len(result['sources'])}개):")
        for i, source in enumerate(result['sources'], 1):
            rating = source.get('rating', 'N/A')
            print(f"   [{i}] (평점: {rating}) {source['text'][:50]}...")

    # 6. 스트리밍 테스트
    print(f"\n{'─' * 50}")
    print("5단계: 스트리밍 테스트")
    print(f"{'─' * 50}")
    print("\n질문: 이 제품을 추천하시나요?")
    print("\n답변: ", end="", flush=True)

    for chunk in rag_chain.stream("이 제품을 추천하시나요?"):
        print(chunk, end="", flush=True)
    print()

    # 7. 정리
    print("\n6단계: 테스트 데이터 정리")
    embedder.delete_collection()
    print("   테스트 컬렉션 삭제 완료")

    print("\n" + "=" * 60)
    print("RAG Chain 검증 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
