#!/usr/bin/env python
"""
에이전트 그래프 통합 테스트 스크립트.

실행:
    ./venv/bin/python scripts/test_agent_graph.py
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    print("   .env 파일에 OPENAI_API_KEY=sk-... 형식으로 설정하세요.")
    sys.exit(1)

from src.agents import (
    IntentType,
    create_initial_state,
    create_review_agent_graph,
)
from src.chains.rag_chain import create_rag_chain
from src.crawler.base import Review
from src.pipeline.embedder import create_embedder
from src.pipeline.preprocessor import create_default_preprocessor


def main():
    print("=" * 60)
    print("🤖 LangGraph 에이전트 그래프 통합 테스트")
    print("=" * 60)

    # 1. 샘플 리뷰 데이터
    print("\n📝 1단계: 샘플 리뷰 준비")
    sample_reviews = [
        Review(
            text="배송이 정말 빨라서 좋았어요! 주문 다음날 도착했습니다. 포장도 꼼꼼하게 되어있었어요.",
            rating=5.0,
            date="2024-01-15",
        ),
        Review(
            text="품질이 기대 이상이에요. 가격 대비 정말 좋습니다. 디자인도 예쁘고요.",
            rating=5.0,
            date="2024-01-14",
        ),
        Review(
            text="사이즈가 생각보다 작아요. 교환하려니 번거롭네요. 품질은 괜찮습니다.",
            rating=3.0,
            date="2024-01-13",
        ),
        Review(
            text="가성비 좋아요. 이 가격에 이 정도면 훌륭합니다.",
            rating=4.0,
            date="2024-01-12",
        ),
        Review(
            text="배송은 빨랐는데 제품이 약간 불량이에요. 교환 요청했습니다.",
            rating=2.0,
            date="2024-01-11",
        ),
    ]
    print(f"   → {len(sample_reviews)}개 리뷰 준비 완료")

    # 2. 전처리 & 임베딩
    print("\n🔧 2단계: 전처리 및 벡터화")
    preprocessor = create_default_preprocessor(chunk_size=300)
    processed = preprocessor.process_batch(sample_reviews)
    print(f"   → {len(processed)}개 리뷰 전처리 완료")

    embedder = create_embedder(
        collection_name="agent_test",
        persist_directory="./data/chroma_db_agent_test",
    )
    embedder.reset_collection()
    embedder.add_reviews(processed, show_progress=False)
    print("   → 벡터 DB 저장 완료")

    # 3. RAG Chain & 그래프 생성
    print("\n🔗 3단계: 에이전트 그래프 생성")
    rag_chain = create_rag_chain(embedder=embedder, top_k=3)
    graph = create_review_agent_graph(rag_chain)
    print("   → 그래프 생성 완료")

    # 4. 테스트 질문들
    test_cases = [
        {
            "query": "배송이 빠른가요?",
            "expected_intent": IntentType.QA,
            "description": "일반 Q&A 질문",
        },
        {
            "query": "리뷰 요약해줘",
            "expected_intent": IntentType.SUMMARY,
            "description": "요약 요청",
        },
        {
            "query": "장단점 비교해줘",
            "expected_intent": IntentType.COMPARE,
            "description": "비교 분석 요청",
        },
    ]

    print("\n💬 4단계: 에이전트 테스트")
    print("-" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {test['description']}")
        print(f"❓ 질문: {test['query']}")
        print(f"📋 예상 의도: {test['expected_intent'].value}")

        # 상태 생성 및 그래프 실행
        state = create_initial_state(
            query=test["query"],
            product_name="테스트 제품",
        )

        try:
            result = graph.invoke(state)

            # 결과 출력
            actual_intent = result.get("intent", IntentType.UNKNOWN)
            print(f"✅ 분류 의도: {actual_intent.value}")
            print(f"🎯 분류 방식: {result.get('metadata', {}).get('classification_method', 'N/A')}")
            print(f"📝 응답 (앞 200자):")
            print(f"   {result.get('response', '')[:200]}...")

            # 의도 일치 확인
            if actual_intent == test["expected_intent"]:
                print("✅ 의도 분류 정확!")
            else:
                print(f"⚠️ 의도 불일치 (예상: {test['expected_intent'].value})")

        except Exception as e:
            print(f"❌ 에러 발생: {e}")

        print("-" * 60)

    # 5. 정리
    print("\n🧹 5단계: 테스트 데이터 정리")
    embedder.delete_collection()
    print("   → 테스트 컬렉션 삭제 완료")

    print("\n" + "=" * 60)
    print("✅ 에이전트 그래프 통합 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
