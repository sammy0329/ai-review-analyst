#!/usr/bin/env python3
"""
프롬프트 엔지니어링 검증 스크립트.

사용법:
    ./venv/bin/python scripts/test_prompts.py
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
from src.prompts.templates import (
    PromptManager,
    get_prompt,
    list_prompts,
    QA_PROMPT,
    SUMMARY_PROMPT,
)


def main():
    # 환경변수 로드
    load_dotenv()

    print("=" * 60)
    print("프롬프트 엔지니어링 검증 테스트")
    print("=" * 60)

    # 1. 프롬프트 목록 확인
    print("\n1단계: 등록된 프롬프트 목록")
    prompts = list_prompts()
    for name in prompts:
        info = PromptManager.info()[name]
        print(f"   - {name}: {info['description']} (v{info['version']})")

    # 2. Q&A 프롬프트 확인
    print("\n2단계: Q&A 프롬프트 내용")
    qa = get_prompt("qa")
    print(f"   시스템 프롬프트 (처음 100자):")
    print(f"   {qa.system_prompt[:100]}...")
    print(f"\n   Few-shot 예시 수: {len(qa.few_shot_examples)}개")

    # 3. 요약 프롬프트 확인
    print("\n3단계: 요약 프롬프트 내용")
    summary = get_prompt("summary")
    print(f"   할루시네이션 방지 포함: {'근거 기반' in summary.system_prompt}")
    print(f"   수치화 지시 포함: {'수치화' in summary.system_prompt}")

    # 4. API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n" + "=" * 60)
        print("OPENAI_API_KEY 없음 - 프롬프트 구조 검증만 완료")
        print("=" * 60)
        return

    # 5. RAG Chain에서 프롬프트 전환 테스트
    print("\n4단계: RAG Chain 프롬프트 전환 테스트")

    # 샘플 리뷰 데이터
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
    ]

    # 전처리 및 임베딩
    preprocessor = create_default_preprocessor(chunk_size=300)
    processed_reviews = preprocessor.process_batch(sample_reviews)

    embedder = create_embedder(
        collection_name="prompt_test",
        persist_directory="./data/chroma_db_test",
    )
    embedder.reset_collection()
    embedder.add_reviews(processed_reviews)

    # RAG Chain 생성
    rag_chain = create_rag_chain(
        embedder=embedder,
        model_name="gpt-4o-mini",
        top_k=3,
    )

    # Q&A 프롬프트로 질문
    print("\n   [Q&A 프롬프트]")
    print(f"   질문: 배송이 빠른가요?")
    result = rag_chain.query("배송이 빠른가요?")
    print(f"   답변: {result.answer[:150]}...")

    # 요약 프롬프트로 전환
    print("\n   [요약 프롬프트로 전환]")
    rag_chain.set_prompt("summary")
    print("   프롬프트 전환 완료")

    # 정리
    embedder.delete_collection()

    print("\n" + "=" * 60)
    print("프롬프트 엔지니어링 검증 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
