"""
시맨틱 캐싱 모듈.

유사 질문에 대한 답변을 캐싱하여 API 비용 절감 및 응답 속도 향상.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheConfig:
    """캐시 설정."""

    persist_directory: str = "./data/chroma_db"
    collection_name: str = "qa_cache"
    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = 0.95  # 캐시 히트 임계값
    ttl_days: int = 7  # 캐시 유효 기간 (일)


@dataclass
class CacheResult:
    """캐시 조회 결과."""

    hit: bool
    answer: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    similarity: float = 0.0
    cached_question: str | None = None
    cache_id: str | None = None


@dataclass
class CacheStats:
    """캐시 통계."""

    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    estimated_savings_usd: float = 0.0  # API 비용 절감액


class SemanticCache:
    """시맨틱 캐시 클래스."""

    # 질문당 예상 API 비용 (GPT-4o-mini 기준)
    COST_PER_QUERY_USD = 0.002

    def __init__(
        self,
        config: CacheConfig | None = None,
        openai_api_key: str | None = None,
    ):
        """
        초기화.

        Args:
            config: 캐시 설정
            openai_api_key: OpenAI API 키
        """
        import os

        self.config = config or CacheConfig()
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError("OpenAI API 키가 필요합니다.")

        # 저장 디렉토리 생성
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)

        # ChromaDB 클라이언트 초기화
        self._client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # OpenAI 임베딩 초기화
        self._embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self._api_key,
        )

        # 컬렉션 가져오기 또는 생성
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},  # 코사인 유사도 사용
        )

        # 통계 (메모리)
        self._hits = 0
        self._misses = 0

        logger.info(
            f"SemanticCache 초기화 완료: {self._collection.count()}개 캐시 항목"
        )

    def lookup(self, question: str, product_name: str) -> CacheResult:
        """
        캐시에서 유사 질문 검색.

        Args:
            question: 사용자 질문
            product_name: 제품명 (같은 제품 내에서만 캐시 히트)

        Returns:
            CacheResult 객체
        """
        if self._collection.count() == 0:
            self._misses += 1
            return CacheResult(hit=False)

        # 질문 임베딩 생성
        question_embedding = self._embeddings.embed_query(question)

        # 유사 질문 검색 (같은 제품만)
        results = self._collection.query(
            query_embeddings=[question_embedding],
            n_results=1,
            where={"product_name": product_name},
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            self._misses += 1
            return CacheResult(hit=False)

        # 유사도 계산 (ChromaDB는 distance 반환, cosine의 경우 1 - distance = similarity)
        distance = results["distances"][0][0]
        similarity = 1 - distance

        # 임계값 체크
        if similarity < self.config.similarity_threshold:
            self._misses += 1
            logger.debug(
                f"캐시 미스: similarity={similarity:.3f} < threshold={self.config.similarity_threshold}"
            )
            return CacheResult(hit=False, similarity=similarity)

        # TTL 체크
        metadata = results["metadatas"][0][0]
        cached_at = metadata.get("cached_at")
        if cached_at:
            cached_time = datetime.fromisoformat(cached_at)
            if datetime.now() - cached_time > timedelta(days=self.config.ttl_days):
                # 만료된 캐시 삭제
                cache_id = results["ids"][0][0]
                self._collection.delete(ids=[cache_id])
                self._misses += 1
                logger.info(f"만료된 캐시 삭제: {cache_id}")
                return CacheResult(hit=False)

        # 캐시 히트!
        self._hits += 1
        cache_id = results["ids"][0][0]
        cached_question = results["documents"][0][0]
        answer = metadata.get("answer", "")
        sources_json = metadata.get("sources", "[]")

        try:
            sources = json.loads(sources_json)
        except json.JSONDecodeError:
            sources = []

        logger.info(
            f"캐시 히트! similarity={similarity:.3f}, question='{cached_question[:30]}...'"
        )

        return CacheResult(
            hit=True,
            answer=answer,
            sources=sources,
            similarity=similarity,
            cached_question=cached_question,
            cache_id=cache_id,
        )

    def store(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        product_name: str,
    ) -> str:
        """
        질문-답변 쌍을 캐시에 저장.

        Args:
            question: 사용자 질문
            answer: AI 답변
            sources: 출처 리뷰 목록
            product_name: 제품명

        Returns:
            캐시 ID
        """
        import hashlib

        # 캐시 ID 생성 (질문 + 제품명 해시)
        cache_id = hashlib.md5(
            f"{product_name}:{question}".encode()
        ).hexdigest()[:16]

        # 질문 임베딩 생성
        question_embedding = self._embeddings.embed_query(question)

        # sources를 JSON으로 직렬화 (ChromaDB 메타데이터 제한)
        # ChromaDB 메타데이터는 기본 타입만 지원
        sources_json = json.dumps(sources, ensure_ascii=False)

        # 메타데이터 크기 제한 (ChromaDB 제한 회피)
        if len(sources_json) > 30000:
            # 소스 텍스트 축약
            truncated_sources = []
            for s in sources:
                truncated_sources.append({
                    "text": s.get("text", "")[:500],
                    "rating": s.get("rating"),
                    "date": s.get("date"),
                })
            sources_json = json.dumps(truncated_sources, ensure_ascii=False)

        # 기존 캐시 있으면 업데이트
        existing = self._collection.get(ids=[cache_id])
        if existing["ids"]:
            self._collection.update(
                ids=[cache_id],
                documents=[question],
                embeddings=[question_embedding],
                metadatas=[{
                    "product_name": product_name,
                    "answer": answer,
                    "sources": sources_json,
                    "cached_at": datetime.now().isoformat(),
                }],
            )
            logger.debug(f"캐시 업데이트: {cache_id}")
        else:
            self._collection.add(
                ids=[cache_id],
                documents=[question],
                embeddings=[question_embedding],
                metadatas=[{
                    "product_name": product_name,
                    "answer": answer,
                    "sources": sources_json,
                    "cached_at": datetime.now().isoformat(),
                }],
            )
            logger.debug(f"캐시 저장: {cache_id}")

        return cache_id

    def invalidate(self, cache_id: str) -> bool:
        """
        특정 캐시 항목 무효화.

        Args:
            cache_id: 캐시 ID

        Returns:
            삭제 성공 여부
        """
        try:
            self._collection.delete(ids=[cache_id])
            logger.info(f"캐시 무효화: {cache_id}")
            return True
        except Exception as e:
            logger.error(f"캐시 무효화 실패: {e}")
            return False

    def invalidate_product(self, product_name: str) -> int:
        """
        특정 제품의 모든 캐시 무효화.

        Args:
            product_name: 제품명

        Returns:
            삭제된 캐시 수
        """
        # 해당 제품의 캐시 조회
        results = self._collection.get(
            where={"product_name": product_name},
        )

        if not results["ids"]:
            return 0

        # 삭제
        self._collection.delete(ids=results["ids"])
        count = len(results["ids"])
        logger.info(f"제품 '{product_name}' 캐시 {count}개 무효화")
        return count

    def clear_expired(self) -> int:
        """
        만료된 캐시 정리.

        Returns:
            삭제된 캐시 수
        """
        # 모든 캐시 조회
        results = self._collection.get(include=["metadatas"])

        if not results["ids"]:
            return 0

        expired_ids = []
        cutoff = datetime.now() - timedelta(days=self.config.ttl_days)

        for cache_id, metadata in zip(results["ids"], results["metadatas"]):
            cached_at = metadata.get("cached_at")
            if cached_at:
                try:
                    cached_time = datetime.fromisoformat(cached_at)
                    if cached_time < cutoff:
                        expired_ids.append(cache_id)
                except ValueError:
                    pass

        if expired_ids:
            self._collection.delete(ids=expired_ids)
            logger.info(f"만료된 캐시 {len(expired_ids)}개 삭제")

        return len(expired_ids)

    def clear_all(self) -> int:
        """
        모든 캐시 삭제.

        Returns:
            삭제된 캐시 수
        """
        count = self._collection.count()
        if count > 0:
            # 컬렉션 삭제 후 재생성
            self._client.delete_collection(self.config.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"모든 캐시 삭제: {count}개")
        return count

    def get_stats(self) -> CacheStats:
        """
        캐시 통계 반환.

        Returns:
            CacheStats 객체
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        savings = self._hits * self.COST_PER_QUERY_USD

        return CacheStats(
            total_entries=self._collection.count(),
            total_hits=self._hits,
            total_misses=self._misses,
            hit_rate=round(hit_rate, 1),
            estimated_savings_usd=round(savings, 4),
        )

    def get_cache_entries(self, product_name: str | None = None) -> list[dict]:
        """
        캐시 항목 목록 조회.

        Args:
            product_name: 제품명 (None이면 전체)

        Returns:
            캐시 항목 리스트
        """
        if product_name:
            results = self._collection.get(
                where={"product_name": product_name},
                include=["documents", "metadatas"],
            )
        else:
            results = self._collection.get(include=["documents", "metadatas"])

        entries = []
        for cache_id, doc, metadata in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            entries.append({
                "cache_id": cache_id,
                "question": doc,
                "product_name": metadata.get("product_name"),
                "cached_at": metadata.get("cached_at"),
            })

        return entries


# 싱글톤 인스턴스 (앱 전역에서 공유)
_cache_instance: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """
    SemanticCache 싱글톤 인스턴스 반환.

    Returns:
        SemanticCache 인스턴스
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance


def reset_cache_instance() -> None:
    """싱글톤 인스턴스 리셋 (테스트용)."""
    global _cache_instance
    _cache_instance = None
