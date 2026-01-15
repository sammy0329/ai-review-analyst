"""
벡터 데이터베이스 임베딩 파이프라인.

ChromaDB와 OpenAI 임베딩을 사용하여 리뷰 데이터를 벡터화하고 검색합니다.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.pipeline.preprocessor import ProcessedReview, TextChunk


@dataclass
class SearchResult:
    """검색 결과 데이터 구조."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None


@dataclass
class EmbedderConfig:
    """임베딩 파이프라인 설정."""

    # ChromaDB 설정
    persist_directory: str = "./data/chroma_db"
    collection_name: str = "reviews"

    # OpenAI 임베딩 설정
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # 검색 설정
    default_top_k: int = 5

    # 배치 처리 설정
    batch_size: int = 100


class ReviewEmbedder:
    """리뷰 임베딩 및 벡터 검색 클래스."""

    def __init__(
        self,
        config: EmbedderConfig | None = None,
        openai_api_key: str | None = None,
    ):
        """
        초기화.

        Args:
            config: 임베딩 설정 (None이면 기본값 사용)
            openai_api_key: OpenAI API 키 (None이면 환경변수에서 로드)
        """
        self.config = config or EmbedderConfig()

        # OpenAI API 키 설정
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API 키가 필요합니다. "
                "OPENAI_API_KEY 환경변수를 설정하거나 openai_api_key 파라미터를 전달하세요."
            )

        # 저장 디렉토리 생성
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)

        # ChromaDB 클라이언트 초기화
        self._chroma_client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # OpenAI 임베딩 모델 초기화
        self._embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self._api_key,
        )

        # LangChain Chroma 래퍼 초기화
        self._vectorstore: Chroma | None = None
        self._init_vectorstore()

    def _init_vectorstore(self) -> None:
        """벡터 스토어 초기화."""
        self._vectorstore = Chroma(
            client=self._chroma_client,
            collection_name=self.config.collection_name,
            embedding_function=self._embeddings,
        )

    @property
    def vectorstore(self) -> Chroma:
        """LangChain Chroma 벡터스토어 반환."""
        if self._vectorstore is None:
            self._init_vectorstore()
        return self._vectorstore

    def add_reviews(
        self,
        reviews: list[ProcessedReview],
        show_progress: bool = False,
    ) -> int:
        """
        전처리된 리뷰들을 벡터 DB에 추가.

        Args:
            reviews: 전처리된 리뷰 리스트
            show_progress: 진행 상황 출력 여부

        Returns:
            추가된 청크 수
        """
        texts = []
        metadatas = []
        ids = []

        for review in reviews:
            for chunk in review.chunks:
                # 청크 ID 생성
                chunk_id = f"{review.text_hash}_{chunk.chunk_index}"

                # 메타데이터 구성
                metadata = {
                    "review_hash": review.text_hash,
                    "chunk_index": chunk.chunk_index,
                    "original_text": review.original_text,  # 원본 텍스트 전체 저장
                }

                # 평점 추가
                if review.rating is not None:
                    metadata["rating"] = review.rating

                # 날짜 추가
                if review.date:
                    metadata["date"] = review.date

                # 리뷰 메타데이터 병합
                for key, value in review.metadata.items():
                    # ChromaDB는 기본 타입만 지원
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value

                texts.append(chunk.text)
                metadatas.append(metadata)
                ids.append(chunk_id)

        if not texts:
            return 0

        # 배치 처리
        total_added = 0
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            self._vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

            total_added += len(batch_texts)

            if show_progress:
                print(f"  진행: {total_added}/{len(texts)} 청크 추가됨")

        return total_added

    def add_chunks(
        self,
        chunks: list[TextChunk],
        review_hash: str,
        review_metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        청크들을 벡터 DB에 추가.

        Args:
            chunks: 텍스트 청크 리스트
            review_hash: 리뷰 해시
            review_metadata: 리뷰 메타데이터

        Returns:
            추가된 청크 수
        """
        texts = []
        metadatas = []
        ids = []

        for chunk in chunks:
            chunk_id = f"{review_hash}_{chunk.chunk_index}"

            metadata = {
                "review_hash": review_hash,
                "chunk_index": chunk.chunk_index,
            }

            if review_metadata:
                for key, value in review_metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value

            texts.append(chunk.text)
            metadatas.append(metadata)
            ids.append(chunk_id)

        if texts:
            self._vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )

        return len(texts)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        filter_rating_min: float | None = None,
        filter_rating_max: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        자연어 쿼리로 유사한 리뷰 검색.

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            min_score: 최소 유사도 점수 (이 값 미만은 필터링)
            filter_rating_min: 최소 평점 필터
            filter_rating_max: 최대 평점 필터
            filter_metadata: 추가 메타데이터 필터

        Returns:
            검색 결과 리스트
        """
        k = top_k or self.config.default_top_k

        # 필터 구성
        where_filter = self._build_filter(
            filter_rating_min=filter_rating_min,
            filter_rating_max=filter_rating_max,
            filter_metadata=filter_metadata,
        )

        # 검색 실행
        if where_filter:
            results = self._vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=where_filter,
            )
        else:
            results = self._vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
            )

        # 결과 변환 (min_score 필터링 적용)
        search_results = []
        for doc, score in results:
            # min_score 임계값 미만은 제외
            if min_score is not None and score < min_score:
                continue

            search_results.append(
                SearchResult(
                    text=doc.page_content,
                    score=score,
                    metadata=doc.metadata,
                    chunk_id=doc.metadata.get("review_hash"),
                )
            )

        return search_results

    def search_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        점수와 함께 검색 결과 반환.

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수

        Returns:
            (텍스트, 점수, 메타데이터) 튜플 리스트
        """
        results = self.search(query, top_k)
        return [(r.text, r.score, r.metadata) for r in results]

    def _build_filter(
        self,
        filter_rating_min: float | None = None,
        filter_rating_max: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        ChromaDB where 필터 구성.

        Args:
            filter_rating_min: 최소 평점
            filter_rating_max: 최대 평점
            filter_metadata: 추가 메타데이터 필터

        Returns:
            ChromaDB where 필터 딕셔너리
        """
        conditions = []

        # 평점 필터
        if filter_rating_min is not None:
            conditions.append({"rating": {"$gte": filter_rating_min}})

        if filter_rating_max is not None:
            conditions.append({"rating": {"$lte": filter_rating_max}})

        # 메타데이터 필터
        if filter_metadata:
            for key, value in filter_metadata.items():
                conditions.append({key: value})

        # 조건 결합
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def get_collection_stats(self) -> dict[str, Any]:
        """
        컬렉션 통계 반환.

        Returns:
            통계 정보 딕셔너리
        """
        collection = self._chroma_client.get_collection(self.config.collection_name)
        count = collection.count()

        return {
            "collection_name": self.config.collection_name,
            "total_chunks": count,
            "persist_directory": self.config.persist_directory,
            "embedding_model": self.config.embedding_model,
        }

    def delete_collection(self) -> bool:
        """
        컬렉션 삭제.

        Returns:
            성공 여부
        """
        try:
            self._chroma_client.delete_collection(self.config.collection_name)
            self._vectorstore = None
            return True
        except Exception:
            return False

    def reset_collection(self) -> None:
        """컬렉션을 삭제하고 새로 생성."""
        self.delete_collection()
        self._init_vectorstore()

    def list_collections(self) -> list[str]:
        """
        모든 컬렉션 이름 반환.

        Returns:
            컬렉션 이름 리스트
        """
        collections = self._chroma_client.list_collections()
        return [c.name for c in collections]

    def get_retriever(
        self,
        search_type: str = "similarity",
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ):
        """
        LangChain Retriever 반환.

        Args:
            search_type: 검색 타입 ("similarity", "mmr")
            top_k: 반환할 결과 수
            filter_metadata: 메타데이터 필터

        Returns:
            LangChain Retriever
        """
        k = top_k or self.config.default_top_k

        search_kwargs = {"k": k}

        if filter_metadata:
            search_kwargs["filter"] = filter_metadata

        return self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )


class CollectionManager:
    """ChromaDB 컬렉션 관리 유틸리티."""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        초기화.

        Args:
            persist_directory: ChromaDB 저장 경로
        """
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

    def list_collections(self) -> list[dict[str, Any]]:
        """
        모든 컬렉션 정보 반환.

        Returns:
            컬렉션 정보 리스트
        """
        collections = self._client.list_collections()
        return [
            {
                "name": c.name,
                "count": c.count(),
                "metadata": c.metadata,
            }
            for c in collections
        ]

    def create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        새 컬렉션 생성.

        Args:
            name: 컬렉션 이름
            metadata: 컬렉션 메타데이터

        Returns:
            성공 여부
        """
        try:
            self._client.get_or_create_collection(name=name)
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False

    def delete_collection(self, name: str) -> bool:
        """
        컬렉션 삭제.

        Args:
            name: 컬렉션 이름

        Returns:
            성공 여부
        """
        try:
            self._client.delete_collection(name)
            return True
        except Exception:
            return False

    def get_collection_info(self, name: str) -> dict[str, Any] | None:
        """
        컬렉션 정보 조회.

        Args:
            name: 컬렉션 이름

        Returns:
            컬렉션 정보 (없으면 None)
        """
        try:
            # 컬렉션 목록에서 확인
            collections = self._client.list_collections()
            for c in collections:
                if c.name == name:
                    return {
                        "name": c.name,
                        "count": c.count(),
                        "metadata": c.metadata,
                    }
            return None
        except Exception:
            return None

    def collection_exists(self, name: str) -> bool:
        """
        컬렉션 존재 여부 확인.

        Args:
            name: 컬렉션 이름

        Returns:
            존재 여부
        """
        try:
            collections = self._client.list_collections()
            return any(c.name == name for c in collections)
        except Exception:
            return False


def create_embedder(
    collection_name: str = "reviews",
    persist_directory: str = "./data/chroma_db",
    openai_api_key: str | None = None,
) -> ReviewEmbedder:
    """
    임베더 인스턴스 생성 헬퍼 함수.

    Args:
        collection_name: 컬렉션 이름
        persist_directory: 저장 경로
        openai_api_key: OpenAI API 키

    Returns:
        ReviewEmbedder 인스턴스
    """
    config = EmbedderConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return ReviewEmbedder(config=config, openai_api_key=openai_api_key)


def main():
    """테스트 실행."""
    from dotenv import load_dotenv

    from src.crawler.base import Review
    from src.pipeline.preprocessor import create_default_preprocessor

    # 환경변수 로드
    load_dotenv()

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print(".env 파일에 OPENAI_API_KEY=sk-... 형식으로 추가하세요.")
        return

    print("=== 벡터 DB 임베딩 테스트 ===\n")

    # 샘플 리뷰 생성
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

    # 전처리
    print("1. 리뷰 전처리 중...")
    preprocessor = create_default_preprocessor(chunk_size=300)
    processed_reviews = preprocessor.process_batch(sample_reviews)
    print(f"   → {len(processed_reviews)}개 리뷰 전처리 완료")

    # 임베더 생성 (테스트용 컬렉션)
    print("\n2. 벡터 DB 초기화 중...")
    embedder = create_embedder(
        collection_name="test_reviews",
        persist_directory="./data/chroma_db_test",
    )

    # 기존 데이터 삭제
    embedder.reset_collection()
    print("   → 테스트 컬렉션 초기화 완료")

    # 리뷰 추가
    print("\n3. 리뷰 임베딩 및 저장 중...")
    added_count = embedder.add_reviews(processed_reviews, show_progress=True)
    print(f"   → {added_count}개 청크 저장 완료")

    # 통계 확인
    stats = embedder.get_collection_stats()
    print(f"\n4. 컬렉션 통계:")
    for key, value in stats.items():
        print(f"   - {key}: {value}")

    # 검색 테스트
    test_queries = [
        "배송이 빠른가요?",
        "품질이 좋은가요?",
        "가격 대비 어때요?",
    ]

    print("\n5. 검색 테스트:")
    for query in test_queries:
        print(f"\n   쿼리: '{query}'")
        results = embedder.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"   [{i}] (점수: {result.score:.3f}) {result.text[:50]}...")

    # 평점 필터 테스트
    print("\n6. 평점 필터 테스트 (4점 이상):")
    results = embedder.search("좋은 제품", top_k=3, filter_rating_min=4.0)
    for i, result in enumerate(results, 1):
        rating = result.metadata.get("rating", "N/A")
        print(f"   [{i}] (평점: {rating}, 점수: {result.score:.3f}) {result.text[:40]}...")

    # 정리
    print("\n7. 테스트 컬렉션 삭제...")
    embedder.delete_collection()
    print("   → 완료")

    print("\n=== 테스트 완료 ===")


if __name__ == "__main__":
    main()
