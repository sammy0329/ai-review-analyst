"""
RAG (Retrieval-Augmented Generation) Chain 구현.

리뷰 데이터를 기반으로 질문에 답변하는 RAG 체인을 제공합니다.
"""

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.pipeline.embedder import ReviewEmbedder, create_embedder
from src.prompts.templates import QA_PROMPT, SUMMARY_PROMPT, get_prompt, PromptTemplate


# 기본 프롬프트 (prompts 모듈에서 가져옴)
SYSTEM_PROMPT = QA_PROMPT.system_prompt
USER_PROMPT_TEMPLATE = QA_PROMPT.user_prompt_template


@dataclass
class RAGResponse:
    """RAG 응답 데이터 구조."""

    answer: str
    source_documents: list[Document] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """RAG Chain 설정."""

    # LLM 설정
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    streaming: bool = True

    # 검색 설정
    top_k: int = 5
    search_type: str = "similarity"  # "similarity" or "mmr"

    # 프롬프트 설정
    system_prompt: str = SYSTEM_PROMPT
    user_prompt_template: str = USER_PROMPT_TEMPLATE


class ReviewRAGChain:
    """리뷰 RAG Chain 클래스."""

    def __init__(
        self,
        embedder: ReviewEmbedder | None = None,
        config: RAGConfig | None = None,
        openai_api_key: str | None = None,
    ):
        """
        초기화.

        Args:
            embedder: ReviewEmbedder 인스턴스 (None이면 새로 생성)
            config: RAG 설정 (None이면 기본값 사용)
            openai_api_key: OpenAI API 키
        """
        self.config = config or RAGConfig()

        # API 키 설정
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API 키가 필요합니다. "
                "OPENAI_API_KEY 환경변수를 설정하거나 openai_api_key 파라미터를 전달하세요."
            )

        # Embedder 설정
        self.embedder = embedder or create_embedder(openai_api_key=self._api_key)

        # LLM 초기화
        self._llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=self.config.streaming,
            openai_api_key=self._api_key,
        )

        # Retriever 설정
        self._retriever = self.embedder.get_retriever(
            search_type=self.config.search_type,
            top_k=self.config.top_k,
        )

        # 프롬프트 템플릿 생성
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.user_prompt_template),
        ])

        # RAG Chain 구성
        self._chain = self._build_chain()

    def _build_chain(self):
        """RAG Chain 구성."""

        def format_docs(docs: list[Document]) -> str:
            """문서 포맷팅."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                rating = doc.metadata.get("rating", "N/A")
                date = doc.metadata.get("date", "N/A")
                text = doc.page_content

                formatted.append(
                    f"[리뷰 {i}] (평점: {rating}, 날짜: {date})\n{text}"
                )

            return "\n\n".join(formatted)

        # Chain 구성: 검색 → 포맷팅 → 프롬프트 → LLM → 파싱
        chain = (
            {
                "context": self._retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self._prompt
            | self._llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str) -> RAGResponse:
        """
        질문에 대한 답변 생성.

        Args:
            question: 사용자 질문

        Returns:
            RAGResponse 객체
        """
        # 관련 문서 검색
        source_docs = self._retriever.invoke(question)

        # 답변 생성
        answer = self._chain.invoke(question)

        return RAGResponse(
            answer=answer,
            source_documents=source_docs,
            metadata={
                "model": self.config.model_name,
                "top_k": self.config.top_k,
                "num_sources": len(source_docs),
            },
        )

    def stream(self, question: str) -> Iterator[str]:
        """
        스트리밍 방식으로 답변 생성.

        Args:
            question: 사용자 질문

        Yields:
            답변 텍스트 청크
        """
        for chunk in self._chain.stream(question):
            yield chunk

    async def astream(self, question: str) -> AsyncIterator[str]:
        """
        비동기 스트리밍 방식으로 답변 생성.

        Args:
            question: 사용자 질문

        Yields:
            답변 텍스트 청크
        """
        async for chunk in self._chain.astream(question):
            yield chunk

    def query_with_sources(self, question: str) -> dict[str, Any]:
        """
        출처와 함께 답변 반환.

        Args:
            question: 사용자 질문

        Returns:
            답변과 출처 정보를 포함한 딕셔너리
        """
        response = self.query(question)

        sources = []
        for doc in response.source_documents:
            # original_text가 있으면 사용, 없으면 청크 텍스트 사용
            text = doc.metadata.get("original_text") or doc.page_content
            sources.append({
                "text": text,
                "rating": doc.metadata.get("rating"),
                "date": doc.metadata.get("date"),
                "review_hash": doc.metadata.get("review_hash"),
            })

        return {
            "answer": response.answer,
            "sources": sources,
            "metadata": response.metadata,
        }

    def update_config(self, **kwargs) -> None:
        """
        설정 업데이트.

        Args:
            **kwargs: 업데이트할 설정 값들
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # LLM 재초기화
        if any(k in kwargs for k in ["model_name", "temperature", "max_tokens", "streaming"]):
            self._llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming,
                openai_api_key=self._api_key,
            )

        # Retriever 재설정
        if any(k in kwargs for k in ["top_k", "search_type"]):
            self._retriever = self.embedder.get_retriever(
                search_type=self.config.search_type,
                top_k=self.config.top_k,
            )

        # Chain 재구성
        self._chain = self._build_chain()

    def set_prompt(self, prompt_name: str) -> None:
        """
        프롬프트 템플릿 변경.

        Args:
            prompt_name: 프롬프트 이름 ("qa", "summary", "compare", "sentiment")
        """
        prompt = get_prompt(prompt_name)
        self.config.system_prompt = prompt.system_prompt
        self.config.user_prompt_template = prompt.user_prompt_template

        # 프롬프트 템플릿 재생성
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.user_prompt_template),
        ])

        # Chain 재구성
        self._chain = self._build_chain()

    def set_prompt_template(self, prompt_template: PromptTemplate) -> None:
        """
        커스텀 프롬프트 템플릿 설정.

        Args:
            prompt_template: PromptTemplate 인스턴스
        """
        self.config.system_prompt = prompt_template.system_prompt
        self.config.user_prompt_template = prompt_template.user_prompt_template

        # 프롬프트 템플릿 재생성
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.user_prompt_template),
        ])

        # Chain 재구성
        self._chain = self._build_chain()

    @property
    def retriever(self):
        """Retriever 반환."""
        return self._retriever

    @property
    def llm(self):
        """LLM 반환."""
        return self._llm


def create_rag_chain(
    embedder: ReviewEmbedder | None = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_k: int = 5,
    streaming: bool = True,
    openai_api_key: str | None = None,
) -> ReviewRAGChain:
    """
    RAG Chain 생성 헬퍼 함수.

    Args:
        embedder: ReviewEmbedder 인스턴스
        model_name: LLM 모델명
        temperature: 온도 설정
        top_k: 검색할 문서 수
        streaming: 스트리밍 사용 여부
        openai_api_key: OpenAI API 키

    Returns:
        ReviewRAGChain 인스턴스
    """
    config = RAGConfig(
        model_name=model_name,
        temperature=temperature,
        top_k=top_k,
        streaming=streaming,
    )

    return ReviewRAGChain(
        embedder=embedder,
        config=config,
        openai_api_key=openai_api_key,
    )


def main():
    """테스트 실행."""
    from dotenv import load_dotenv

    from src.crawler.base import Review
    from src.pipeline.preprocessor import create_default_preprocessor

    # 환경변수 로드
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    print("=" * 60)
    print("🤖 RAG Chain 테스트")
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
    print("\n📝 1단계: 리뷰 전처리 및 임베딩")
    preprocessor = create_default_preprocessor(chunk_size=300)
    processed_reviews = preprocessor.process_batch(sample_reviews)
    print(f"   → {len(processed_reviews)}개 리뷰 전처리 완료")

    # 3. Embedder 생성 및 데이터 추가
    embedder = create_embedder(
        collection_name="rag_test_reviews",
        persist_directory="./data/chroma_db_test",
    )
    embedder.reset_collection()
    embedder.add_reviews(processed_reviews)
    print(f"   → 벡터 DB에 저장 완료")

    # 4. RAG Chain 생성
    print("\n🔗 2단계: RAG Chain 초기화")
    rag_chain = create_rag_chain(
        embedder=embedder,
        model_name="gpt-4o-mini",
        top_k=3,
    )
    print("   → RAG Chain 생성 완료")

    # 5. 질의응답 테스트
    test_questions = [
        "배송이 빠른가요?",
        "품질은 어떤가요?",
        "가격 대비 가치가 있나요?",
        "사이즈는 어떤가요?",
    ]

    print("\n💬 3단계: 질의응답 테스트")
    for question in test_questions:
        print(f"\n{'─' * 50}")
        print(f"❓ 질문: {question}")
        print(f"{'─' * 50}")

        result = rag_chain.query_with_sources(question)

        print(f"\n📝 답변:\n{result['answer']}")

        print(f"\n📚 참조 리뷰 ({len(result['sources'])}개):")
        for i, source in enumerate(result['sources'], 1):
            rating = source.get('rating', 'N/A')
            print(f"   [{i}] (평점: {rating}) {source['text'][:50]}...")

    # 6. 스트리밍 테스트
    print(f"\n{'─' * 50}")
    print("🌊 스트리밍 테스트")
    print(f"{'─' * 50}")
    print("\n❓ 질문: 이 제품을 추천하시나요?")
    print("\n📝 답변: ", end="", flush=True)

    for chunk in rag_chain.stream("이 제품을 추천하시나요?"):
        print(chunk, end="", flush=True)
    print()

    # 7. 정리
    print("\n🧹 테스트 데이터 정리")
    embedder.delete_collection()
    print("   → 테스트 컬렉션 삭제 완료")

    print("\n" + "=" * 60)
    print("✅ RAG Chain 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
