"""
에이전트 기본 클래스.

모든 에이전트가 상속받는 추상 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.agents.state import AgentState
from src.chains.rag_chain import ReviewRAGChain


class BaseAgent(ABC):
    """에이전트 추상 기본 클래스.

    모든 에이전트는 이 클래스를 상속받아 구현합니다.

    Attributes:
        rag_chain: 리뷰 RAG 체인 인스턴스
        prompt_name: 사용할 프롬프트 이름
        name: 에이전트 이름
    """

    def __init__(
        self,
        rag_chain: ReviewRAGChain,
        prompt_name: str,
        name: str | None = None,
    ):
        """초기화.

        Args:
            rag_chain: ReviewRAGChain 인스턴스
            prompt_name: 프롬프트 이름 ("qa", "summary", "compare", "sentiment")
            name: 에이전트 이름 (기본값: 클래스명)
        """
        self.rag_chain = rag_chain
        self.prompt_name = prompt_name
        self.name = name or self.__class__.__name__

    @abstractmethod
    def invoke(self, state: AgentState) -> AgentState:
        """에이전트 실행.

        Args:
            state: 현재 에이전트 상태

        Returns:
            업데이트된 상태
        """
        pass

    def _set_prompt(self) -> None:
        """프롬프트 설정."""
        self.rag_chain.set_prompt(self.prompt_name)

    def _query(self, query: str) -> dict[str, Any]:
        """RAG 쿼리 실행.

        Args:
            query: 질문

        Returns:
            RAG 응답 (answer, sources, metadata)
        """
        return self.rag_chain.query_with_sources(query)

    def _update_config(self, **kwargs) -> None:
        """RAG Chain 설정 업데이트.

        Args:
            **kwargs: 업데이트할 설정 (top_k, temperature 등)
        """
        self.rag_chain.update_config(**kwargs)

    def _create_error_state(self, state: AgentState, error_message: str) -> AgentState:
        """에러 상태 생성.

        Args:
            state: 현재 상태
            error_message: 에러 메시지

        Returns:
            에러가 설정된 상태
        """
        return {
            **state,
            "error": error_message,
            "response": f"오류가 발생했습니다: {error_message}",
            "metadata": {
                "agent": self.name,
                "error": True,
            },
        }
