"""
Q&A Agent.

RAG 기반 질의응답 에이전트입니다.
"""

from src.agents.base import BaseAgent
from src.agents.state import AgentState, IntentType
from src.chains.rag_chain import ReviewRAGChain


class QAAgent(BaseAgent):
    """RAG 기반 질의응답 에이전트.

    사용자의 구체적인 질문에 대해 리뷰를 검색하고 답변을 생성합니다.
    """

    def __init__(self, rag_chain: ReviewRAGChain):
        """초기화.

        Args:
            rag_chain: ReviewRAGChain 인스턴스
        """
        super().__init__(rag_chain, prompt_name="qa", name="QAAgent")

    def invoke(self, state: AgentState) -> AgentState:
        """Q&A 에이전트 실행.

        Args:
            state: 현재 에이전트 상태

        Returns:
            업데이트된 상태 (response, source_documents 포함)
        """
        try:
            # 프롬프트 설정
            self._set_prompt()

            # RAG 쿼리 실행
            query = state["query"]
            result = self._query(query)

            return {
                **state,
                "response": result["answer"],
                "source_documents": result["sources"],
                "metadata": {
                    **state.get("metadata", {}),
                    "agent": self.name,
                    "intent": IntentType.QA.value,
                    "num_sources": len(result["sources"]),
                    "model": result.get("metadata", {}).get("model"),
                },
                "error": None,
            }

        except Exception as e:
            return self._create_error_state(state, str(e))


def create_qa_agent(rag_chain: ReviewRAGChain):
    """QA Agent 노드 생성.

    Args:
        rag_chain: ReviewRAGChain 인스턴스

    Returns:
        노드 함수
    """
    agent = QAAgent(rag_chain)

    def node(state: AgentState) -> AgentState:
        return agent.invoke(state)

    return node
