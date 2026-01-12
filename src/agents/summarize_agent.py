"""
Summarize Agent.

리뷰 요약 에이전트입니다.
"""

from src.agents.base import BaseAgent
from src.agents.state import AgentState, IntentType
from src.chains.rag_chain import ReviewRAGChain


class SummarizeAgent(BaseAgent):
    """리뷰 요약 에이전트.

    제품의 전체 리뷰를 종합하여 요약합니다.
    더 많은 컨텍스트(top_k=10)를 사용하여 포괄적인 요약을 제공합니다.
    """

    def __init__(self, rag_chain: ReviewRAGChain, top_k: int = 10):
        """초기화.

        Args:
            rag_chain: ReviewRAGChain 인스턴스
            top_k: 검색할 리뷰 수 (기본값: 10)
        """
        super().__init__(rag_chain, prompt_name="summary", name="SummarizeAgent")
        self.top_k = top_k

    def invoke(self, state: AgentState) -> AgentState:
        """요약 에이전트 실행.

        Args:
            state: 현재 에이전트 상태

        Returns:
            업데이트된 상태 (response, source_documents 포함)
        """
        try:
            # QA 프롬프트 사용 (summary 프롬프트는 review_count 변수 필요)
            # 대신 요약 지시를 쿼리에 포함
            self.rag_chain.set_prompt("qa")

            # 더 많은 리뷰 검색을 위해 top_k 증가
            self._update_config(top_k=self.top_k)

            # 요약용 쿼리 구성
            product_name = state.get("product_name", "이 제품")
            base_query = state.get("query") or ""

            # 요약 지시 추가
            query = f"""다음 요청에 대해 리뷰를 종합적으로 분석해주세요.

요청: {base_query if base_query else f"'{product_name}'의 리뷰를 요약해주세요."}

다음 형식으로 답변해주세요:
1. 전체 평가 (긍정/부정 비율)
2. 주요 장점
3. 주요 단점
4. 추천 여부"""

            # RAG 쿼리 실행
            result = self._query(query)

            return {
                **state,
                "response": result["answer"],
                "source_documents": result["sources"],
                "metadata": {
                    **state.get("metadata", {}),
                    "agent": self.name,
                    "intent": IntentType.SUMMARY.value,
                    "num_sources": len(result["sources"]),
                    "top_k": self.top_k,
                },
                "error": None,
            }

        except Exception as e:
            return self._create_error_state(state, str(e))

        finally:
            # top_k 원래대로 복원
            self._update_config(top_k=5)


def create_summarize_agent(rag_chain: ReviewRAGChain, top_k: int = 10):
    """Summarize Agent 노드 생성.

    Args:
        rag_chain: ReviewRAGChain 인스턴스
        top_k: 검색할 리뷰 수

    Returns:
        노드 함수
    """
    agent = SummarizeAgent(rag_chain, top_k)

    def node(state: AgentState) -> AgentState:
        return agent.invoke(state)

    return node
