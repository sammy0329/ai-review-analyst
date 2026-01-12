"""
Compare Agent.

제품 비교 에이전트입니다.
"""

from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.base import BaseAgent
from src.agents.state import AgentState, IntentType
from src.chains.rag_chain import ReviewRAGChain


COMPARE_SYSTEM_PROMPT = """당신은 상품 비교 분석 전문가입니다.
제공된 각 제품의 리뷰 정보를 바탕으로 객관적인 비교 분석을 제공합니다.

## 분석 원칙
1. **동일 기준 적용**: 모든 제품에 동일한 평가 기준을 적용
2. **수치 기반**: 구체적인 수치나 비율 사용
3. **장단점 명시**: 각 제품의 장단점을 명확히 구분
4. **근거 기반**: 리뷰 내용을 근거로 분석

## 출력 형식
1. 전체 비교 요약
2. 항목별 비교 (품질, 가격, 배송 등)
3. 각 제품 추천 상황"""


class CompareAgent(BaseAgent):
    """제품 비교 에이전트.

    여러 제품의 리뷰를 비교 분석합니다.
    """

    def __init__(
        self,
        rag_chain: ReviewRAGChain,
        llm: ChatOpenAI | None = None,
    ):
        """초기화.

        Args:
            rag_chain: 기본 ReviewRAGChain 인스턴스
            llm: 비교 분석용 LLM (None이면 새로 생성)
        """
        super().__init__(rag_chain, prompt_name="compare", name="CompareAgent")
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def invoke(self, state: AgentState) -> AgentState:
        """비교 에이전트 실행.

        단일 제품에 대한 비교 요청인 경우 장단점 분석으로 처리합니다.

        Args:
            state: 현재 에이전트 상태

        Returns:
            업데이트된 상태
        """
        try:
            # QA 프롬프트 사용 (compare 프롬프트는 product_a/b_reviews 변수 필요)
            self.rag_chain.set_prompt("qa")

            # 비교할 제품 목록 확인
            product_names = state.get("product_names") or []
            product_name = state.get("product_name")

            # 단일 제품인 경우 장단점 분석
            if len(product_names) < 2:
                base_query = state.get("query") or ""
                query = f"""다음 요청에 대해 리뷰를 분석해주세요.

요청: {base_query if base_query else f"'{product_name}'의 장단점을 분석해주세요."}

다음 형식으로 답변해주세요:
1. 주요 장점 (리뷰 근거 포함)
2. 주요 단점 (리뷰 근거 포함)
3. 종합 평가"""

                result = self._query(query)

                return {
                    **state,
                    "response": result["answer"],
                    "source_documents": result["sources"],
                    "metadata": {
                        **state.get("metadata", {}),
                        "agent": self.name,
                        "intent": IntentType.COMPARE.value,
                        "comparison_type": "single_product",
                        "num_sources": len(result["sources"]),
                    },
                    "error": None,
                }

            # 다중 제품 비교 (현재는 단일 RAG Chain만 지원)
            # 향후 확장: 제품별 RAG Chain 관리
            query = state.get("query") or "이 제품들을 비교 분석해주세요."
            result = self._query(query)

            return {
                **state,
                "response": result["answer"],
                "source_documents": result["sources"],
                "metadata": {
                    **state.get("metadata", {}),
                    "agent": self.name,
                    "intent": IntentType.COMPARE.value,
                    "comparison_type": "multi_product",
                    "compared_products": product_names,
                    "num_sources": len(result["sources"]),
                },
                "error": None,
            }

        except Exception as e:
            return self._create_error_state(state, str(e))

    def compare_products(
        self,
        product_reviews: dict[str, list[dict[str, Any]]],
    ) -> str:
        """여러 제품의 리뷰를 비교 분석.

        Args:
            product_reviews: {제품명: 리뷰 목록} 딕셔너리

        Returns:
            비교 분석 결과
        """
        # 각 제품의 리뷰를 포맷팅
        formatted_reviews = []
        for product_name, reviews in product_reviews.items():
            review_texts = "\n".join([
                f"- (평점: {r.get('rating', 'N/A')}) {r.get('text', '')[:200]}"
                for r in reviews[:5]
            ])
            formatted_reviews.append(f"## {product_name}\n{review_texts}")

        context = "\n\n".join(formatted_reviews)

        # 비교 분석 프롬프트
        prompt = f"""{COMPARE_SYSTEM_PROMPT}

## 제품별 리뷰
{context}

위 리뷰들을 바탕으로 제품들을 비교 분석해주세요."""

        response = self.llm.invoke(prompt)
        return response.content


def create_compare_agent(rag_chain: ReviewRAGChain, llm: ChatOpenAI | None = None):
    """Compare Agent 노드 생성.

    Args:
        rag_chain: ReviewRAGChain 인스턴스
        llm: ChatOpenAI 인스턴스 (선택)

    Returns:
        노드 함수
    """
    agent = CompareAgent(rag_chain, llm)

    def node(state: AgentState) -> AgentState:
        return agent.invoke(state)

    return node
