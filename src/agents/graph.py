"""
LangGraph StateGraph 구성.

리뷰 분석 멀티 에이전트 그래프를 정의합니다.
"""

from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.compare_agent import create_compare_agent
from src.agents.intent_classifier import create_intent_classifier_node
from src.agents.qa_agent import create_qa_agent
from src.agents.state import AgentState, IntentType
from src.agents.summarize_agent import create_summarize_agent
from src.chains.rag_chain import ReviewRAGChain


def route_by_intent(state: AgentState) -> Literal["qa", "summary", "compare"]:
    """의도에 따른 라우팅.

    Args:
        state: 현재 에이전트 상태

    Returns:
        다음 노드 이름
    """
    intent = state.get("intent", IntentType.UNKNOWN)

    if intent == IntentType.SUMMARY:
        return "summary"
    elif intent == IntentType.COMPARE:
        return "compare"
    else:
        # QA 또는 UNKNOWN은 qa_agent로 라우팅
        return "qa"


def create_review_agent_graph(
    rag_chain: ReviewRAGChain,
    llm: ChatOpenAI | None = None,
) -> StateGraph:
    """리뷰 분석 에이전트 그래프 생성.

    그래프 구조:
    ```
    START → Intent Classifier → Router → [QA|Summary|Compare] → END
    ```

    Args:
        rag_chain: ReviewRAGChain 인스턴스
        llm: ChatOpenAI 인스턴스 (Intent Classifier용)

    Returns:
        컴파일된 StateGraph
    """
    # 그래프 생성
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("classify_intent", create_intent_classifier_node(llm))
    graph.add_node("qa", create_qa_agent(rag_chain))
    graph.add_node("summary", create_summarize_agent(rag_chain))
    graph.add_node("compare", create_compare_agent(rag_chain, llm))

    # 엣지: START -> classify_intent
    graph.set_entry_point("classify_intent")

    # 조건부 엣지: intent에 따른 분기
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "qa": "qa",
            "summary": "summary",
            "compare": "compare",
        },
    )

    # 각 에이전트 -> END
    graph.add_edge("qa", END)
    graph.add_edge("summary", END)
    graph.add_edge("compare", END)

    # 그래프 컴파일
    return graph.compile()


def get_graph_visualization(graph: StateGraph) -> str:
    """그래프 시각화 (Mermaid 형식).

    Args:
        graph: StateGraph 인스턴스

    Returns:
        Mermaid 다이어그램 문자열
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception:
        # Mermaid 출력 불가 시 기본 텍스트 반환
        return """
graph TD
    START --> classify_intent
    classify_intent -->|qa| qa
    classify_intent -->|summary| summary
    classify_intent -->|compare| compare
    qa --> END
    summary --> END
    compare --> END
"""
