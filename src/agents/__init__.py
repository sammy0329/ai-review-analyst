"""
LangGraph 기반 멀티 에이전트 시스템.

리뷰 분석을 위한 에이전트 모듈을 제공합니다.

사용 예시:
    ```python
    from src.agents import create_review_agent_graph, create_initial_state, IntentType
    from src.chains.rag_chain import create_rag_chain
    from src.pipeline.embedder import create_embedder

    # RAG Chain 생성
    embedder = create_embedder(collection_name="my_reviews")
    rag_chain = create_rag_chain(embedder=embedder)

    # 에이전트 그래프 생성
    graph = create_review_agent_graph(rag_chain)

    # 질문 처리
    state = create_initial_state(
        query="이 제품의 배송은 어떤가요?",
        product_name="테스트 제품"
    )
    result = graph.invoke(state)

    print(result["intent"])      # IntentType.QA
    print(result["response"])    # AI 응답
    ```
"""

from src.agents.base import BaseAgent
from src.agents.compare_agent import CompareAgent, create_compare_agent
from src.agents.graph import create_review_agent_graph, get_graph_visualization
from src.agents.intent_classifier import (
    IntentClassification,
    classify_intent,
    create_intent_classifier_node,
)
from src.agents.qa_agent import QAAgent, create_qa_agent
from src.agents.state import AgentState, IntentType, create_initial_state
from src.agents.summarize_agent import SummarizeAgent, create_summarize_agent

__all__ = [
    # State
    "AgentState",
    "IntentType",
    "create_initial_state",
    # Base
    "BaseAgent",
    # Intent Classifier
    "IntentClassification",
    "classify_intent",
    "create_intent_classifier_node",
    # Agents
    "QAAgent",
    "create_qa_agent",
    "SummarizeAgent",
    "create_summarize_agent",
    "CompareAgent",
    "create_compare_agent",
    # Graph
    "create_review_agent_graph",
    "get_graph_visualization",
]
