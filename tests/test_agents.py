"""
에이전트 모듈 테스트.

LangGraph 기반 멀티 에이전트 시스템 테스트입니다.
"""

import pytest

from src.agents.intent_classifier import rule_based_classification
from src.agents.state import AgentState, IntentType, create_initial_state


# =============================================================================
# AgentState 테스트
# =============================================================================


class TestAgentState:
    """AgentState 테스트."""

    def test_create_initial_state(self):
        """초기 상태 생성 테스트."""
        state = create_initial_state(
            query="테스트 질문",
            product_name="테스트 제품",
        )

        assert state["query"] == "테스트 질문"
        assert state["product_name"] == "테스트 제품"
        assert state["intent"] == IntentType.UNKNOWN
        assert state["intent_confidence"] == 0.0
        assert state["response"] == ""
        assert state["error"] is None

    def test_create_initial_state_minimal(self):
        """최소 초기 상태 생성 테스트."""
        state = create_initial_state(query="간단한 질문")

        assert state["query"] == "간단한 질문"
        assert state["product_name"] is None
        assert state["product_names"] is None

    def test_create_initial_state_with_product_names(self):
        """비교용 초기 상태 생성 테스트."""
        state = create_initial_state(
            query="비교해줘",
            product_names=["제품A", "제품B"],
        )

        assert state["product_names"] == ["제품A", "제품B"]


# =============================================================================
# Intent Classifier 테스트
# =============================================================================


class TestIntentClassifier:
    """Intent Classifier 테스트."""

    def test_rule_based_summary_intent(self):
        """규칙 기반 요약 의도 분류 테스트."""
        test_cases = [
            "리뷰 요약해줘",
            "전체 리뷰 정리해줘",
            "종합적으로 분석해줘",
            "전반적인 평가는?",
        ]

        for query in test_cases:
            intent, confidence = rule_based_classification(query)
            assert intent == IntentType.SUMMARY, f"Failed for: {query}"
            assert confidence >= 0.8

    def test_rule_based_compare_intent(self):
        """규칙 기반 비교 의도 분류 테스트."""
        test_cases = [
            "두 제품 비교해줘",
            "A vs B 뭐가 나아?",
            "어떤 게 더 좋아?",
            "선택 고민 중이야",
        ]

        for query in test_cases:
            intent, confidence = rule_based_classification(query)
            assert intent == IntentType.COMPARE, f"Failed for: {query}"
            assert confidence >= 0.8

    def test_rule_based_unknown_intent(self):
        """규칙 기반 미분류 테스트."""
        test_cases = [
            "배송은 빠른가요?",
            "품질이 좋아요?",
            "가격이 적당해요?",
        ]

        for query in test_cases:
            intent, confidence = rule_based_classification(query)
            assert intent is None, f"Should be None for: {query}"
            assert confidence == 0.0


# =============================================================================
# IntentType Enum 테스트
# =============================================================================


class TestIntentType:
    """IntentType Enum 테스트."""

    def test_intent_type_values(self):
        """의도 유형 값 테스트."""
        assert IntentType.QA.value == "qa"
        assert IntentType.SUMMARY.value == "summary"
        assert IntentType.COMPARE.value == "compare"
        assert IntentType.UNKNOWN.value == "unknown"

    def test_intent_type_from_string(self):
        """문자열로부터 의도 유형 생성 테스트."""
        assert IntentType("qa") == IntentType.QA
        assert IntentType("summary") == IntentType.SUMMARY
        assert IntentType("compare") == IntentType.COMPARE


# =============================================================================
# 통합 테스트 (RAG Chain 필요)
# =============================================================================


class TestAgentIntegration:
    """에이전트 통합 테스트.

    이 테스트들은 실제 RAG Chain과 OpenAI API가 필요합니다.
    CI/CD에서는 skip될 수 있습니다.
    """

    @pytest.mark.skip(reason="OpenAI API 필요")
    def test_full_graph_qa(self):
        """전체 그래프 Q&A 테스트."""
        from src.agents import create_initial_state, create_review_agent_graph
        from src.chains.rag_chain import create_rag_chain
        from src.pipeline.embedder import create_embedder

        # Setup
        embedder = create_embedder(collection_name="test_agents")
        rag_chain = create_rag_chain(embedder=embedder)
        graph = create_review_agent_graph(rag_chain)

        # Test
        state = create_initial_state(
            query="배송이 빠른가요?",
            product_name="테스트 제품",
        )
        result = graph.invoke(state)

        # Assert
        assert result["intent"] == IntentType.QA
        assert len(result["response"]) > 0

    @pytest.mark.skip(reason="OpenAI API 필요")
    def test_full_graph_summary(self):
        """전체 그래프 요약 테스트."""
        from src.agents import create_initial_state, create_review_agent_graph
        from src.chains.rag_chain import create_rag_chain
        from src.pipeline.embedder import create_embedder

        # Setup
        embedder = create_embedder(collection_name="test_agents")
        rag_chain = create_rag_chain(embedder=embedder)
        graph = create_review_agent_graph(rag_chain)

        # Test
        state = create_initial_state(
            query="리뷰 요약해줘",
            product_name="테스트 제품",
        )
        result = graph.invoke(state)

        # Assert
        assert result["intent"] == IntentType.SUMMARY
        assert len(result["response"]) > 0
