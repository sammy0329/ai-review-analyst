"""
에이전트 모듈 테스트.

LangGraph 기반 멀티 에이전트 시스템 테스트입니다.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import BaseAgent
from src.agents.compare_agent import CompareAgent, create_compare_agent
from src.agents.graph import get_graph_visualization, route_by_intent
from src.agents.intent_classifier import (
    IntentClassification,
    classify_intent,
    create_intent_classifier_node,
    llm_based_classification,
    rule_based_classification,
)
from src.agents.qa_agent import QAAgent, create_qa_agent
from src.agents.state import AgentState, IntentType, create_initial_state
from src.agents.summarize_agent import SummarizeAgent, create_summarize_agent


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

    def test_initial_state_has_empty_source_documents(self):
        """초기 상태에 빈 source_documents 확인."""
        state = create_initial_state(query="질문")
        assert state["source_documents"] == []

    def test_initial_state_has_empty_metadata(self):
        """초기 상태에 빈 metadata 확인."""
        state = create_initial_state(query="질문")
        assert state["metadata"] == {}

    def test_initial_state_context_is_none(self):
        """초기 상태에 context가 None인지 확인."""
        state = create_initial_state(query="질문")
        assert state["context"] is None


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
            "총평 알려줘",
            "개요를 알려줘",
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
            "둘 중 뭐가 좋아?",
            "차이점이 뭐야?",
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

    def test_classify_intent_with_rule_based(self):
        """classify_intent 함수 - 규칙 기반 분류."""
        state = create_initial_state(query="요약해줘", product_name="테스트 제품")
        result = classify_intent(state)

        assert result["intent"] == IntentType.SUMMARY
        assert result["intent_confidence"] >= 0.8
        assert result["metadata"]["classification_method"] == "rule_based"

    @patch("src.agents.intent_classifier.llm_based_classification")
    def test_classify_intent_with_llm_fallback(self, mock_llm_classification):
        """classify_intent 함수 - LLM 폴백."""
        mock_llm_classification.return_value = (IntentType.QA, 0.85)

        state = create_initial_state(query="배송은 어때요?", product_name="테스트 제품")
        result = classify_intent(state)

        assert result["intent"] == IntentType.QA
        assert result["metadata"]["classification_method"] == "llm_based"

    def test_create_intent_classifier_node(self):
        """create_intent_classifier_node 함수 테스트."""
        node = create_intent_classifier_node()
        assert callable(node)

        # 노드 실행 테스트 (규칙 기반 분류)
        state = create_initial_state(query="리뷰 요약해줘")
        result = node(state)
        assert result["intent"] == IntentType.SUMMARY


class TestLLMBasedClassification:
    """LLM 기반 분류 테스트."""

    @patch("src.agents.intent_classifier.ChatOpenAI")
    def test_llm_classification_qa(self, mock_chat_openai):
        """LLM 기반 QA 분류 테스트."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = IntentClassification(
            intent="qa", confidence=0.9, reasoning="품질에 대한 질문"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_openai.return_value = mock_llm

        intent, confidence = llm_based_classification("품질은 어때요?", "테스트 제품")

        assert intent == IntentType.QA
        assert confidence == 0.9

    @patch("src.agents.intent_classifier.ChatOpenAI")
    def test_llm_classification_summary(self, mock_chat_openai):
        """LLM 기반 Summary 분류 테스트."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = IntentClassification(
            intent="summary", confidence=0.95, reasoning="요약 요청"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_openai.return_value = mock_llm

        intent, confidence = llm_based_classification("전체적인 평가 알려줘", "테스트 제품")

        assert intent == IntentType.SUMMARY
        assert confidence == 0.95

    @patch("src.agents.intent_classifier.ChatOpenAI")
    def test_llm_classification_compare(self, mock_chat_openai):
        """LLM 기반 Compare 분류 테스트."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = IntentClassification(
            intent="compare", confidence=0.88, reasoning="비교 요청"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_openai.return_value = mock_llm

        intent, confidence = llm_based_classification("비교해줘", "테스트 제품")

        assert intent == IntentType.COMPARE
        assert confidence == 0.88

    @patch("src.agents.intent_classifier.ChatOpenAI")
    def test_llm_classification_unknown_intent(self, mock_chat_openai):
        """LLM 기반 Unknown 분류 테스트."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = IntentClassification(
            intent="unknown_type", confidence=0.5, reasoning="알 수 없음"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_openai.return_value = mock_llm

        intent, confidence = llm_based_classification("???", None)

        assert intent == IntentType.UNKNOWN
        assert confidence == 0.5

    @patch("src.agents.intent_classifier.ChatOpenAI")
    def test_llm_classification_exception_fallback(self, mock_chat_openai):
        """LLM 호출 실패 시 기본값 반환."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("API Error")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_openai.return_value = mock_llm

        intent, confidence = llm_based_classification("테스트", "제품")

        assert intent == IntentType.QA
        assert confidence == 0.5


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

    def test_intent_type_is_string_enum(self):
        """IntentType이 str Enum인지 확인."""
        assert isinstance(IntentType.QA, str)
        assert IntentType.QA == "qa"


# =============================================================================
# BaseAgent 테스트
# =============================================================================


class TestBaseAgent:
    """BaseAgent 테스트."""

    def test_base_agent_is_abstract(self):
        """BaseAgent가 추상 클래스인지 확인."""
        mock_rag_chain = MagicMock()

        with pytest.raises(TypeError):
            BaseAgent(mock_rag_chain, "qa")

    def test_create_error_state(self):
        """에러 상태 생성 테스트."""

        class ConcreteAgent(BaseAgent):
            def invoke(self, state):
                return state

        mock_rag_chain = MagicMock()
        agent = ConcreteAgent(mock_rag_chain, "qa", "TestAgent")

        state = create_initial_state(query="테스트")
        error_state = agent._create_error_state(state, "테스트 에러")

        assert error_state["error"] == "테스트 에러"
        assert "오류가 발생했습니다" in error_state["response"]
        assert error_state["metadata"]["agent"] == "TestAgent"
        assert error_state["metadata"]["error"] is True


# =============================================================================
# QAAgent 테스트
# =============================================================================


class TestQAAgent:
    """QAAgent 테스트."""

    def test_qa_agent_invoke(self):
        """QAAgent invoke 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.return_value = {
            "answer": "배송은 매우 빠릅니다.",
            "sources": [{"text": "리뷰1", "rating": 5}],
            "metadata": {"model": "gpt-4o-mini"},
        }

        agent = QAAgent(mock_rag_chain)
        state = create_initial_state(query="배송은 빠른가요?", product_name="테스트 제품")
        result = agent.invoke(state)

        assert result["response"] == "배송은 매우 빠릅니다."
        assert len(result["source_documents"]) == 1
        assert result["metadata"]["agent"] == "QAAgent"
        assert result["error"] is None

    def test_qa_agent_error_handling(self):
        """QAAgent 에러 처리 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.side_effect = Exception("API 오류")

        agent = QAAgent(mock_rag_chain)
        state = create_initial_state(query="테스트 질문")
        result = agent.invoke(state)

        assert result["error"] == "API 오류"
        assert "오류가 발생했습니다" in result["response"]

    def test_create_qa_agent_node(self):
        """create_qa_agent 함수 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.return_value = {
            "answer": "답변",
            "sources": [],
            "metadata": {},
        }

        node = create_qa_agent(mock_rag_chain)
        assert callable(node)

        state = create_initial_state(query="테스트")
        result = node(state)
        assert result["response"] == "답변"


# =============================================================================
# SummarizeAgent 테스트
# =============================================================================


class TestSummarizeAgent:
    """SummarizeAgent 테스트."""

    def test_summarize_agent_invoke(self):
        """SummarizeAgent invoke 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.return_value = {
            "answer": "전체적으로 긍정적인 평가입니다.",
            "sources": [{"text": "리뷰1"}, {"text": "리뷰2"}],
            "metadata": {},
        }

        agent = SummarizeAgent(mock_rag_chain, top_k=10)
        state = create_initial_state(query="리뷰 요약해줘", product_name="테스트 제품")
        result = agent.invoke(state)

        assert "긍정적인 평가" in result["response"]
        assert result["metadata"]["agent"] == "SummarizeAgent"
        assert result["metadata"]["top_k"] == 10
        assert result["error"] is None

    def test_summarize_agent_restores_top_k(self):
        """SummarizeAgent가 top_k를 복원하는지 확인."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.return_value = {
            "answer": "요약 결과",
            "sources": [],
            "metadata": {},
        }

        agent = SummarizeAgent(mock_rag_chain, top_k=15)
        state = create_initial_state(query="요약해줘")
        agent.invoke(state)

        # update_config가 두 번 호출됨 (top_k=15, top_k=5)
        calls = mock_rag_chain.update_config.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["top_k"] == 15
        assert calls[1].kwargs["top_k"] == 5

    def test_summarize_agent_error_handling(self):
        """SummarizeAgent 에러 처리 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.side_effect = Exception("요약 실패")

        agent = SummarizeAgent(mock_rag_chain)
        state = create_initial_state(query="요약해줘")
        result = agent.invoke(state)

        assert result["error"] == "요약 실패"

    def test_create_summarize_agent_node(self):
        """create_summarize_agent 함수 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.return_value = {
            "answer": "요약",
            "sources": [],
            "metadata": {},
        }

        node = create_summarize_agent(mock_rag_chain, top_k=8)
        assert callable(node)


# =============================================================================
# CompareAgent 테스트
# =============================================================================


class TestCompareAgent:
    """CompareAgent 테스트."""

    def test_compare_agent_single_product(self):
        """CompareAgent 단일 제품 (장단점 분석) 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.return_value = {
            "answer": "장점: 배송 빠름, 단점: 가격 비쌈",
            "sources": [{"text": "리뷰"}],
            "metadata": {},
        }
        mock_llm = MagicMock()

        agent = CompareAgent(mock_rag_chain, llm=mock_llm)
        state = create_initial_state(query="장단점 분석해줘", product_name="테스트 제품")
        result = agent.invoke(state)

        assert "장점" in result["response"]
        assert result["metadata"]["comparison_type"] == "single_product"
        assert result["error"] is None

    def test_compare_agent_multi_product(self):
        """CompareAgent 다중 제품 비교 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.return_value = {
            "answer": "제품A가 제품B보다 좋습니다.",
            "sources": [{"text": "리뷰1"}, {"text": "리뷰2"}],
            "metadata": {},
        }
        mock_llm = MagicMock()

        agent = CompareAgent(mock_rag_chain, llm=mock_llm)
        state = create_initial_state(
            query="비교해줘",
            product_names=["제품A", "제품B"],
        )
        result = agent.invoke(state)

        assert result["metadata"]["comparison_type"] == "multi_product"
        assert result["metadata"]["compared_products"] == ["제품A", "제품B"]

    def test_compare_agent_error_handling(self):
        """CompareAgent 에러 처리 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.side_effect = Exception("비교 실패")
        mock_llm = MagicMock()

        agent = CompareAgent(mock_rag_chain, llm=mock_llm)
        state = create_initial_state(query="비교해줘", product_names=["A", "B"])
        result = agent.invoke(state)

        assert result["error"] == "비교 실패"

    def test_compare_products_method(self):
        """compare_products 메서드 테스트."""
        mock_rag_chain = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="비교 분석 결과")

        agent = CompareAgent(mock_rag_chain, llm=mock_llm)
        product_reviews = {
            "제품A": [{"text": "좋아요", "rating": 5}],
            "제품B": [{"text": "별로예요", "rating": 2}],
        }

        result = agent.compare_products(product_reviews)
        assert result == "비교 분석 결과"
        mock_llm.invoke.assert_called_once()

    def test_create_compare_agent_node(self):
        """create_compare_agent 함수 테스트."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.query_with_sources.return_value = {
            "answer": "비교 결과",
            "sources": [],
            "metadata": {},
        }
        mock_llm = MagicMock()

        node = create_compare_agent(mock_rag_chain, llm=mock_llm)
        assert callable(node)


# =============================================================================
# Graph 테스트
# =============================================================================


class TestGraph:
    """Graph 모듈 테스트."""

    def test_route_by_intent_qa(self):
        """route_by_intent - QA 라우팅."""
        state: AgentState = {
            "query": "테스트",
            "intent": IntentType.QA,
        }
        assert route_by_intent(state) == "qa"

    def test_route_by_intent_summary(self):
        """route_by_intent - Summary 라우팅."""
        state: AgentState = {
            "query": "테스트",
            "intent": IntentType.SUMMARY,
        }
        assert route_by_intent(state) == "summary"

    def test_route_by_intent_compare(self):
        """route_by_intent - Compare 라우팅."""
        state: AgentState = {
            "query": "테스트",
            "intent": IntentType.COMPARE,
        }
        assert route_by_intent(state) == "compare"

    def test_route_by_intent_unknown_defaults_to_qa(self):
        """route_by_intent - UNKNOWN은 QA로 기본 라우팅."""
        state: AgentState = {
            "query": "테스트",
            "intent": IntentType.UNKNOWN,
        }
        assert route_by_intent(state) == "qa"

    def test_route_by_intent_missing_defaults_to_qa(self):
        """route_by_intent - intent 없으면 QA로 기본 라우팅."""
        state: AgentState = {"query": "테스트"}
        assert route_by_intent(state) == "qa"

    def test_get_graph_visualization(self):
        """get_graph_visualization 함수 테스트."""
        mock_graph = MagicMock()
        mock_graph.get_graph.return_value.draw_mermaid.return_value = "graph TD"

        result = get_graph_visualization(mock_graph)
        assert "graph TD" in result

    def test_get_graph_visualization_fallback(self):
        """get_graph_visualization 예외 시 기본 텍스트 반환."""
        mock_graph = MagicMock()
        mock_graph.get_graph.side_effect = Exception("Mermaid 오류")

        result = get_graph_visualization(mock_graph)
        assert "classify_intent" in result
        assert "qa" in result
        assert "summary" in result
        assert "compare" in result


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
