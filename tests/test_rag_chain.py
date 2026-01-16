"""
RAG Chain 단위 테스트.
"""

import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# playwright 의존성 없이 테스트할 수 있도록 모킹
sys.modules["playwright"] = MagicMock()
sys.modules["playwright.async_api"] = MagicMock()
sys.modules["playwright_stealth"] = MagicMock()

from src.chains.rag_chain import (
    RAGConfig,
    RAGResponse,
    ReviewRAGChain,
    create_rag_chain,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


class TestRAGConfig:
    """RAGConfig 테스트."""

    def test_default_values(self):
        """기본값 테스트."""
        config = RAGConfig()
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 2048
        assert config.streaming is True
        assert config.top_k == 10
        assert config.min_score == 0.1
        assert config.min_results == 3
        assert config.search_type == "similarity"

    def test_custom_values(self):
        """커스텀 값 테스트."""
        config = RAGConfig(
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
            streaming=False,
            top_k=10,
            search_type="mmr",
        )
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.streaming is False
        assert config.top_k == 10
        assert config.search_type == "mmr"

    def test_custom_prompts(self):
        """커스텀 프롬프트 테스트."""
        custom_system = "커스텀 시스템 프롬프트"
        custom_user = "커스텀 유저 프롬프트: {context}\n{question}"

        config = RAGConfig(
            system_prompt=custom_system,
            user_prompt_template=custom_user,
        )
        assert config.system_prompt == custom_system
        assert config.user_prompt_template == custom_user


class TestRAGResponse:
    """RAGResponse 테스트."""

    def test_basic_response(self):
        """기본 응답 생성 테스트."""
        response = RAGResponse(answer="테스트 답변입니다.")
        assert response.answer == "테스트 답변입니다."
        assert response.source_documents == []
        assert response.metadata == {}

    def test_response_with_sources(self):
        """출처가 있는 응답 테스트."""
        mock_doc = MagicMock()
        mock_doc.page_content = "리뷰 내용"
        mock_doc.metadata = {"rating": 5.0}

        response = RAGResponse(
            answer="테스트 답변",
            source_documents=[mock_doc],
            metadata={"model": "gpt-4o-mini", "top_k": 5},
        )
        assert len(response.source_documents) == 1
        assert response.metadata["model"] == "gpt-4o-mini"


class TestPrompts:
    """프롬프트 템플릿 테스트."""

    def test_system_prompt_exists(self):
        """시스템 프롬프트 존재 확인."""
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 0
        assert "리뷰" in SYSTEM_PROMPT

    def test_user_prompt_template_placeholders(self):
        """유저 프롬프트 템플릿 플레이스홀더 확인."""
        assert "{context}" in USER_PROMPT_TEMPLATE
        assert "{question}" in USER_PROMPT_TEMPLATE

    def test_system_prompt_principles(self):
        """시스템 프롬프트 원칙 포함 확인."""
        assert "근거" in SYSTEM_PROMPT or "객관" in SYSTEM_PROMPT
        assert "답변" in SYSTEM_PROMPT


class TestReviewRAGChainInit:
    """ReviewRAGChain 초기화 테스트."""

    def test_missing_api_key_raises_error(self):
        """API 키 없을 때 에러 발생 테스트."""
        with patch.dict("os.environ", {}, clear=True):
            # 환경변수에서 OPENAI_API_KEY 제거
            with patch("os.getenv", return_value=None):
                with pytest.raises(ValueError, match="OpenAI API 키가 필요합니다"):
                    ReviewRAGChain()

    @patch("src.chains.rag_chain.ChatOpenAI")
    @patch("src.chains.rag_chain.create_embedder")
    def test_init_with_api_key(self, mock_create_embedder, mock_chat_openai):
        """API 키로 초기화 테스트."""
        mock_embedder = MagicMock()
        mock_embedder.get_retriever.return_value = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        chain = ReviewRAGChain(openai_api_key="test-api-key")
        assert chain.config.model_name == "gpt-4o-mini"
        mock_chat_openai.assert_called_once()

    @patch("src.chains.rag_chain.ChatOpenAI")
    @patch("src.chains.rag_chain.create_embedder")
    def test_init_with_custom_config(self, mock_create_embedder, mock_chat_openai):
        """커스텀 설정으로 초기화 테스트."""
        mock_embedder = MagicMock()
        mock_embedder.get_retriever.return_value = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        config = RAGConfig(
            model_name="gpt-4o",
            temperature=0.5,
            top_k=10,
        )
        chain = ReviewRAGChain(config=config, openai_api_key="test-api-key")

        assert chain.config.model_name == "gpt-4o"
        assert chain.config.temperature == 0.5
        assert chain.config.top_k == 10

    @patch("src.chains.rag_chain.ChatOpenAI")
    def test_init_with_existing_embedder(self, mock_chat_openai):
        """기존 Embedder로 초기화 테스트."""
        mock_embedder = MagicMock()
        mock_embedder.get_retriever.return_value = MagicMock()

        chain = ReviewRAGChain(
            embedder=mock_embedder,
            openai_api_key="test-api-key",
        )
        assert chain.embedder == mock_embedder


class TestReviewRAGChainMethods:
    """ReviewRAGChain 메서드 테스트."""

    @pytest.fixture
    def mock_chain(self):
        """모킹된 RAG Chain 픽스처."""
        with patch("src.chains.rag_chain.ChatOpenAI") as mock_llm:
            with patch("src.chains.rag_chain.create_embedder") as mock_create:
                mock_embedder = MagicMock()
                mock_create.return_value = mock_embedder

                chain = ReviewRAGChain(openai_api_key="test-key")
                chain._chain = MagicMock()

                yield chain

    def test_query_returns_rag_response(self, mock_chain):
        """query 메서드가 RAGResponse 반환 테스트."""
        mock_doc = MagicMock()
        mock_doc.page_content = "좋은 제품입니다"
        mock_doc.metadata = {"rating": 5.0, "date": "2024-01-15"}

        # _retrieve_filtered 메서드 모킹
        mock_chain._retrieve_filtered = MagicMock(return_value=[mock_doc])
        mock_chain._chain.invoke.return_value = "배송이 빠릅니다."

        response = mock_chain.query("배송이 빠른가요?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "배송이 빠릅니다."
        assert len(response.source_documents) == 1
        assert response.metadata["num_sources"] == 1

    def test_query_with_sources(self, mock_chain):
        """query_with_sources 메서드 테스트."""
        mock_doc = MagicMock()
        mock_doc.page_content = "품질이 좋습니다"
        mock_doc.metadata = {
            "rating": 4.5,
            "date": "2024-01-10",
            "review_hash": "abc123",
        }

        # _retrieve_filtered 메서드 모킹
        mock_chain._retrieve_filtered = MagicMock(return_value=[mock_doc])
        mock_chain._chain.invoke.return_value = "품질에 대해 긍정적입니다."

        result = mock_chain.query_with_sources("품질은 어떤가요?")

        assert "answer" in result
        assert "sources" in result
        assert "metadata" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["text"] == "품질이 좋습니다"
        assert result["sources"][0]["rating"] == 4.5

    def test_stream_yields_chunks(self, mock_chain):
        """stream 메서드가 청크를 yield하는지 테스트."""
        mock_chain._chain.stream.return_value = iter(["안녕", "하세요", "!"])

        chunks = list(mock_chain.stream("테스트 질문"))

        assert chunks == ["안녕", "하세요", "!"]

    def test_update_config_llm_settings(self, mock_chain):
        """update_config로 LLM 설정 변경 테스트."""
        with patch("src.chains.rag_chain.ChatOpenAI") as mock_llm:
            mock_chain.update_config(
                model_name="gpt-4o",
                temperature=0.8,
            )

            assert mock_chain.config.model_name == "gpt-4o"
            assert mock_chain.config.temperature == 0.8
            mock_llm.assert_called()

    def test_update_config_retriever_settings(self, mock_chain):
        """update_config로 Retriever 설정 변경 테스트."""
        mock_chain.embedder.get_retriever.return_value = MagicMock()

        mock_chain.update_config(top_k=10, search_type="mmr")

        assert mock_chain.config.top_k == 10
        assert mock_chain.config.search_type == "mmr"

    def test_llm_property(self, mock_chain):
        """llm 프로퍼티 테스트."""
        assert mock_chain.llm is not None


class TestCreateRagChain:
    """create_rag_chain 헬퍼 함수 테스트."""

    @patch("src.chains.rag_chain.ChatOpenAI")
    @patch("src.chains.rag_chain.create_embedder")
    def test_create_with_defaults(self, mock_create_embedder, mock_chat_openai):
        """기본값으로 생성 테스트."""
        mock_embedder = MagicMock()
        mock_embedder.get_retriever.return_value = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        chain = create_rag_chain(openai_api_key="test-key")

        assert isinstance(chain, ReviewRAGChain)
        assert chain.config.model_name == "gpt-4o-mini"
        assert chain.config.temperature == 0.0
        assert chain.config.top_k == 5

    @patch("src.chains.rag_chain.ChatOpenAI")
    @patch("src.chains.rag_chain.create_embedder")
    def test_create_with_custom_params(self, mock_create_embedder, mock_chat_openai):
        """커스텀 파라미터로 생성 테스트."""
        mock_embedder = MagicMock()
        mock_embedder.get_retriever.return_value = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        chain = create_rag_chain(
            model_name="gpt-4o",
            temperature=0.7,
            top_k=10,
            streaming=False,
            openai_api_key="test-key",
        )

        assert chain.config.model_name == "gpt-4o"
        assert chain.config.temperature == 0.7
        assert chain.config.top_k == 10
        assert chain.config.streaming is False

    @patch("src.chains.rag_chain.ChatOpenAI")
    def test_create_with_existing_embedder(self, mock_chat_openai):
        """기존 Embedder로 생성 테스트."""
        mock_embedder = MagicMock()
        mock_embedder.get_retriever.return_value = MagicMock()

        chain = create_rag_chain(
            embedder=mock_embedder,
            openai_api_key="test-key",
        )

        assert chain.embedder == mock_embedder


class TestDocumentFormatting:
    """문서 포맷팅 테스트."""

    @patch("src.chains.rag_chain.ChatOpenAI")
    @patch("src.chains.rag_chain.create_embedder")
    def test_format_docs_in_chain(self, mock_create_embedder, mock_chat_openai):
        """Chain 내 문서 포맷팅 테스트."""
        mock_embedder = MagicMock()
        mock_embedder.get_retriever.return_value = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        chain = ReviewRAGChain(openai_api_key="test-key")

        # _build_chain 내부의 format_docs 함수 테스트
        # Chain이 정상적으로 구성되었는지 확인
        assert chain._chain is not None
