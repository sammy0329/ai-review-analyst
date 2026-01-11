"""
프롬프트 템플릿 단위 테스트.
"""

import pytest

from src.prompts.templates import (
    PromptTemplate,
    PromptType,
    PromptManager,
    QA_PROMPT,
    SUMMARY_PROMPT,
    COMPARE_PROMPT,
    SENTIMENT_PROMPT,
    get_prompt,
    list_prompts,
    ANTI_HALLUCINATION_INSTRUCTIONS,
)


class TestPromptTemplate:
    """PromptTemplate 테스트."""

    def test_create_template(self):
        """템플릿 생성 테스트."""
        template = PromptTemplate(
            name="test",
            type=PromptType.QA,
            system_prompt="시스템 프롬프트",
            user_prompt_template="질문: {question}",
            description="테스트용",
            version="1.0",
        )
        assert template.name == "test"
        assert template.type == PromptType.QA
        assert template.version == "1.0"

    def test_format_user_prompt(self):
        """사용자 프롬프트 포맷팅 테스트."""
        template = PromptTemplate(
            name="test",
            type=PromptType.QA,
            system_prompt="시스템",
            user_prompt_template="컨텍스트: {context}\n질문: {question}",
        )
        result = template.format_user_prompt(
            context="리뷰 내용",
            question="배송이 빠른가요?",
        )
        assert "리뷰 내용" in result
        assert "배송이 빠른가요?" in result

    def test_get_messages_without_few_shot(self):
        """Few-shot 없는 메시지 생성 테스트."""
        template = PromptTemplate(
            name="test",
            type=PromptType.QA,
            system_prompt="시스템 프롬프트",
            user_prompt_template="질문: {question}",
        )
        messages = template.get_messages(question="테스트 질문")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_get_messages_with_few_shot(self):
        """Few-shot 포함 메시지 생성 테스트."""
        template = PromptTemplate(
            name="test",
            type=PromptType.QA,
            system_prompt="시스템 프롬프트",
            user_prompt_template="질문: {question}",
            few_shot_examples=[
                {"user": "예시 질문", "assistant": "예시 답변"},
            ],
        )
        messages = template.get_messages(question="테스트 질문")

        assert len(messages) == 4  # system + few_shot_user + few_shot_assistant + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "예시 질문"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"


class TestQAPrompt:
    """Q&A 프롬프트 테스트."""

    def test_qa_prompt_exists(self):
        """QA 프롬프트 존재 확인."""
        assert QA_PROMPT is not None
        assert QA_PROMPT.name == "qa"
        assert QA_PROMPT.type == PromptType.QA

    def test_qa_prompt_system_content(self):
        """QA 시스템 프롬프트 내용 확인."""
        assert "리뷰 분석 전문가" in QA_PROMPT.system_prompt
        assert "근거 기반" in QA_PROMPT.system_prompt
        assert "객관성" in QA_PROMPT.system_prompt

    def test_qa_prompt_user_template_placeholders(self):
        """QA 유저 템플릿 플레이스홀더 확인."""
        assert "{context}" in QA_PROMPT.user_prompt_template
        assert "{question}" in QA_PROMPT.user_prompt_template

    def test_qa_prompt_has_few_shot(self):
        """QA 프롬프트 few-shot 예시 확인."""
        assert len(QA_PROMPT.few_shot_examples) > 0
        example = QA_PROMPT.few_shot_examples[0]
        assert "user" in example
        assert "assistant" in example


class TestSummaryPrompt:
    """요약 프롬프트 테스트."""

    def test_summary_prompt_exists(self):
        """요약 프롬프트 존재 확인."""
        assert SUMMARY_PROMPT is not None
        assert SUMMARY_PROMPT.name == "summary"
        assert SUMMARY_PROMPT.type == PromptType.SUMMARY

    def test_summary_prompt_content(self):
        """요약 프롬프트 내용 확인."""
        assert "요약" in SUMMARY_PROMPT.system_prompt
        assert "수치화" in SUMMARY_PROMPT.system_prompt
        assert "카테고리" in SUMMARY_PROMPT.system_prompt

    def test_summary_prompt_user_template(self):
        """요약 유저 템플릿 확인."""
        assert "{context}" in SUMMARY_PROMPT.user_prompt_template
        assert "{review_count}" in SUMMARY_PROMPT.user_prompt_template

    def test_summary_has_anti_hallucination(self):
        """할루시네이션 방지 지시문 포함 확인."""
        assert "근거 기반" in SUMMARY_PROMPT.system_prompt
        assert "추측 금지" in SUMMARY_PROMPT.system_prompt


class TestComparePrompt:
    """비교 분석 프롬프트 테스트."""

    def test_compare_prompt_exists(self):
        """비교 프롬프트 존재 확인."""
        assert COMPARE_PROMPT is not None
        assert COMPARE_PROMPT.name == "compare"
        assert COMPARE_PROMPT.type == PromptType.COMPARE

    def test_compare_prompt_content(self):
        """비교 프롬프트 내용 확인."""
        assert "비교" in COMPARE_PROMPT.system_prompt
        assert "동일 기준" in COMPARE_PROMPT.system_prompt

    def test_compare_prompt_user_template(self):
        """비교 유저 템플릿 확인."""
        assert "{product_a_reviews}" in COMPARE_PROMPT.user_prompt_template
        assert "{product_b_reviews}" in COMPARE_PROMPT.user_prompt_template


class TestSentimentPrompt:
    """감성 분석 프롬프트 테스트."""

    def test_sentiment_prompt_exists(self):
        """감성 프롬프트 존재 확인."""
        assert SENTIMENT_PROMPT is not None
        assert SENTIMENT_PROMPT.name == "sentiment"
        assert SENTIMENT_PROMPT.type == PromptType.SENTIMENT

    def test_sentiment_prompt_content(self):
        """감성 프롬프트 내용 확인."""
        assert "감성" in SENTIMENT_PROMPT.system_prompt
        assert "긍정" in SENTIMENT_PROMPT.system_prompt or "Positive" in SENTIMENT_PROMPT.system_prompt

    def test_sentiment_prompt_json_format(self):
        """감성 프롬프트 JSON 형식 확인."""
        assert "json" in SENTIMENT_PROMPT.user_prompt_template.lower()


class TestAntiHallucinationInstructions:
    """할루시네이션 방지 지시문 테스트."""

    def test_anti_hallucination_exists(self):
        """할루시네이션 방지 지시문 존재 확인."""
        assert ANTI_HALLUCINATION_INSTRUCTIONS is not None
        assert len(ANTI_HALLUCINATION_INSTRUCTIONS) > 0

    def test_anti_hallucination_content(self):
        """할루시네이션 방지 내용 확인."""
        assert "근거 기반" in ANTI_HALLUCINATION_INSTRUCTIONS
        assert "추측 금지" in ANTI_HALLUCINATION_INSTRUCTIONS
        assert "불확실성" in ANTI_HALLUCINATION_INSTRUCTIONS
        assert "출처" in ANTI_HALLUCINATION_INSTRUCTIONS


class TestPromptManager:
    """PromptManager 테스트."""

    def test_get_prompt(self):
        """프롬프트 가져오기 테스트."""
        prompt = PromptManager.get("qa")
        assert prompt == QA_PROMPT

    def test_get_invalid_prompt(self):
        """없는 프롬프트 에러 테스트."""
        with pytest.raises(KeyError):
            PromptManager.get("nonexistent")

    def test_list_prompts(self):
        """프롬프트 목록 테스트."""
        prompts = PromptManager.list()
        assert "qa" in prompts
        assert "summary" in prompts
        assert "compare" in prompts
        assert "sentiment" in prompts

    def test_register_new_prompt(self):
        """새 프롬프트 등록 테스트."""
        custom = PromptTemplate(
            name="custom_test",
            type=PromptType.QA,
            system_prompt="커스텀",
            user_prompt_template="{question}",
        )
        PromptManager.register(custom)

        assert "custom_test" in PromptManager.list()
        assert PromptManager.get("custom_test") == custom

        # 정리
        del PromptManager._prompts["custom_test"]

    def test_get_by_type(self):
        """타입별 프롬프트 가져오기 테스트."""
        qa_prompts = PromptManager.get_by_type(PromptType.QA)
        assert len(qa_prompts) >= 1
        assert all(p.type == PromptType.QA for p in qa_prompts)

    def test_info(self):
        """프롬프트 정보 테스트."""
        info = PromptManager.info()
        assert "qa" in info
        assert "type" in info["qa"]
        assert "description" in info["qa"]
        assert "version" in info["qa"]
        assert "has_few_shot" in info["qa"]


class TestConvenienceFunctions:
    """편의 함수 테스트."""

    def test_get_prompt_function(self):
        """get_prompt 함수 테스트."""
        prompt = get_prompt("summary")
        assert prompt == SUMMARY_PROMPT

    def test_list_prompts_function(self):
        """list_prompts 함수 테스트."""
        prompts = list_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) >= 4


class TestPromptType:
    """PromptType 열거형 테스트."""

    def test_prompt_types(self):
        """프롬프트 타입 확인."""
        assert PromptType.QA.value == "qa"
        assert PromptType.SUMMARY.value == "summary"
        assert PromptType.COMPARE.value == "compare"
        assert PromptType.SENTIMENT.value == "sentiment"
