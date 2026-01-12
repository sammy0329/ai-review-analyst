"""
Intent Classifier 노드.

사용자 질문의 의도를 분류합니다.
하이브리드 방식: 규칙 기반 + LLM 기반
"""

import re
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agents.state import AgentState, IntentType


# 규칙 기반 분류를 위한 키워드 패턴
INTENT_PATTERNS: dict[IntentType, list[str]] = {
    IntentType.SUMMARY: [
        r"요약",
        r"정리",
        r"종합",
        r"전체.*리뷰",
        r"리뷰.*분석",
        r"어떤.*평가",
        r"평가.*어떤",
        r"전반적",
        r"총평",
        r"개요",
    ],
    IntentType.COMPARE: [
        r"비교",
        r"vs",
        r"versus",
        r"차이",
        r"뭐가.*나아",
        r"어떤.*더",
        r"둘.*중",
        r"선택",
        r"고민",
        r"어느.*좋",
    ],
}


class IntentClassification(BaseModel):
    """의도 분류 결과."""

    intent: str = Field(description="분류된 의도 (qa, summary, compare)")
    confidence: float = Field(description="분류 신뢰도 (0.0 ~ 1.0)")
    reasoning: str = Field(description="분류 이유")


INTENT_CLASSIFICATION_PROMPT = """사용자 질문의 의도를 분류하세요.

## 의도 유형
- qa: 제품에 대한 구체적 질문 (품질, 배송, 가격, 사용감 등)
- summary: 리뷰 전체 요약 요청
- compare: 두 개 이상 제품 비교

## 분류 기준
- "요약해줘", "전체적으로", "종합" → summary
- "비교", "vs", "뭐가 더 좋아" → compare
- 구체적 질문 (배송, 품질, 가격 등) → qa
- 불명확한 경우 → qa (기본값)

## 질문
{query}

## 제품
{product_name}

JSON 형식으로 답변해주세요."""


def rule_based_classification(query: str) -> tuple[IntentType | None, float]:
    """규칙 기반 의도 분류.

    Args:
        query: 사용자 질문

    Returns:
        (의도 유형, 신뢰도) 또는 (None, 0.0) if 분류 불가
    """
    query_lower = query.lower()

    for intent_type, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return intent_type, 0.9

    return None, 0.0


def llm_based_classification(
    query: str,
    product_name: str | None,
    llm: ChatOpenAI | None = None,
) -> tuple[IntentType, float]:
    """LLM 기반 의도 분류.

    Args:
        query: 사용자 질문
        product_name: 제품명
        llm: ChatOpenAI 인스턴스 (None이면 새로 생성)

    Returns:
        (의도 유형, 신뢰도)
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Structured Output 사용
    structured_llm = llm.with_structured_output(IntentClassification)

    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        query=query,
        product_name=product_name or "지정되지 않음",
    )

    try:
        result: IntentClassification = structured_llm.invoke(prompt)

        # 문자열을 IntentType으로 변환
        intent_str = result.intent.lower()
        if intent_str == "summary":
            intent = IntentType.SUMMARY
        elif intent_str == "compare":
            intent = IntentType.COMPARE
        elif intent_str == "qa":
            intent = IntentType.QA
        else:
            intent = IntentType.UNKNOWN

        return intent, result.confidence

    except Exception:
        # LLM 호출 실패 시 기본값
        return IntentType.QA, 0.5


def classify_intent(
    state: AgentState,
    llm: ChatOpenAI | None = None,
) -> AgentState:
    """의도 분류 노드.

    하이브리드 방식:
    1. 규칙 기반 분류 시도
    2. 실패 시 LLM 기반 분류

    Args:
        state: 현재 에이전트 상태
        llm: ChatOpenAI 인스턴스 (선택)

    Returns:
        의도가 분류된 상태
    """
    query = state["query"]

    # 1차: 규칙 기반 분류
    intent, confidence = rule_based_classification(query)

    if intent is not None and confidence >= 0.8:
        return {
            **state,
            "intent": intent,
            "intent_confidence": confidence,
            "metadata": {
                **state.get("metadata", {}),
                "classification_method": "rule_based",
            },
        }

    # 2차: LLM 기반 분류
    intent, confidence = llm_based_classification(
        query=query,
        product_name=state.get("product_name"),
        llm=llm,
    )

    return {
        **state,
        "intent": intent,
        "intent_confidence": confidence,
        "metadata": {
            **state.get("metadata", {}),
            "classification_method": "llm_based",
        },
    }


def create_intent_classifier_node(llm: ChatOpenAI | None = None):
    """Intent Classifier 노드 생성.

    Args:
        llm: ChatOpenAI 인스턴스 (선택)

    Returns:
        노드 함수
    """

    def node(state: AgentState) -> AgentState:
        return classify_intent(state, llm)

    return node
