"""
LangGraph 에이전트 상태 정의.

멀티 에이전트 시스템의 공유 상태 스키마를 정의합니다.
"""

from enum import Enum
from typing import Any, TypedDict


class IntentType(str, Enum):
    """사용자 의도 유형."""

    QA = "qa"  # 일반 질문 (RAG 기반)
    SUMMARY = "summary"  # 리뷰 요약
    COMPARE = "compare"  # 제품 비교
    UNKNOWN = "unknown"  # 알 수 없음


class AgentState(TypedDict, total=False):
    """LangGraph 에이전트 상태.

    모든 노드 간 공유되는 상태 정의.

    Attributes:
        query: 사용자 질문
        product_name: 대상 제품명
        product_names: 비교 시 제품 목록
        intent: 분류된 의도
        intent_confidence: 분류 신뢰도 (0.0 ~ 1.0)
        context: 검색된 리뷰 컨텍스트
        source_documents: 출처 문서들
        response: 최종 응답
        metadata: 응답 메타데이터
        error: 에러 메시지
    """

    # 입력
    query: str
    product_name: str | None
    product_names: list[str] | None

    # 의도 분류
    intent: IntentType
    intent_confidence: float

    # 컨텍스트
    context: str | None
    source_documents: list[dict[str, Any]]

    # 출력
    response: str
    metadata: dict[str, Any]

    # 에러 핸들링
    error: str | None


def create_initial_state(
    query: str,
    product_name: str | None = None,
    product_names: list[str] | None = None,
) -> AgentState:
    """초기 상태 생성.

    Args:
        query: 사용자 질문
        product_name: 대상 제품명
        product_names: 비교할 제품 목록

    Returns:
        초기화된 AgentState
    """
    return AgentState(
        query=query,
        product_name=product_name,
        product_names=product_names,
        intent=IntentType.UNKNOWN,
        intent_confidence=0.0,
        context=None,
        source_documents=[],
        response="",
        metadata={},
        error=None,
    )
