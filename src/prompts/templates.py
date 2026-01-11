"""
프롬프트 템플릿 모듈.

리뷰 분석을 위한 다양한 프롬프트 템플릿을 제공합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PromptType(Enum):
    """프롬프트 유형."""

    QA = "qa"
    SUMMARY = "summary"
    COMPARE = "compare"
    SENTIMENT = "sentiment"


@dataclass
class PromptTemplate:
    """프롬프트 템플릿 데이터 구조."""

    name: str
    type: PromptType
    system_prompt: str
    user_prompt_template: str
    description: str = ""
    version: str = "1.0"
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)

    def format_user_prompt(self, **kwargs) -> str:
        """사용자 프롬프트 포맷팅."""
        return self.user_prompt_template.format(**kwargs)

    def get_messages(self, **kwargs) -> list[dict[str, str]]:
        """LangChain 메시지 형식으로 반환."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Few-shot 예시 추가
        for example in self.few_shot_examples:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})

        # 실제 사용자 메시지
        messages.append({
            "role": "user",
            "content": self.format_user_prompt(**kwargs),
        })

        return messages


# =============================================================================
# 공통 지시사항 (할루시네이션 방지)
# =============================================================================

ANTI_HALLUCINATION_INSTRUCTIONS = """
## 중요 제약사항 (반드시 준수)

1. **근거 기반 답변만**: 제공된 리뷰 내용에 있는 정보만 사용하세요.
2. **추측 금지**: 리뷰에 없는 내용을 추측하거나 지어내지 마세요.
3. **불확실성 인정**: 정보가 부족하면 솔직하게 "제공된 리뷰에서는 해당 정보를 찾을 수 없습니다"라고 답하세요.
4. **출처 명시**: 가능하면 어떤 리뷰를 참고했는지 언급하세요.
5. **수치화**: 구체적인 수치나 비율을 사용하세요. (예: "5개 리뷰 중 4개가 긍정적")
"""


# =============================================================================
# Q&A 프롬프트 (질의응답)
# =============================================================================

QA_SYSTEM_PROMPT = """당신은 상품 리뷰 분석 전문가입니다.
사용자의 질문에 대해 제공된 리뷰 데이터를 기반으로 정확하고 객관적으로 답변합니다.

## 답변 원칙
1. **근거 기반**: 반드시 제공된 리뷰 내용을 근거로 답변하세요.
2. **객관성 유지**: 리뷰에 없는 내용을 추측하거나 지어내지 마세요.
3. **수치화**: 가능하면 구체적인 수치나 비율을 언급하세요. (예: "5개 리뷰 중 4개가 긍정적")
4. **균형잡힌 시각**: 긍정적/부정적 의견이 모두 있다면 양쪽을 언급하세요.
5. **불확실성 인정**: 정보가 부족하면 "제공된 리뷰에서는 해당 정보를 찾을 수 없습니다"라고 답하세요.

## 답변 형식
- 먼저 핵심 답변을 제시하세요.
- 그 다음 근거가 되는 리뷰 내용을 인용하세요.
- 마지막으로 추가 참고사항이 있으면 언급하세요."""

QA_USER_TEMPLATE = """## 참고 리뷰
{context}

## 질문
{question}

위 리뷰들을 참고하여 질문에 답변해주세요."""

QA_FEW_SHOT_EXAMPLES = [
    {
        "user": """## 참고 리뷰
[리뷰 1] (평점: 5.0, 날짜: 2024-01-15)
배송이 정말 빨라서 놀랐어요! 주문 다음날 도착했습니다.

[리뷰 2] (평점: 4.0, 날짜: 2024-01-14)
배송은 빨랐는데 포장이 좀 아쉬웠어요.

## 질문
배송은 어떤가요?""",
        "assistant": """배송은 전반적으로 빠른 편입니다.

**근거:**
- 리뷰 1에서 "주문 다음날 도착"이라고 언급하며 배송 속도에 만족감을 표현했습니다.
- 리뷰 2에서도 "배송은 빨랐다"고 언급했습니다.

**참고사항:**
- 2개 리뷰 모두 배송 속도에 대해 긍정적이나, 리뷰 2에서는 포장 상태에 대한 아쉬움이 있었습니다.""",
    },
]

QA_PROMPT = PromptTemplate(
    name="qa",
    type=PromptType.QA,
    system_prompt=QA_SYSTEM_PROMPT,
    user_prompt_template=QA_USER_TEMPLATE,
    description="리뷰 기반 질의응답 프롬프트",
    version="1.0",
    few_shot_examples=QA_FEW_SHOT_EXAMPLES,
)


# =============================================================================
# 요약 프롬프트 (Summarization)
# =============================================================================

SUMMARY_SYSTEM_PROMPT = """당신은 상품 리뷰 요약 전문가입니다.
여러 리뷰를 분석하여 핵심 내용을 구조화된 형태로 요약합니다.

## 요약 원칙
1. **수치화**: 긍정/부정/중립 비율을 명시하세요.
2. **카테고리화**: 주요 언급 주제별로 분류하세요. (배송, 품질, 가격, 디자인 등)
3. **대표 의견**: 각 카테고리에서 가장 대표적인 의견을 인용하세요.
4. **객관성**: 편향되지 않게 다양한 관점을 포함하세요.
""" + ANTI_HALLUCINATION_INSTRUCTIONS

SUMMARY_USER_TEMPLATE = """## 분석할 리뷰 ({review_count}개)
{context}

## 요청
위 리뷰들을 분석하여 다음 형식으로 요약해주세요:

1. **전체 평가**: 전반적인 평가 요약 (긍정/부정/중립 비율 포함)
2. **주요 장점**: 고객들이 언급한 주요 장점 (빈도순)
3. **주요 단점**: 고객들이 언급한 주요 단점 (빈도순)
4. **추천 여부**: 종합적인 추천 의견"""

SUMMARY_FEW_SHOT_EXAMPLES = [
    {
        "user": """## 분석할 리뷰 (3개)
[리뷰 1] (평점: 5.0) 가격 대비 품질이 훌륭합니다. 배송도 빨라요.
[리뷰 2] (평점: 4.0) 디자인이 예쁘고 품질도 괜찮아요. 다만 사이즈가 좀 작네요.
[리뷰 3] (평점: 2.0) 품질이 기대 이하입니다. 가격이 아깝네요.

## 요청
위 리뷰들을 분석하여 요약해주세요.""",
        "assistant": """## 리뷰 요약 분석

### 1. 전체 평가
- **긍정**: 2개 (67%)
- **부정**: 1개 (33%)
- **평균 평점**: 3.7점

전반적으로 긍정적인 평가가 우세하나, 품질에 대한 의견이 엇갈립니다.

### 2. 주요 장점
1. **가격 대비 품질** - "가격 대비 품질이 훌륭합니다" (리뷰 1)
2. **디자인** - "디자인이 예쁘고" (리뷰 2)
3. **배송 속도** - "배송도 빨라요" (리뷰 1)

### 3. 주요 단점
1. **품질 일관성** - "품질이 기대 이하" (리뷰 3)
2. **사이즈** - "사이즈가 좀 작네요" (리뷰 2)

### 4. 추천 여부
3개 리뷰 중 2개가 긍정적이므로 **조건부 추천**합니다. 단, 사이즈 선택에 주의가 필요하며, 품질에 대한 기대치를 조절할 필요가 있습니다.""",
    },
]

SUMMARY_PROMPT = PromptTemplate(
    name="summary",
    type=PromptType.SUMMARY,
    system_prompt=SUMMARY_SYSTEM_PROMPT,
    user_prompt_template=SUMMARY_USER_TEMPLATE,
    description="리뷰 요약 프롬프트 (수치화 포함)",
    version="1.0",
    few_shot_examples=SUMMARY_FEW_SHOT_EXAMPLES,
)


# =============================================================================
# 비교 분석 프롬프트 (Comparison)
# =============================================================================

COMPARE_SYSTEM_PROMPT = """당신은 상품 비교 분석 전문가입니다.
여러 상품의 리뷰를 분석하여 객관적인 비교 정보를 제공합니다.

## 비교 원칙
1. **동일 기준 적용**: 모든 상품에 동일한 평가 기준을 적용하세요.
2. **카테고리별 비교**: 배송, 품질, 가격, 디자인 등 카테고리별로 비교하세요.
3. **수치 기반**: 평점, 긍정/부정 비율 등 수치를 활용하세요.
4. **장단점 명시**: 각 상품의 장단점을 명확히 구분하세요.
5. **추천 상황 제시**: 어떤 상황에 어떤 상품이 적합한지 안내하세요.
""" + ANTI_HALLUCINATION_INSTRUCTIONS

COMPARE_USER_TEMPLATE = """## 상품 A 리뷰
{product_a_reviews}

## 상품 B 리뷰
{product_b_reviews}

## 요청
위 두 상품의 리뷰를 비교 분석하여 다음 형식으로 정리해주세요:

1. **전체 평가 비교**: 평점, 긍정 비율 등
2. **카테고리별 비교**: 품질, 가격, 배송, 디자인 등
3. **장단점 요약**: 각 상품의 장단점
4. **추천 의견**: 상황별 추천 상품"""

COMPARE_PROMPT = PromptTemplate(
    name="compare",
    type=PromptType.COMPARE,
    system_prompt=COMPARE_SYSTEM_PROMPT,
    user_prompt_template=COMPARE_USER_TEMPLATE,
    description="상품 비교 분석 프롬프트",
    version="1.0",
    few_shot_examples=[],  # 비교는 상황이 다양하여 few-shot 생략
)


# =============================================================================
# 감성 분석 프롬프트 (Sentiment Analysis)
# =============================================================================

SENTIMENT_SYSTEM_PROMPT = """당신은 리뷰 감성 분석 전문가입니다.
리뷰 텍스트를 분석하여 감성과 주요 속성을 추출합니다.

## 분석 원칙
1. **감성 분류**: 긍정(Positive), 부정(Negative), 중립(Neutral)으로 분류
2. **속성 추출**: 언급된 주요 속성(가격, 품질, 배송, 디자인 등) 식별
3. **속성별 감성**: 각 속성에 대한 감성도 분석
4. **신뢰도 점수**: 분석 결과의 신뢰도를 0-1 사이로 표시
""" + ANTI_HALLUCINATION_INSTRUCTIONS

SENTIMENT_USER_TEMPLATE = """## 분석할 리뷰
{review_text}

## 요청
위 리뷰를 분석하여 다음 JSON 형식으로 결과를 반환해주세요:

```json
{{
  "overall_sentiment": "positive|negative|neutral",
  "confidence": 0.0-1.0,
  "aspects": [
    {{
      "name": "속성명",
      "sentiment": "positive|negative|neutral",
      "keywords": ["관련 키워드"]
    }}
  ],
  "summary": "한 줄 요약"
}}
```"""

SENTIMENT_PROMPT = PromptTemplate(
    name="sentiment",
    type=PromptType.SENTIMENT,
    system_prompt=SENTIMENT_SYSTEM_PROMPT,
    user_prompt_template=SENTIMENT_USER_TEMPLATE,
    description="감성 분석 프롬프트 (속성별 분석)",
    version="1.0",
    few_shot_examples=[],
)


# =============================================================================
# 프롬프트 관리자
# =============================================================================

class PromptManager:
    """프롬프트 템플릿 관리 클래스."""

    _prompts: dict[str, PromptTemplate] = {
        "qa": QA_PROMPT,
        "summary": SUMMARY_PROMPT,
        "compare": COMPARE_PROMPT,
        "sentiment": SENTIMENT_PROMPT,
    }

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        """이름으로 프롬프트 템플릿 가져오기."""
        if name not in cls._prompts:
            raise KeyError(f"프롬프트 '{name}'을 찾을 수 없습니다. 사용 가능: {list(cls._prompts.keys())}")
        return cls._prompts[name]

    @classmethod
    def list(cls) -> list[str]:
        """사용 가능한 프롬프트 목록."""
        return list(cls._prompts.keys())

    @classmethod
    def register(cls, prompt: PromptTemplate) -> None:
        """새 프롬프트 등록."""
        cls._prompts[prompt.name] = prompt

    @classmethod
    def get_by_type(cls, prompt_type: PromptType) -> list[PromptTemplate]:
        """타입별 프롬프트 목록."""
        return [p for p in cls._prompts.values() if p.type == prompt_type]

    @classmethod
    def info(cls) -> dict[str, Any]:
        """모든 프롬프트 정보."""
        return {
            name: {
                "type": p.type.value,
                "description": p.description,
                "version": p.version,
                "has_few_shot": len(p.few_shot_examples) > 0,
            }
            for name, p in cls._prompts.items()
        }


# 편의 함수
def get_prompt(name: str) -> PromptTemplate:
    """프롬프트 가져오기 편의 함수."""
    return PromptManager.get(name)


def list_prompts() -> list[str]:
    """프롬프트 목록 편의 함수."""
    return PromptManager.list()
