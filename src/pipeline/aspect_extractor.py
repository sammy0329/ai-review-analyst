"""
LLM 기반 속성 추출 모듈.

Raw 리뷰 텍스트에서 속성(Aspect)을 자동 추출하고 감정을 분석합니다.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Sentiment(str, Enum):
    """감정 극성."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AspectCategory(str, Enum):
    """속성 카테고리."""

    PRICE = "가격/가성비"
    DESIGN = "디자인/외관"
    SIZE = "사이즈/치수"
    QUALITY = "소재/품질"
    DELIVERY = "배송/포장"
    COLOR = "색상"
    FUNCTION = "기능/성능"
    COMFORT = "착용감/편안함"
    DURABILITY = "내구성"
    SERVICE = "서비스/응대"
    OTHER = "기타"


# Pydantic 모델 (Structured Output용)
class ExtractedAspect(BaseModel):
    """추출된 개별 속성."""

    category: str = Field(description="속성 카테고리 (가격/가성비, 디자인/외관, 사이즈/치수, 소재/품질, 배송/포장, 색상, 기능/성능, 착용감/편안함, 내구성, 서비스/응대, 기타)")
    sentiment: str = Field(description="감정 (positive, negative, neutral)")
    text: str = Field(description="해당 속성에 대한 원문 발췌")
    keywords: list[str] = Field(default_factory=list, description="관련 키워드")


class AspectExtractionResult(BaseModel):
    """속성 추출 결과."""

    aspects: list[ExtractedAspect] = Field(default_factory=list, description="추출된 속성 목록")
    overall_sentiment: str = Field(description="전체 감정 (positive, negative, neutral)")
    confidence: float = Field(ge=0.0, le=1.0, description="추출 신뢰도 (0.0~1.0)")


@dataclass
class AspectResult:
    """속성 추출 결과 데이터 구조."""

    review_text: str
    overall_sentiment: Sentiment
    confidence: float
    aspects: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_extraction_result(
        cls,
        review_text: str,
        result: AspectExtractionResult,
        metadata: dict[str, Any] | None = None,
    ) -> "AspectResult":
        """AspectExtractionResult에서 AspectResult 생성."""
        aspects = [
            {
                "category": aspect.category,
                "sentiment": aspect.sentiment,
                "text": aspect.text,
                "keywords": aspect.keywords,
            }
            for aspect in result.aspects
        ]

        return cls(
            review_text=review_text,
            overall_sentiment=Sentiment(result.overall_sentiment),
            confidence=result.confidence,
            aspects=aspects,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "review_text": self.review_text,
            "overall_sentiment": self.overall_sentiment.value,
            "confidence": self.confidence,
            "aspects": self.aspects,
            "metadata": self.metadata,
        }

    def get_aspect_by_category(self, category: str) -> list[dict[str, Any]]:
        """특정 카테고리의 속성 반환."""
        return [a for a in self.aspects if a["category"] == category]

    def get_positive_aspects(self) -> list[dict[str, Any]]:
        """긍정 속성 반환."""
        return [a for a in self.aspects if a["sentiment"] == "positive"]

    def get_negative_aspects(self) -> list[dict[str, Any]]:
        """부정 속성 반환."""
        return [a for a in self.aspects if a["sentiment"] == "negative"]


# 속성 추출 프롬프트
ASPECT_EXTRACTION_SYSTEM_PROMPT = """당신은 이커머스 리뷰 분석 전문가입니다.
리뷰 텍스트를 분석하여 언급된 속성(Aspect)과 각 속성에 대한 감정을 추출합니다.

## 추출 대상 속성 카테고리
- 가격/가성비: 가격, 비용, 가성비, 할인 등
- 디자인/외관: 디자인, 모양, 외관, 스타일 등
- 사이즈/치수: 사이즈, 크기, 길이, 폭, 두께 등
- 소재/품질: 소재, 재질, 품질, 마감 등
- 배송/포장: 배송, 포장, 택배, 배달 등
- 색상: 색상, 색깔, 컬러 등
- 기능/성능: 기능, 성능, 효과, 효능 등
- 착용감/편안함: 착용감, 편안함, 착화감 등
- 내구성: 내구성, 수명, 튼튼함 등
- 서비스/응대: 서비스, 응대, 고객센터, AS 등
- 기타: 위 카테고리에 해당하지 않는 기타 속성

## 분석 원칙
1. 리뷰에 실제로 언급된 속성만 추출하세요.
2. 각 속성에 대한 감정(긍정/부정/중립)을 정확히 판단하세요.
3. 해당 속성과 관련된 원문을 발췌하세요.
4. 관련 키워드를 추출하세요.
5. 속성이 없으면 빈 리스트를 반환하세요."""


class AspectExtractor:
    """LLM 기반 속성 추출기."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        openai_api_key: str | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ):
        """
        초기화.

        Args:
            model_name: LLM 모델명
            temperature: 온도 설정
            openai_api_key: OpenAI API 키
            cache_dir: 캐시 디렉토리 (기본: ./data/aspect_cache)
            use_cache: 캐시 사용 여부
        """
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API 키가 필요합니다. "
                "OPENAI_API_KEY 환경변수를 설정하거나 openai_api_key 파라미터를 전달하세요."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.use_cache = use_cache

        # 캐시 디렉토리 설정
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/aspect_cache")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # LLM 초기화 (with_structured_output 사용)
        self._llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self._api_key,
        ).with_structured_output(AspectExtractionResult)

    def _get_cache_key(self, text: str) -> str:
        """텍스트의 캐시 키 생성."""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_from_cache(self, text: str) -> AspectResult | None:
        """캐시에서 결과 조회."""
        if not self.use_cache:
            return None

        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return AspectResult(
                    review_text=data["review_text"],
                    overall_sentiment=Sentiment(data["overall_sentiment"]),
                    confidence=data["confidence"],
                    aspects=data["aspects"],
                    metadata=data.get("metadata", {}),
                )
            except Exception:
                return None
        return None

    def _save_to_cache(self, result: AspectResult) -> None:
        """결과를 캐시에 저장."""
        if not self.use_cache:
            return

        cache_key = self._get_cache_key(result.review_text)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 캐시 저장 실패는 무시

    def extract(self, review_text: str, metadata: dict[str, Any] | None = None) -> AspectResult:
        """
        단일 리뷰에서 속성 추출.

        Args:
            review_text: 리뷰 텍스트
            metadata: 추가 메타데이터

        Returns:
            AspectResult 객체
        """
        # 캐시 확인
        cached = self._get_from_cache(review_text)
        if cached:
            if metadata:
                cached.metadata.update(metadata)
            return cached

        # 프롬프트 구성
        messages = [
            {"role": "system", "content": ASPECT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"다음 리뷰를 분석해주세요:\n\n{review_text}"},
        ]

        # LLM 호출
        try:
            extraction_result: AspectExtractionResult = self._llm.invoke(messages)

            result = AspectResult.from_extraction_result(
                review_text=review_text,
                result=extraction_result,
                metadata=metadata,
            )

            # 캐시 저장
            self._save_to_cache(result)

            return result

        except Exception as e:
            # 에러 시 빈 결과 반환
            return AspectResult(
                review_text=review_text,
                overall_sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
                aspects=[],
                metadata={"error": str(e), **(metadata or {})},
            )

    def extract_batch(
        self,
        reviews: list[str | dict[str, Any]],
        show_progress: bool = True,
    ) -> list[AspectResult]:
        """
        여러 리뷰에서 속성 추출 (배치 처리).

        Args:
            reviews: 리뷰 텍스트 리스트 또는 {"text": ..., "metadata": ...} 형태의 딕셔너리 리스트
            show_progress: 진행 상황 출력 여부

        Returns:
            AspectResult 리스트
        """
        results = []
        total = len(reviews)

        for i, review in enumerate(reviews):
            if isinstance(review, str):
                text = review
                metadata = {}
            else:
                text = review.get("text", "")
                metadata = review.get("metadata", {})

            if show_progress:
                print(f"\r속성 추출 중: {i + 1}/{total}", end="", flush=True)

            result = self.extract(text, metadata)
            results.append(result)

        if show_progress:
            print()  # 줄바꿈

        return results

    def get_aspect_statistics(
        self,
        results: list[AspectResult],
    ) -> dict[str, Any]:
        """
        속성 추출 결과 통계 계산.

        Args:
            results: AspectResult 리스트

        Returns:
            통계 딕셔너리
        """
        stats = {
            "total_reviews": len(results),
            "overall_sentiment": {
                "positive": 0,
                "negative": 0,
                "neutral": 0,
            },
            "aspect_counts": {},
            "aspect_sentiment": {},
            "avg_confidence": 0.0,
        }

        confidence_sum = 0.0

        for result in results:
            # 전체 감정 집계
            sentiment_key = result.overall_sentiment.value
            stats["overall_sentiment"][sentiment_key] += 1
            confidence_sum += result.confidence

            # 속성별 집계
            for aspect in result.aspects:
                category = aspect["category"]
                sentiment = aspect["sentiment"]

                # 속성 개수
                if category not in stats["aspect_counts"]:
                    stats["aspect_counts"][category] = 0
                stats["aspect_counts"][category] += 1

                # 속성별 감정
                if category not in stats["aspect_sentiment"]:
                    stats["aspect_sentiment"][category] = {
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0,
                    }
                stats["aspect_sentiment"][category][sentiment] += 1

        # 평균 신뢰도
        if results:
            stats["avg_confidence"] = confidence_sum / len(results)

        # 속성 개수 정렬 (빈도순)
        stats["aspect_counts"] = dict(
            sorted(stats["aspect_counts"].items(), key=lambda x: -x[1])
        )

        return stats

    def clear_cache(self) -> int:
        """
        캐시 삭제.

        Returns:
            삭제된 파일 수
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        return count


def create_aspect_extractor(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    openai_api_key: str | None = None,
    use_cache: bool = True,
) -> AspectExtractor:
    """
    AspectExtractor 생성 헬퍼 함수.

    Args:
        model_name: LLM 모델명
        temperature: 온도 설정
        openai_api_key: OpenAI API 키
        use_cache: 캐시 사용 여부

    Returns:
        AspectExtractor 인스턴스
    """
    return AspectExtractor(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key,
        use_cache=use_cache,
    )


def main():
    """테스트 실행."""
    from dotenv import load_dotenv

    # 환경변수 로드
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    print("=" * 60)
    print("🔍 속성 추출기 (Aspect Extractor) 테스트")
    print("=" * 60)

    # 테스트 리뷰
    test_reviews = [
        "가격은 좀 비싸지만 소재가 정말 좋아요. 배송도 빨랐습니다.",
        "디자인이 예쁘고 색상도 마음에 들어요. 다만 사이즈가 조금 작네요.",
        "품질이 기대 이하입니다. 가격 대비 별로예요. 실망했어요.",
        "착용감이 편하고 기능도 좋아요. 재구매 의사 있습니다!",
        "배송이 느렸어요. 포장도 엉성했고요. 제품은 괜찮은데...",
    ]

    # 추출기 생성
    extractor = create_aspect_extractor(use_cache=True)
    print("\n✅ 속성 추출기 생성 완료")

    # 단일 추출 테스트
    print("\n" + "─" * 50)
    print("📝 단일 리뷰 속성 추출 테스트")
    print("─" * 50)

    review = test_reviews[0]
    print(f"\n📄 리뷰: {review}")

    result = extractor.extract(review)

    print(f"\n🎯 전체 감정: {result.overall_sentiment.value}")
    print(f"📊 신뢰도: {result.confidence:.2f}")
    print(f"\n📋 추출된 속성 ({len(result.aspects)}개):")
    for aspect in result.aspects:
        sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        emoji = sentiment_emoji.get(aspect["sentiment"], "❓")
        print(f"   {emoji} [{aspect['category']}] {aspect['sentiment']}")
        print(f"      원문: {aspect['text']}")
        if aspect["keywords"]:
            print(f"      키워드: {', '.join(aspect['keywords'])}")

    # 배치 추출 테스트
    print("\n" + "─" * 50)
    print("📦 배치 속성 추출 테스트")
    print("─" * 50)

    results = extractor.extract_batch(test_reviews)
    print(f"\n✅ {len(results)}개 리뷰 처리 완료")

    # 통계
    print("\n" + "─" * 50)
    print("📊 속성 추출 통계")
    print("─" * 50)

    stats = extractor.get_aspect_statistics(results)

    print(f"\n📌 전체 리뷰: {stats['total_reviews']}개")
    print(f"📌 평균 신뢰도: {stats['avg_confidence']:.2f}")

    print("\n🎭 전체 감정 분포:")
    for sentiment, count in stats["overall_sentiment"].items():
        pct = count / stats["total_reviews"] * 100
        bar = "█" * int(pct / 5)
        print(f"   {sentiment}: {count}개 ({pct:.1f}%) {bar}")

    print("\n📋 속성별 언급 빈도:")
    for category, count in stats["aspect_counts"].items():
        print(f"   {category}: {count}회")

        if category in stats["aspect_sentiment"]:
            sent = stats["aspect_sentiment"][category]
            print(f"      긍정: {sent['positive']}, 부정: {sent['negative']}, 중립: {sent['neutral']}")

    print("\n" + "=" * 60)
    print("✅ 속성 추출기 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
