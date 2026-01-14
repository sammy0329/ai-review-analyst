"""
가짜 리뷰 필터링 모듈.

어뷰징 패턴 탐지 및 LLM 기반 가짜 리뷰 분류를 제공합니다.
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.crawler.base import Review


class FakeReviewReason(str, Enum):
    """가짜 리뷰 의심 사유."""

    EXCESSIVE_PRAISE = "excessive_praise"  # 과도한 칭찬
    REPETITIVE_PATTERN = "repetitive_pattern"  # 반복 문구
    TOO_SHORT = "too_short"  # 너무 짧은 리뷰
    NO_SPECIFICS = "no_specifics"  # 구체성 부족
    IRRELEVANT_CONTENT = "irrelevant_content"  # 관련 없는 내용
    SPAM_KEYWORDS = "spam_keywords"  # 스팸 키워드
    EXTREME_RATING = "extreme_rating"  # 극단적 평점과 내용 불일치
    LLM_DETECTED = "llm_detected"  # LLM이 탐지


@dataclass
class FakeReviewResult:
    """가짜 리뷰 판정 결과."""

    is_suspicious: bool  # 의심 여부
    confidence: float  # 신뢰도 (0.0 ~ 1.0)
    reasons: list[FakeReviewReason] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    weight: float = 1.0  # 가중치 (의심되면 낮아짐)


# =============================================================================
# 규칙 기반 패턴 정의
# =============================================================================

# 과도한 칭찬 패턴
EXCESSIVE_PRAISE_PATTERNS = [
    r"최고의\s*제품",
    r"인생\s*(템|제품|아이템)",
    r"완전\s*강추",
    r"무조건\s*(사세요|구매)",
    r"이것만\s*쓰세요",
    r"두\s*번\s*사도\s*아깝지",
    r"100점\s*만점",
    r"완벽(합니다|해요)",
    r"흠잡을\s*데\s*없",
    r"너무너무\s*좋아",
]

# 스팸/광고 키워드
SPAM_KEYWORDS = [
    r"협찬",
    r"광고",
    r"체험단",
    r"업체\s*제공",
    r"무료\s*제공",
    r"서포터즈",
    r"인플루언서",
    r"판매자\s*입니다",
    r"문의\s*주세요",
    r"연락\s*주시면",
    r"카톡\s*문의",
    r"010-",
    r"http[s]?://",
]

# 반복 패턴 (동일 문구 반복)
REPETITIVE_THRESHOLD = 3  # 같은 단어가 3번 이상 반복

# 최소 리뷰 길이 (너무 짧은 리뷰)
MIN_REVIEW_LENGTH = 10

# 구체성 키워드 (이런 키워드가 없으면 구체성 부족으로 판단)
SPECIFICITY_KEYWORDS = [
    r"배송",
    r"품질",
    r"가격",
    r"색상",
    r"사이즈",
    r"디자인",
    r"소재",
    r"포장",
    r"착용",
    r"사용",
    r"효과",
    r"향",
    r"발림",
    r"흡수",
]


# =============================================================================
# LLM 기반 분류
# =============================================================================

class LLMFakeReviewResult(BaseModel):
    """LLM 가짜 리뷰 판정 결과."""

    is_fake: bool = Field(description="가짜 리뷰 여부")
    confidence: float = Field(description="신뢰도 (0.0 ~ 1.0)")
    reason: str = Field(description="판정 사유")


FAKE_REVIEW_DETECTION_PROMPT = """당신은 이커머스 리뷰 품질 분석 전문가입니다.
주어진 리뷰가 가짜(어뷰징) 리뷰인지 판정해주세요.

## 가짜 리뷰 특징
1. **과도한 칭찬**: 비현실적으로 긍정적, 광고성 표현
2. **구체성 부족**: 실제 사용 경험 없이 일반적인 표현만
3. **관련 없는 내용**: 제품과 무관한 내용
4. **스팸/광고**: 홍보성 문구, 연락처, 링크 포함
5. **복붙 의심**: 너무 정형화된 표현
6. **평점-내용 불일치**: 평점은 높은데 내용은 부정적 (또는 반대)

## 진짜 리뷰 특징
1. 구체적인 사용 경험 언급
2. 장단점 균형있게 서술
3. 개인적인 의견과 감정 표현
4. 제품 특성에 대한 구체적 언급

## 리뷰 정보
- 텍스트: {text}
- 평점: {rating}

## 판정 기준
- is_fake: true면 가짜 리뷰로 의심
- confidence: 0.5 이하면 확신 없음, 0.8 이상이면 확신
- reason: 판정 사유를 간단히 설명

JSON 형식으로 답변해주세요."""


# =============================================================================
# FakeReviewFilter 클래스
# =============================================================================

class FakeReviewFilter:
    """가짜 리뷰 필터 클래스.

    하이브리드 방식:
    1. 규칙 기반 탐지 (빠르고 비용 없음)
    2. LLM 기반 탐지 (의심되는 경우에만)
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_threshold: float = 0.5,
        llm: ChatOpenAI | None = None,
    ):
        """초기화.

        Args:
            use_llm: LLM 기반 탐지 사용 여부
            llm_threshold: 규칙 기반에서 이 이상 의심되면 LLM 호출
            llm: ChatOpenAI 인스턴스 (None이면 새로 생성)
        """
        self.use_llm = use_llm
        self.llm_threshold = llm_threshold
        self.llm = llm

    def _get_llm(self) -> ChatOpenAI:
        """LLM 인스턴스 반환 (지연 로딩)."""
        if self.llm is None:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        return self.llm

    def check_review(self, review: Review) -> FakeReviewResult:
        """리뷰가 가짜인지 확인.

        Args:
            review: 검사할 리뷰

        Returns:
            FakeReviewResult 객체
        """
        text = review.text
        rating = review.rating or 0

        # 1단계: 규칙 기반 검사
        rule_result = self._rule_based_check(text, rating)

        # 2단계: LLM 기반 검사 (옵션)
        # 규칙 기반에서 의심되지만 확신이 낮을 때 LLM으로 추가 검증
        if (
            self.use_llm
            and rule_result.confidence >= self.llm_threshold
            and rule_result.confidence < 0.8
        ):
            llm_result = self._llm_based_check(text, rating)

            # LLM 결과 병합
            if llm_result.is_suspicious:
                rule_result.is_suspicious = True
                rule_result.confidence = max(rule_result.confidence, llm_result.confidence)
                rule_result.reasons.extend(llm_result.reasons)
                rule_result.details.update(llm_result.details)

        # 가중치 계산 (항상 수행)
        rule_result.weight = self._calculate_weight(rule_result)

        return rule_result

    def _rule_based_check(self, text: str, rating: float) -> FakeReviewResult:
        """규칙 기반 가짜 리뷰 검사.

        Args:
            text: 리뷰 텍스트
            rating: 평점

        Returns:
            FakeReviewResult 객체
        """
        reasons = []
        details = {}
        scores = []

        # 1. 과도한 칭찬 패턴
        praise_matches = []
        for pattern in EXCESSIVE_PRAISE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            praise_matches.extend(matches)

        if praise_matches:
            reasons.append(FakeReviewReason.EXCESSIVE_PRAISE)
            details["praise_matches"] = praise_matches
            scores.append(min(len(praise_matches) * 0.3, 0.9))

        # 2. 스팸/광고 키워드
        spam_matches = []
        for pattern in SPAM_KEYWORDS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            spam_matches.extend(matches)

        if spam_matches:
            reasons.append(FakeReviewReason.SPAM_KEYWORDS)
            details["spam_matches"] = spam_matches
            scores.append(0.9)  # 스팸은 높은 점수

        # 3. 너무 짧은 리뷰
        if len(text.strip()) < MIN_REVIEW_LENGTH:
            reasons.append(FakeReviewReason.TOO_SHORT)
            details["text_length"] = len(text.strip())
            scores.append(0.4)

        # 4. 반복 패턴
        words = re.findall(r'\b\w{2,}\b', text)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        repetitive_words = [w for w, c in word_counts.items() if c >= REPETITIVE_THRESHOLD]
        if repetitive_words:
            reasons.append(FakeReviewReason.REPETITIVE_PATTERN)
            details["repetitive_words"] = repetitive_words
            scores.append(0.5)

        # 5. 구체성 부족
        has_specifics = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in SPECIFICITY_KEYWORDS
        )
        if not has_specifics and len(text) > 20:
            reasons.append(FakeReviewReason.NO_SPECIFICS)
            scores.append(0.3)

        # 6. 극단적 평점과 내용 불일치
        if rating >= 4.5:
            # 높은 평점인데 부정적 키워드
            negative_patterns = [r"별로", r"아쉽", r"실망", r"후회", r"불만"]
            has_negative = any(
                re.search(p, text, re.IGNORECASE) for p in negative_patterns
            )
            if has_negative:
                reasons.append(FakeReviewReason.EXTREME_RATING)
                details["rating_mismatch"] = "high_rating_negative_content"
                scores.append(0.6)

        elif rating <= 1.5:
            # 낮은 평점인데 긍정적 키워드
            positive_patterns = [r"좋아요", r"만족", r"추천", r"최고"]
            has_positive = any(
                re.search(p, text, re.IGNORECASE) for p in positive_patterns
            )
            if has_positive:
                reasons.append(FakeReviewReason.EXTREME_RATING)
                details["rating_mismatch"] = "low_rating_positive_content"
                scores.append(0.6)

        # 신뢰도 계산
        confidence = max(scores) if scores else 0.0

        return FakeReviewResult(
            is_suspicious=confidence >= 0.5,
            confidence=confidence,
            reasons=reasons,
            details=details,
        )

    def _llm_based_check(self, text: str, rating: float) -> FakeReviewResult:
        """LLM 기반 가짜 리뷰 검사.

        Args:
            text: 리뷰 텍스트
            rating: 평점

        Returns:
            FakeReviewResult 객체
        """
        try:
            llm = self._get_llm()
            structured_llm = llm.with_structured_output(LLMFakeReviewResult)

            prompt = FAKE_REVIEW_DETECTION_PROMPT.format(
                text=text[:500],  # 토큰 절약
                rating=rating,
            )

            result: LLMFakeReviewResult = structured_llm.invoke(prompt)

            return FakeReviewResult(
                is_suspicious=result.is_fake,
                confidence=result.confidence,
                reasons=[FakeReviewReason.LLM_DETECTED] if result.is_fake else [],
                details={"llm_reason": result.reason},
            )

        except Exception as e:
            # LLM 실패 시 빈 결과 반환
            return FakeReviewResult(
                is_suspicious=False,
                confidence=0.0,
                details={"llm_error": str(e)},
            )

    def _calculate_weight(self, result: FakeReviewResult) -> float:
        """의심도에 따른 가중치 계산.

        Args:
            result: 가짜 리뷰 판정 결과

        Returns:
            가중치 (0.0 ~ 1.0, 의심되면 낮음)
        """
        if not result.is_suspicious:
            return 1.0

        # 의심도에 따라 가중치 감소
        # confidence 0.5 → weight 0.75
        # confidence 0.8 → weight 0.4
        # confidence 1.0 → weight 0.2
        weight = max(0.2, 1.0 - (result.confidence * 0.8))

        # 스팸은 더 강하게 감소
        if FakeReviewReason.SPAM_KEYWORDS in result.reasons:
            weight *= 0.5

        return weight

    def filter_reviews(
        self,
        reviews: list[Review],
        remove_suspicious: bool = False,
    ) -> tuple[list[Review], list[FakeReviewResult]]:
        """리뷰 목록 필터링.

        Args:
            reviews: 리뷰 목록
            remove_suspicious: True면 의심 리뷰 제거, False면 가중치만 조절

        Returns:
            (필터링된 리뷰, 판정 결과 목록)
        """
        results = []
        filtered_reviews = []

        for review in reviews:
            result = self.check_review(review)
            results.append(result)

            if remove_suspicious and result.is_suspicious:
                continue

            filtered_reviews.append(review)

        return filtered_reviews, results

    def get_statistics(
        self,
        results: list[FakeReviewResult],
    ) -> dict:
        """필터링 결과 통계.

        Args:
            results: 판정 결과 목록

        Returns:
            통계 딕셔너리
        """
        total = len(results)
        suspicious = sum(1 for r in results if r.is_suspicious)

        reason_counts = {}
        for result in results:
            for reason in result.reasons:
                reason_counts[reason.value] = reason_counts.get(reason.value, 0) + 1

        return {
            "total": total,
            "suspicious": suspicious,
            "suspicious_rate": suspicious / total if total > 0 else 0,
            "clean": total - suspicious,
            "reason_counts": reason_counts,
            "avg_weight": sum(r.weight for r in results) / total if total > 0 else 1.0,
        }


def create_fake_review_filter(
    use_llm: bool = True,
    llm_threshold: float = 0.5,
) -> FakeReviewFilter:
    """가짜 리뷰 필터 생성 헬퍼.

    Args:
        use_llm: LLM 사용 여부
        llm_threshold: LLM 호출 임계값

    Returns:
        FakeReviewFilter 인스턴스
    """
    return FakeReviewFilter(use_llm=use_llm, llm_threshold=llm_threshold)


def check_review_text(text: str, rating: int | None = None) -> FakeReviewResult:
    """텍스트 기반 가짜 리뷰 검사 (규칙 기반만).

    Review 객체 없이 텍스트와 평점만으로 검사할 때 사용.

    Args:
        text: 리뷰 텍스트
        rating: 평점 (1-5, 없으면 None)

    Returns:
        FakeReviewResult 객체
    """
    filter_instance = FakeReviewFilter(use_llm=False)
    result = filter_instance._rule_based_check(text, rating or 3)
    result.weight = filter_instance._calculate_weight(result)
    return result
