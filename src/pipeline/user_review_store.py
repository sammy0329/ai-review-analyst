"""
사용자 리뷰 저장소.

사용자가 추가한 리뷰를 JSON 파일로 저장하고 관리합니다.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class UserReview:
    """사용자 추가 리뷰."""

    id: str
    product_name: str
    text: str
    created_at: str

    # 사용자 평점 (1-5)
    rating: int = 0

    # LLM 분석 결과
    overall_sentiment: str = "neutral"  # positive, negative, neutral
    confidence: float = 0.0
    aspects: list[dict[str, Any]] = field(default_factory=list)

    # 메타데이터
    source: str = "user"
    analyzed: bool = False

    @classmethod
    def create(cls, product_name: str, text: str, rating: int = 0) -> "UserReview":
        """새 리뷰 생성."""
        return cls(
            id=str(uuid.uuid4())[:8],
            product_name=product_name,
            text=text,
            created_at=datetime.now().isoformat(),
            rating=rating,
        )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserReview":
        """딕셔너리에서 생성."""
        return cls(**data)

    def to_aihub_format(self) -> dict[str, Any]:
        """AI Hub 리뷰 형식으로 변환 (UI 통합용)."""
        # AI Hub 라벨링 데이터의 aspect 형식으로 변환
        aihub_aspects = []
        for aspect in self.aspects:
            sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
            aihub_aspects.append({
                "Aspect": aspect.get("category", ""),
                "SentimentText": aspect.get("text", ""),
                "SentimentPolarity": sentiment_map.get(aspect.get("sentiment", "neutral"), 0),
            })

        polarity_map = {"positive": 1, "negative": -1, "neutral": 0}

        return {
            "raw_text": self.text,
            "general_polarity": polarity_map.get(self.overall_sentiment, 0),
            "aspects": aihub_aspects,
            "source": "user",
            "user_review_id": self.id,
            "created_at": self.created_at,
        }


class UserReviewStore:
    """사용자 리뷰 저장소."""

    def __init__(self, store_path: str | Path = "data/user_reviews"):
        """
        초기화.

        Args:
            store_path: 리뷰 저장 디렉토리
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._reviews: dict[str, list[UserReview]] = {}
        self._load_all()

    def _get_product_file(self, product_name: str) -> Path:
        """제품별 저장 파일 경로."""
        # 파일명에 사용 불가능한 문자 제거
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in product_name)
        safe_name = safe_name[:50]  # 파일명 길이 제한
        return self.store_path / f"{safe_name}.json"

    def _load_all(self) -> None:
        """모든 리뷰 로드."""
        for file_path in self.store_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    product_name = data.get("product_name", "")
                    reviews = [UserReview.from_dict(r) for r in data.get("reviews", [])]
                    self._reviews[product_name] = reviews
            except Exception:
                continue

    def _save_product(self, product_name: str) -> None:
        """특정 제품의 리뷰 저장."""
        file_path = self._get_product_file(product_name)
        reviews = self._reviews.get(product_name, [])

        data = {
            "product_name": product_name,
            "updated_at": datetime.now().isoformat(),
            "reviews": [r.to_dict() for r in reviews],
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_review(self, review: UserReview) -> None:
        """리뷰 추가."""
        product_name = review.product_name

        if product_name not in self._reviews:
            self._reviews[product_name] = []

        self._reviews[product_name].append(review)
        self._save_product(product_name)

    def get_reviews(self, product_name: str) -> list[UserReview]:
        """특정 제품의 리뷰 조회."""
        return self._reviews.get(product_name, [])

    def get_review_count(self, product_name: str) -> int:
        """특정 제품의 리뷰 수."""
        return len(self._reviews.get(product_name, []))

    def delete_review(self, product_name: str, review_id: str) -> bool:
        """리뷰 삭제."""
        if product_name not in self._reviews:
            return False

        original_len = len(self._reviews[product_name])
        self._reviews[product_name] = [
            r for r in self._reviews[product_name] if r.id != review_id
        ]

        if len(self._reviews[product_name]) < original_len:
            self._save_product(product_name)
            return True

        return False

    def update_review_analysis(
        self,
        product_name: str,
        review_id: str,
        overall_sentiment: str,
        confidence: float,
        aspects: list[dict[str, Any]],
    ) -> bool:
        """리뷰 분석 결과 업데이트."""
        reviews = self._reviews.get(product_name, [])

        for review in reviews:
            if review.id == review_id:
                review.overall_sentiment = overall_sentiment
                review.confidence = confidence
                review.aspects = aspects
                review.analyzed = True
                self._save_product(product_name)
                return True

        return False


def create_user_review_store(store_path: str | Path = "data/user_reviews") -> UserReviewStore:
    """UserReviewStore 팩토리 함수."""
    return UserReviewStore(store_path=store_path)
