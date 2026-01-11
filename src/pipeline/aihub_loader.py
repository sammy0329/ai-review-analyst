"""
AI Hub 속성기반 감정분석 데이터 로더.

AI Hub 데이터셋을 프로젝트의 Review 형식으로 변환합니다.
데이터셋: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71603
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.crawler.base import Review


@dataclass
class Product:
    """제품 정보."""

    name: str
    category: str
    main_category: str
    review_count: int
    avg_rating: float
    sentiment_distribution: dict[str, int]  # {"긍정": n, "중립": n, "부정": n}
    top_aspects: list[str]  # 자주 언급되는 속성들
    reviews: list["AIHubReview"] = field(default_factory=list)

    def get_sentiment_ratio(self) -> dict[str, float]:
        """감정 비율 반환."""
        total = sum(self.sentiment_distribution.values())
        if total == 0:
            return {"긍정": 0.0, "중립": 0.0, "부정": 0.0}
        return {k: v / total * 100 for k, v in self.sentiment_distribution.items()}


@dataclass
class AIHubReview:
    """AI Hub 원본 리뷰 데이터 구조."""

    index: str
    raw_text: str
    source: str  # 쇼핑몰 or SNS
    domain: str  # 패션, 화장품, 가전, IT기기, 생활용품
    main_category: str  # 상세 카테고리
    product_name: str
    review_score: int  # 100점 만점
    general_polarity: int  # -1, 0, 1
    aspects: list[dict[str, Any]] = field(default_factory=list)
    date: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AIHubReview":
        """딕셔너리에서 AIHubReview 객체 생성."""
        # ReviewScore 파싱 (문자열 또는 빈 문자열 처리)
        review_score_raw = data.get("ReviewScore", "")
        if review_score_raw and str(review_score_raw).strip():
            try:
                review_score = int(review_score_raw)
            except ValueError:
                review_score = -1  # 파싱 실패 시 -1 (알 수 없음)
        else:
            review_score = -1  # 빈 값이면 -1 (알 수 없음)

        return cls(
            index=str(data.get("Index", "")),
            raw_text=data.get("RawText", ""),
            source=data.get("Source", ""),
            domain=data.get("Domain", ""),
            main_category=data.get("MainCategory", ""),
            product_name=data.get("ProductName", ""),
            review_score=review_score,
            general_polarity=int(data.get("GeneralPolarity", 0)),
            aspects=data.get("Aspects", []),
            date=data.get("RDate"),
        )

    def to_review(self) -> Review:
        """프로젝트 Review 형식으로 변환."""
        # 100점 만점을 5점 만점으로 변환
        # -1은 알 수 없음을 의미, 0~100은 유효한 점수
        if self.review_score >= 0:
            rating = self.review_score / 20  # 0~100 → 0.0~5.0
        else:
            # review_score가 없으면(-1) general_polarity 기반으로 평점 추정
            # -1 (부정) -> 2.0, 0 (중립) -> 3.0, 1 (긍정) -> 4.0
            polarity_rating_map = {-1: 2.0, 0: 3.0, 1: 4.0}
            rating = polarity_rating_map.get(self.general_polarity, 3.0)

        # 날짜 포맷 변환 (YYYYMMDD -> YYYY-MM-DD)
        formatted_date = None
        if self.date and len(self.date) == 8:
            formatted_date = f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:]}"

        # 감정 극성 문자열 변환
        polarity_map = {1: "긍정", 0: "중립", -1: "부정"}
        polarity_str = polarity_map.get(self.general_polarity, "중립")

        # 속성별 감정 정보를 메타데이터에 저장
        aspects_data = []
        for aspect in self.aspects:
            aspects_data.append({
                "aspect": aspect.get("Aspect", ""),
                "sentiment_text": aspect.get("SentimentText", ""),
                "polarity": aspect.get("SentimentPolarity", 0),
            })

        return Review(
            text=self.raw_text,
            rating=rating,
            date=formatted_date,
            author=None,  # AI Hub 데이터에는 작성자 정보 없음
            option=None,
            helpful_count=None,
            verified_purchase=True,  # 쇼핑몰 리뷰는 구매 인증으로 간주
            images=[],
            metadata={
                "aihub_index": self.index,
                "source": self.source,
                "domain": self.domain,
                "main_category": self.main_category,
                "product_name": self.product_name,
                "general_polarity": self.general_polarity,
                "polarity_label": polarity_str,
                "aspects": aspects_data,
                "original_score": self.review_score,
            },
        )


class AIHubDataLoader:
    """AI Hub 데이터셋 로더."""

    # 카테고리 번호 매핑
    CATEGORY_MAP = {
        "01": "패션",
        "02": "화장품",
        "03": "가전",
        "04": "IT기기",
        "05": "생활용품",
    }

    def __init__(self, data_dir: str | Path):
        """
        초기화.

        Args:
            data_dir: AI Hub 데이터 디렉토리 경로
                      (Training/02.라벨링데이터, Validation/02.라벨링데이터 포함)
        """
        self.data_dir = Path(data_dir)
        self._validate_data_dir()

    def _validate_data_dir(self) -> None:
        """데이터 디렉토리 유효성 검증."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"데이터 디렉토리가 존재하지 않습니다: {self.data_dir}")

        # Training 또는 Validation 폴더 확인
        has_training = (self.data_dir / "Training").exists()
        has_validation = (self.data_dir / "Validation").exists()

        if not has_training and not has_validation:
            raise FileNotFoundError(
                f"Training 또는 Validation 폴더가 없습니다: {self.data_dir}"
            )

    def _get_label_dirs(self, split: str = "all") -> list[Path]:
        """라벨링 데이터 디렉토리 목록 반환."""
        dirs = []

        if split in ("all", "training"):
            training_dir = self.data_dir / "Training" / "02.라벨링데이터"
            if training_dir.exists():
                dirs.append(training_dir)

        if split in ("all", "validation"):
            validation_dir = self.data_dir / "Validation" / "02.라벨링데이터"
            if validation_dir.exists():
                dirs.append(validation_dir)

        return dirs

    def _extract_category_from_folder(self, folder_name: str) -> str | None:
        """폴더명에서 카테고리 추출."""
        for num, category in self.CATEGORY_MAP.items():
            if f"_{num}." in folder_name or f"_{num}_" in folder_name:
                return category
        return None

    def _extract_source_from_folder(self, folder_name: str) -> str:
        """폴더명에서 출처 추출."""
        return "SNS" if "SNS" in folder_name else "쇼핑몰"

    def load_json_file(self, file_path: Path) -> list[AIHubReview]:
        """단일 JSON 파일 로드."""
        reviews = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    try:
                        review = AIHubReview.from_dict(item)
                        reviews.append(review)
                    except Exception:
                        continue
        except Exception as e:
            print(f"파일 로드 실패: {file_path} - {e}")

        return reviews

    def iter_reviews(
        self,
        split: str = "all",
        category: str | None = None,
        source: str | None = None,
        limit: int | None = None,
    ) -> Iterator[AIHubReview]:
        """
        리뷰를 순회하며 반환.

        Args:
            split: 데이터 분할 ("all", "training", "validation")
            category: 카테고리 필터 ("패션", "화장품", "가전", "IT기기", "생활용품")
            source: 출처 필터 ("쇼핑몰", "SNS")
            limit: 최대 반환 개수

        Yields:
            AIHubReview 객체
        """
        count = 0
        label_dirs = self._get_label_dirs(split)

        for label_dir in label_dirs:
            for folder in sorted(label_dir.iterdir()):
                if not folder.is_dir():
                    continue

                folder_name = folder.name

                # 카테고리 필터링
                if category:
                    folder_category = self._extract_category_from_folder(folder_name)
                    if folder_category != category:
                        continue

                # 출처 필터링
                if source:
                    folder_source = self._extract_source_from_folder(folder_name)
                    if folder_source != source:
                        continue

                # JSON 파일 순회
                for json_file in sorted(folder.glob("*.json")):
                    reviews = self.load_json_file(json_file)

                    for review in reviews:
                        yield review
                        count += 1

                        if limit and count >= limit:
                            return

    def load_reviews(
        self,
        split: str = "all",
        category: str | None = None,
        source: str | None = None,
        limit: int | None = None,
        as_project_format: bool = True,
    ) -> list[Review] | list[AIHubReview]:
        """
        리뷰 로드.

        Args:
            split: 데이터 분할 ("all", "training", "validation")
            category: 카테고리 필터
            source: 출처 필터
            limit: 최대 반환 개수
            as_project_format: True면 Review 형식, False면 AIHubReview 형식

        Returns:
            Review 또는 AIHubReview 객체 리스트
        """
        reviews = list(self.iter_reviews(split, category, source, limit))

        if as_project_format:
            return [r.to_review() for r in reviews]
        return reviews

    def get_statistics(self) -> dict[str, Any]:
        """데이터셋 통계 반환."""
        stats = {
            "total": 0,
            "by_category": {},
            "by_source": {"쇼핑몰": 0, "SNS": 0},
            "by_polarity": {"긍정": 0, "중립": 0, "부정": 0},
            "by_split": {"training": 0, "validation": 0},
        }

        polarity_map = {1: "긍정", 0: "중립", -1: "부정"}

        for split in ["training", "validation"]:
            for review in self.iter_reviews(split=split):
                stats["total"] += 1
                stats["by_split"][split] += 1

                # 카테고리별
                domain = review.domain
                if domain not in stats["by_category"]:
                    stats["by_category"][domain] = 0
                stats["by_category"][domain] += 1

                # 출처별
                if review.source in stats["by_source"]:
                    stats["by_source"][review.source] += 1

                # 감정별
                polarity_label = polarity_map.get(review.general_polarity, "중립")
                stats["by_polarity"][polarity_label] += 1

        return stats

    def export_sample(
        self,
        output_path: str | Path,
        sample_size: int = 1000,
        category: str | None = None,
        stratified: bool = True,
    ) -> int:
        """
        샘플 데이터 추출 및 저장.

        Args:
            output_path: 출력 파일 경로
            sample_size: 샘플 크기
            category: 특정 카테고리만 추출
            stratified: 카테고리별 균등 샘플링 여부

        Returns:
            저장된 리뷰 개수
        """
        output_path = Path(output_path)

        if stratified and not category:
            # 카테고리별 균등 샘플링
            per_category = sample_size // len(self.CATEGORY_MAP)
            reviews = []

            for cat in self.CATEGORY_MAP.values():
                cat_reviews = self.load_reviews(
                    category=cat,
                    limit=per_category,
                    as_project_format=True,
                )
                reviews.extend(cat_reviews)
        else:
            reviews = self.load_reviews(
                category=category,
                limit=sample_size,
                as_project_format=True,
            )

        # JSON으로 저장
        output_data = {
            "total_reviews": len(reviews),
            "sample_info": {
                "sample_size": sample_size,
                "category": category,
                "stratified": stratified,
            },
            "reviews": [
                {
                    "text": r.text,
                    "rating": r.rating,
                    "date": r.date,
                    "verified_purchase": r.verified_purchase,
                    "metadata": r.metadata,
                }
                for r in reviews
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return len(reviews)

    def get_products(
        self,
        split: str = "all",
        category: str | None = None,
        source: str | None = None,
        min_reviews: int = 3,
        limit: int | None = None,
    ) -> list[Product]:
        """
        제품 목록 조회 (리뷰를 제품별로 그룹화).

        Args:
            split: 데이터 분할 ("all", "training", "validation")
            category: 카테고리 필터
            source: 출처 필터
            min_reviews: 최소 리뷰 수 (이 이상인 제품만 반환)
            limit: 최대 반환 제품 수

        Returns:
            Product 객체 리스트 (리뷰 수 내림차순 정렬)
        """
        from collections import Counter, defaultdict

        # 제품별 리뷰 그룹화
        product_reviews: dict[str, list[AIHubReview]] = defaultdict(list)
        product_info: dict[str, dict] = {}

        for review in self.iter_reviews(split, category, source):
            product_name = review.product_name.strip()
            if not product_name:
                continue

            product_reviews[product_name].append(review)

            # 제품 정보 저장 (첫 리뷰 기준)
            if product_name not in product_info:
                product_info[product_name] = {
                    "category": review.domain,
                    "main_category": review.main_category,
                }

        # Product 객체 생성
        products = []
        polarity_map = {1: "긍정", 0: "중립", -1: "부정"}

        for name, reviews in product_reviews.items():
            if len(reviews) < min_reviews:
                continue

            # 평균 평점 계산 (review_score > 0만 유효, 0은 미입력으로 간주)
            valid_scores = [r.review_score for r in reviews if r.review_score > 0]
            if valid_scores:
                max_score = max(valid_scores)
                avg_score = sum(valid_scores) / len(valid_scores)
                # 점수 범위 자동 감지: max > 5면 100점 만점, 아니면 5점 만점
                if max_score > 5:
                    avg_rating = avg_score / 20  # 100점 -> 5점
                else:
                    avg_rating = avg_score  # 이미 5점 만점
            else:
                # review_score 없으면 polarity 기반 추정
                polarity_scores = []
                for r in reviews:
                    if r.general_polarity == 1:
                        polarity_scores.append(4.0)
                    elif r.general_polarity == -1:
                        polarity_scores.append(2.0)
                    else:
                        polarity_scores.append(3.0)
                avg_rating = sum(polarity_scores) / len(polarity_scores)

            # 감정 분포
            sentiment_dist = {"긍정": 0, "중립": 0, "부정": 0}
            for r in reviews:
                label = polarity_map.get(r.general_polarity, "중립")
                sentiment_dist[label] += 1

            # 자주 언급되는 속성 (상위 5개)
            aspect_counter: Counter = Counter()
            for r in reviews:
                for aspect in r.aspects:
                    aspect_name = aspect.get("Aspect", "")
                    if aspect_name:
                        aspect_counter[aspect_name] += 1
            top_aspects = [a[0] for a in aspect_counter.most_common(5)]

            info = product_info[name]
            product = Product(
                name=name,
                category=info["category"],
                main_category=info["main_category"],
                review_count=len(reviews),
                avg_rating=avg_rating,
                sentiment_distribution=sentiment_dist,
                top_aspects=top_aspects,
                reviews=reviews,
            )
            products.append(product)

        # 리뷰 수 기준 내림차순 정렬
        products.sort(key=lambda p: p.review_count, reverse=True)

        if limit:
            products = products[:limit]

        return products

    def get_product_by_name(
        self,
        product_name: str,
        split: str = "all",
    ) -> Product | None:
        """
        제품명으로 제품 조회.

        Args:
            product_name: 제품명
            split: 데이터 분할

        Returns:
            Product 객체 또는 None
        """
        products = self.get_products(split=split, min_reviews=1)
        for product in products:
            if product.name == product_name:
                return product
        return None


def main():
    """테스트 실행."""
    import sys

    # 데이터 디렉토리 경로
    data_dir = Path(__file__).parent.parent.parent / "data" / "aihub_data"

    if not data_dir.exists():
        print(f"데이터 디렉토리가 없습니다: {data_dir}")
        sys.exit(1)

    loader = AIHubDataLoader(data_dir)

    # 통계 출력
    print("=== AI Hub 데이터 통계 ===\n")
    stats = loader.get_statistics()
    print(f"전체 리뷰 수: {stats['total']:,}개\n")

    print("카테고리별:")
    for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,}개")

    print("\n출처별:")
    for src, count in stats["by_source"].items():
        print(f"  {src}: {count:,}개")

    print("\n감정별:")
    for pol, count in stats["by_polarity"].items():
        print(f"  {pol}: {count:,}개")

    # 샘플 데이터 로드 테스트
    print("\n=== 샘플 리뷰 (5개) ===\n")
    sample_reviews = loader.load_reviews(limit=5, as_project_format=True)

    for i, review in enumerate(sample_reviews, 1):
        print(f"[{i}] {review.text[:100]}...")
        print(f"    평점: {review.rating:.1f}/5.0")
        print(f"    카테고리: {review.metadata['domain']} > {review.metadata['main_category']}")
        print(f"    감정: {review.metadata['polarity_label']}")
        print()


if __name__ == "__main__":
    main()
