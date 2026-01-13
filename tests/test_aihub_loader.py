"""
AI Hub 데이터 로더 테스트.

AIHubDataLoader 및 관련 클래스 테스트입니다.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.pipeline.aihub_loader import AIHubDataLoader, AIHubReview, Product


# =============================================================================
# AIHubReview 테스트
# =============================================================================


class TestAIHubReview:
    """AIHubReview 테스트."""

    def test_from_dict_full_data(self):
        """딕셔너리에서 AIHubReview 생성 테스트 (전체 데이터)."""
        data = {
            "Index": "12345",
            "RawText": "정말 좋은 제품입니다.",
            "Source": "쇼핑몰",
            "Domain": "패션",
            "MainCategory": "여성의류",
            "ProductName": "니트 스웨터",
            "ReviewScore": "80",
            "GeneralPolarity": 1,
            "Aspects": [
                {"Aspect": "품질", "SentimentText": "좋아요", "SentimentPolarity": 1}
            ],
            "RDate": "20240115",
        }

        review = AIHubReview.from_dict(data)

        assert review.index == "12345"
        assert review.raw_text == "정말 좋은 제품입니다."
        assert review.source == "쇼핑몰"
        assert review.domain == "패션"
        assert review.main_category == "여성의류"
        assert review.product_name == "니트 스웨터"
        assert review.review_score == 80
        assert review.general_polarity == 1
        assert len(review.aspects) == 1
        assert review.date == "20240115"

    def test_from_dict_empty_review_score(self):
        """빈 ReviewScore 처리 테스트."""
        data = {
            "Index": "1",
            "RawText": "테스트",
            "ReviewScore": "",
            "GeneralPolarity": 0,
        }

        review = AIHubReview.from_dict(data)
        assert review.review_score == -1

    def test_from_dict_invalid_review_score(self):
        """잘못된 ReviewScore 처리 테스트."""
        data = {
            "Index": "1",
            "RawText": "테스트",
            "ReviewScore": "invalid",
            "GeneralPolarity": 0,
        }

        review = AIHubReview.from_dict(data)
        assert review.review_score == -1

    def test_from_dict_missing_fields(self):
        """누락된 필드 기본값 테스트."""
        data = {"RawText": "테스트 리뷰"}

        review = AIHubReview.from_dict(data)

        assert review.index == ""
        assert review.raw_text == "테스트 리뷰"
        assert review.source == ""
        assert review.domain == ""
        assert review.main_category == ""
        assert review.product_name == ""
        assert review.review_score == -1
        assert review.general_polarity == 0
        assert review.aspects == []
        assert review.date is None

    def test_to_review_with_score(self):
        """to_review 변환 테스트 (점수 있음)."""
        review = AIHubReview(
            index="1",
            raw_text="좋은 제품입니다",
            source="쇼핑몰",
            domain="패션",
            main_category="여성의류",
            product_name="드레스",
            review_score=80,
            general_polarity=1,
            aspects=[],
            date="20240115",
        )

        project_review = review.to_review()

        assert project_review.text == "좋은 제품입니다"
        assert project_review.rating == 4.0  # 80/20 = 4.0
        assert project_review.date == "2024-01-15"
        assert project_review.verified_purchase is True
        assert project_review.metadata["source"] == "쇼핑몰"
        assert project_review.metadata["polarity_label"] == "긍정"

    def test_to_review_without_score_positive(self):
        """to_review 변환 테스트 (점수 없음, 긍정)."""
        review = AIHubReview(
            index="1",
            raw_text="좋아요",
            source="쇼핑몰",
            domain="패션",
            main_category="여성의류",
            product_name="셔츠",
            review_score=-1,
            general_polarity=1,
            aspects=[],
            date=None,
        )

        project_review = review.to_review()
        assert project_review.rating == 4.0  # 긍정 → 4.0

    def test_to_review_without_score_neutral(self):
        """to_review 변환 테스트 (점수 없음, 중립)."""
        review = AIHubReview(
            index="1",
            raw_text="그냥 그래요",
            source="쇼핑몰",
            domain="패션",
            main_category="여성의류",
            product_name="바지",
            review_score=-1,
            general_polarity=0,
            aspects=[],
            date=None,
        )

        project_review = review.to_review()
        assert project_review.rating == 3.0  # 중립 → 3.0

    def test_to_review_without_score_negative(self):
        """to_review 변환 테스트 (점수 없음, 부정)."""
        review = AIHubReview(
            index="1",
            raw_text="별로예요",
            source="쇼핑몰",
            domain="패션",
            main_category="여성의류",
            product_name="코트",
            review_score=-1,
            general_polarity=-1,
            aspects=[],
            date=None,
        )

        project_review = review.to_review()
        assert project_review.rating == 2.0  # 부정 → 2.0
        assert project_review.metadata["polarity_label"] == "부정"

    def test_to_review_aspects_conversion(self):
        """to_review 속성 변환 테스트."""
        review = AIHubReview(
            index="1",
            raw_text="품질 좋고 배송 빨라요",
            source="쇼핑몰",
            domain="가전",
            main_category="TV",
            product_name="스마트TV",
            review_score=90,
            general_polarity=1,
            aspects=[
                {"Aspect": "품질", "SentimentText": "좋아요", "SentimentPolarity": 1},
                {"Aspect": "배송", "SentimentText": "빨라요", "SentimentPolarity": 1},
            ],
            date="20240120",
        )

        project_review = review.to_review()
        aspects = project_review.metadata["aspects"]

        assert len(aspects) == 2
        assert aspects[0]["aspect"] == "품질"
        assert aspects[0]["sentiment_text"] == "좋아요"
        assert aspects[0]["polarity"] == 1


# =============================================================================
# Product 테스트
# =============================================================================


class TestProduct:
    """Product 테스트."""

    def test_get_sentiment_ratio(self):
        """감정 비율 계산 테스트."""
        product = Product(
            name="테스트 제품",
            category="패션",
            main_category="여성의류",
            review_count=10,
            avg_rating=4.0,
            sentiment_distribution={"긍정": 6, "중립": 3, "부정": 1},
            top_aspects=["품질", "배송"],
            reviews=[],
        )

        ratio = product.get_sentiment_ratio()

        assert ratio["긍정"] == 60.0
        assert ratio["중립"] == 30.0
        assert ratio["부정"] == 10.0

    def test_get_sentiment_ratio_zero_total(self):
        """감정 분포가 0일 때 테스트."""
        product = Product(
            name="테스트 제품",
            category="패션",
            main_category="여성의류",
            review_count=0,
            avg_rating=0.0,
            sentiment_distribution={"긍정": 0, "중립": 0, "부정": 0},
            top_aspects=[],
            reviews=[],
        )

        ratio = product.get_sentiment_ratio()

        assert ratio["긍정"] == 0.0
        assert ratio["중립"] == 0.0
        assert ratio["부정"] == 0.0


# =============================================================================
# AIHubDataLoader 테스트
# =============================================================================


class TestAIHubDataLoader:
    """AIHubDataLoader 테스트."""

    @pytest.fixture
    def flat_data_dir(self, tmp_path):
        """flat 형식 테스트 데이터 디렉토리 생성."""
        reviews_data = [
            {
                "Index": "1",
                "RawText": "좋은 제품이에요",
                "Source": "쇼핑몰",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ProductName": "원피스",
                "ReviewScore": "80",
                "GeneralPolarity": 1,
                "Aspects": [],
            },
            {
                "Index": "2",
                "RawText": "별로예요",
                "Source": "SNS",
                "Domain": "패션",
                "MainCategory": "남성의류",
                "ProductName": "티셔츠",
                "ReviewScore": "40",
                "GeneralPolarity": -1,
                "Aspects": [],
            },
        ]

        json_file = tmp_path / "1-1.여성의류.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, ensure_ascii=False)

        return tmp_path

    @pytest.fixture
    def category_data_dir(self, tmp_path):
        """category 형식 테스트 데이터 디렉토리 생성."""
        # 카테고리 폴더 생성
        cat_dir = tmp_path / "1-1.여성의류"
        cat_dir.mkdir()

        reviews_data = [
            {
                "Index": "1",
                "RawText": "예쁜 원피스",
                "Source": "쇼핑몰",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ProductName": "플로럴 원피스",
                "ReviewScore": "90",
                "GeneralPolarity": 1,
                "Aspects": [],
            },
        ]

        json_file = cat_dir / "reviews.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, ensure_ascii=False)

        return tmp_path

    @pytest.fixture
    def original_data_dir(self, tmp_path):
        """original 형식 테스트 데이터 디렉토리 생성."""
        # Training/02.라벨링데이터/TL_01_... 구조
        training_dir = tmp_path / "Training" / "02.라벨링데이터" / "TL_01_패션"
        training_dir.mkdir(parents=True)

        reviews_data = [
            {
                "Index": "1",
                "RawText": "좋아요",
                "Source": "쇼핑몰",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ProductName": "블라우스",
                "ReviewScore": "85",
                "GeneralPolarity": 1,
                "Aspects": [],
            },
        ]

        json_file = training_dir / "reviews.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, ensure_ascii=False)

        return tmp_path

    def test_detect_format_flat(self, flat_data_dir):
        """flat 형식 감지 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        assert loader._is_merged_format == "flat"

    def test_detect_format_category(self, category_data_dir):
        """category 형식 감지 테스트."""
        loader = AIHubDataLoader(category_data_dir)
        assert loader._is_merged_format == "category"

    def test_detect_format_original(self, original_data_dir):
        """original 형식 감지 테스트."""
        loader = AIHubDataLoader(original_data_dir)
        assert loader._is_merged_format == "original"

    def test_invalid_directory(self):
        """존재하지 않는 디렉토리 테스트."""
        with pytest.raises(FileNotFoundError):
            AIHubDataLoader("/nonexistent/path")

    def test_load_json_file(self, flat_data_dir):
        """JSON 파일 로드 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        json_file = flat_data_dir / "1-1.여성의류.json"
        reviews = loader.load_json_file(json_file)

        assert len(reviews) == 2
        assert reviews[0].raw_text == "좋은 제품이에요"
        assert reviews[1].raw_text == "별로예요"

    def test_load_json_file_invalid(self, tmp_path):
        """잘못된 JSON 파일 로드 테스트."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not json")

        # flat 형식을 위해 정상 파일도 생성
        valid_file = tmp_path / "1-1.test.json"
        with open(valid_file, "w", encoding="utf-8") as f:
            json.dump([{"RawText": "test"}], f)

        loader = AIHubDataLoader(tmp_path)
        reviews = loader.load_json_file(invalid_file)

        assert reviews == []

    def test_iter_reviews_flat(self, flat_data_dir):
        """flat 형식 iter_reviews 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        reviews = list(loader.iter_reviews())

        assert len(reviews) == 2

    def test_iter_reviews_with_limit(self, flat_data_dir):
        """iter_reviews limit 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        reviews = list(loader.iter_reviews(limit=1))

        assert len(reviews) == 1

    def test_iter_reviews_category(self, category_data_dir):
        """category 형식 iter_reviews 테스트."""
        loader = AIHubDataLoader(category_data_dir)
        reviews = list(loader.iter_reviews())

        assert len(reviews) == 1
        assert reviews[0].product_name == "플로럴 원피스"

    def test_load_reviews_as_project_format(self, flat_data_dir):
        """load_reviews Project Review 형식 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        reviews = loader.load_reviews(as_project_format=True)

        assert len(reviews) == 2
        assert reviews[0].text == "좋은 제품이에요"
        assert reviews[0].rating == 4.0  # 80/20

    def test_load_reviews_as_aihub_format(self, flat_data_dir):
        """load_reviews AIHubReview 형식 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        reviews = loader.load_reviews(as_project_format=False)

        assert len(reviews) == 2
        assert isinstance(reviews[0], AIHubReview)

    def test_extract_category_from_filename(self, flat_data_dir):
        """파일명에서 카테고리 추출 테스트."""
        loader = AIHubDataLoader(flat_data_dir)

        assert loader._extract_category_from_filename("1-1.여성의류.json") == "패션"
        assert loader._extract_category_from_filename("2-1.스킨케어.json") == "화장품"
        assert loader._extract_category_from_filename("3-1.냉장고.json") == "가전"
        assert loader._extract_category_from_filename("4-1.스마트폰.json") == "IT기기"
        assert loader._extract_category_from_filename("5-1.주방용품.json") == "생활용품"
        assert loader._extract_category_from_filename("unknown.json") is None

    def test_extract_source_from_filename(self, flat_data_dir):
        """파일명에서 출처 추출 테스트."""
        loader = AIHubDataLoader(flat_data_dir)

        assert loader._extract_source_from_filename("1-1.여성의류.json") == "쇼핑몰"
        assert loader._extract_source_from_filename("1-1.여성의류_SNS.json") == "SNS"

    def test_get_statistics(self, flat_data_dir):
        """통계 조회 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        stats = loader.get_statistics()

        assert stats["total"] == 2
        assert "패션" in stats["by_category"]
        assert stats["by_source"]["쇼핑몰"] == 1
        assert stats["by_source"]["SNS"] == 1
        assert stats["by_polarity"]["긍정"] == 1
        assert stats["by_polarity"]["부정"] == 1

    def test_get_products(self, tmp_path):
        """제품 목록 조회 테스트."""
        # 동일 제품에 대한 여러 리뷰
        reviews_data = [
            {
                "Index": "1",
                "RawText": "좋아요",
                "ProductName": "테스트 제품",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ReviewScore": "80",
                "GeneralPolarity": 1,
                "Source": "쇼핑몰",
                "Aspects": [{"Aspect": "품질", "SentimentText": "좋아요", "SentimentPolarity": 1}],
            },
            {
                "Index": "2",
                "RawText": "그냥 그래요",
                "ProductName": "테스트 제품",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ReviewScore": "60",
                "GeneralPolarity": 0,
                "Source": "쇼핑몰",
                "Aspects": [{"Aspect": "품질", "SentimentText": "보통", "SentimentPolarity": 0}],
            },
            {
                "Index": "3",
                "RawText": "별로예요",
                "ProductName": "테스트 제품",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ReviewScore": "40",
                "GeneralPolarity": -1,
                "Source": "쇼핑몰",
                "Aspects": [{"Aspect": "배송", "SentimentText": "느려요", "SentimentPolarity": -1}],
            },
        ]

        json_file = tmp_path / "1-1.test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, ensure_ascii=False)

        loader = AIHubDataLoader(tmp_path)
        products = loader.get_products(min_reviews=1)

        assert len(products) == 1
        product = products[0]
        assert product.name == "테스트 제품"
        assert product.review_count == 3
        assert product.avg_rating == 3.0  # (80+60+40)/3/20 = 3.0
        assert product.sentiment_distribution == {"긍정": 1, "중립": 1, "부정": 1}
        assert "품질" in product.top_aspects

    def test_get_products_min_reviews_filter(self, flat_data_dir):
        """제품 목록 최소 리뷰 수 필터 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        products = loader.get_products(min_reviews=10)

        assert len(products) == 0

    def test_get_product_by_name(self, tmp_path):
        """제품명으로 조회 테스트."""
        reviews_data = [
            {
                "Index": "1",
                "RawText": "좋아요",
                "ProductName": "특정 제품",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ReviewScore": "80",
                "GeneralPolarity": 1,
                "Source": "쇼핑몰",
                "Aspects": [],
            },
        ]

        json_file = tmp_path / "1-1.test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, ensure_ascii=False)

        loader = AIHubDataLoader(tmp_path)
        product = loader.get_product_by_name("특정 제품")

        assert product is not None
        assert product.name == "특정 제품"

    def test_get_product_by_name_not_found(self, flat_data_dir):
        """존재하지 않는 제품 조회 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        product = loader.get_product_by_name("존재하지 않는 제품")

        assert product is None

    def test_export_sample(self, flat_data_dir, tmp_path):
        """샘플 데이터 추출 테스트."""
        loader = AIHubDataLoader(flat_data_dir)
        output_path = tmp_path / "sample.json"

        count = loader.export_sample(output_path, sample_size=10, stratified=False)

        assert count == 2
        assert output_path.exists()

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_reviews"] == 2
        assert len(data["reviews"]) == 2

    def test_extract_category_from_folder(self, original_data_dir):
        """폴더명에서 카테고리 추출 테스트."""
        loader = AIHubDataLoader(original_data_dir)

        assert loader._extract_category_from_folder("TL_01.패션") == "패션"
        assert loader._extract_category_from_folder("TL_02.화장품") == "화장품"
        assert loader._extract_category_from_folder("TL_03.가전") == "가전"
        assert loader._extract_category_from_folder("TL_04.IT기기") == "IT기기"
        assert loader._extract_category_from_folder("TL_05.생활용품") == "생활용품"
        assert loader._extract_category_from_folder("Unknown") is None

    def test_extract_source_from_folder(self, original_data_dir):
        """폴더명에서 출처 추출 테스트."""
        loader = AIHubDataLoader(original_data_dir)

        assert loader._extract_source_from_folder("TL_01.패션") == "쇼핑몰"
        assert loader._extract_source_from_folder("TL_01.패션_SNS") == "SNS"


# =============================================================================
# 엣지 케이스 테스트
# =============================================================================


class TestEdgeCases:
    """엣지 케이스 테스트."""

    def test_review_score_zero(self):
        """ReviewScore가 0인 경우 테스트."""
        data = {
            "Index": "1",
            "RawText": "테스트",
            "ReviewScore": "0",
            "GeneralPolarity": 0,
        }

        review = AIHubReview.from_dict(data)
        assert review.review_score == 0

        project_review = review.to_review()
        assert project_review.rating == 0.0

    def test_review_score_100(self):
        """ReviewScore가 100인 경우 테스트."""
        review = AIHubReview(
            index="1",
            raw_text="최고",
            source="쇼핑몰",
            domain="패션",
            main_category="여성의류",
            product_name="명품백",
            review_score=100,
            general_polarity=1,
            aspects=[],
            date=None,
        )

        project_review = review.to_review()
        assert project_review.rating == 5.0

    def test_date_format_short(self):
        """짧은 날짜 형식 테스트."""
        review = AIHubReview(
            index="1",
            raw_text="테스트",
            source="쇼핑몰",
            domain="패션",
            main_category="여성의류",
            product_name="셔츠",
            review_score=70,
            general_polarity=1,
            aspects=[],
            date="2024",  # 8자가 아닌 경우
        )

        project_review = review.to_review()
        assert project_review.date is None

    def test_empty_product_name(self, tmp_path):
        """빈 제품명 처리 테스트."""
        reviews_data = [
            {
                "Index": "1",
                "RawText": "좋아요",
                "ProductName": "",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ReviewScore": "80",
                "GeneralPolarity": 1,
                "Source": "쇼핑몰",
                "Aspects": [],
            },
            {
                "Index": "2",
                "RawText": "보통",
                "ProductName": "   ",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ReviewScore": "60",
                "GeneralPolarity": 0,
                "Source": "쇼핑몰",
                "Aspects": [],
            },
        ]

        json_file = tmp_path / "1-1.test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, ensure_ascii=False)

        loader = AIHubDataLoader(tmp_path)
        products = loader.get_products(min_reviews=1)

        # 빈 제품명은 무시됨
        assert len(products) == 0

    def test_polarity_based_rating_calculation(self, tmp_path):
        """극성 기반 평점 계산 테스트."""
        reviews_data = [
            {
                "Index": "1",
                "RawText": "좋아요",
                "ProductName": "제품A",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ReviewScore": "",  # 빈 점수
                "GeneralPolarity": 1,  # 긍정
                "Source": "쇼핑몰",
                "Aspects": [],
            },
            {
                "Index": "2",
                "RawText": "보통",
                "ProductName": "제품A",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ReviewScore": "",  # 빈 점수
                "GeneralPolarity": 0,  # 중립
                "Source": "쇼핑몰",
                "Aspects": [],
            },
            {
                "Index": "3",
                "RawText": "별로",
                "ProductName": "제품A",
                "Domain": "패션",
                "MainCategory": "여성의류",
                "ReviewScore": "",  # 빈 점수
                "GeneralPolarity": -1,  # 부정
                "Source": "쇼핑몰",
                "Aspects": [],
            },
        ]

        json_file = tmp_path / "1-1.test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, ensure_ascii=False)

        loader = AIHubDataLoader(tmp_path)
        products = loader.get_products(min_reviews=1)

        assert len(products) == 1
        # (4.0 + 3.0 + 2.0) / 3 = 3.0
        assert products[0].avg_rating == 3.0
