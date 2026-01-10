"""
Pytest configuration and fixtures.
"""

import pytest


@pytest.fixture
def sample_review():
    """Sample review data for testing."""
    return {
        "text": "배송이 빨라서 좋았어요! 맛도 괜찮습니다.",
        "rating": 5,
        "date": "2024-01-10",
        "option": "기본",
    }


@pytest.fixture
def sample_reviews():
    """Sample reviews list for testing."""
    return [
        {
            "text": "배송이 빨라서 좋았어요! 맛도 괜찮습니다.",
            "rating": 5,
            "date": "2024-01-10",
            "option": "기본",
        },
        {
            "text": "가격 대비 양이 적어요. 포장도 아쉽습니다.",
            "rating": 2,
            "date": "2024-01-09",
            "option": "대용량",
        },
        {
            "text": "재구매 의사 있습니다. 아이들이 잘 먹어요.",
            "rating": 4,
            "date": "2024-01-08",
            "option": "기본",
        },
    ]
