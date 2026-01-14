"""SQLite 데이터베이스 - 제품 및 리뷰 통합 관리."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


# DB 파일 경로
DB_PATH = Path(__file__).parent.parent / "data" / "reviews.db"


class ProductRecord(BaseModel):
    """제품 레코드 모델."""
    id: Optional[int] = None
    name: str
    category: str
    main_category: str
    avg_rating: float = 0.0
    review_count: int = 0
    created_at: Optional[str] = None


class ReviewRecord(BaseModel):
    """리뷰 레코드 모델."""
    id: Optional[int] = None
    product_id: int  # FK to products.id
    text: str
    sentiment: str  # 긍정, 중립, 부정
    aspects: list[dict]  # JSON으로 저장
    rating: Optional[int] = None  # 별점 (1-5)
    created_at: Optional[str] = None


def init_db():
    """데이터베이스 초기화."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 제품 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT,
            main_category TEXT,
            avg_rating REAL DEFAULT 0.0,
            review_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 리뷰 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            aspects TEXT,
            rating INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # 인덱스
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_product_id ON reviews(product_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_name ON products(name)")

    conn.commit()
    conn.close()


def get_connection() -> sqlite3.Connection:
    """DB 연결 가져오기."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# 제품 CRUD
# =============================================================================

def get_or_create_product(name: str, category: str = "", main_category: str = "") -> int:
    """제품 조회 또는 생성, product_id 반환."""
    conn = get_connection()
    cursor = conn.cursor()

    # 기존 제품 조회
    cursor.execute("SELECT id FROM products WHERE name = ?", (name,))
    row = cursor.fetchone()

    if row:
        product_id = row["id"]
    else:
        # 새 제품 생성
        cursor.execute("""
            INSERT INTO products (name, category, main_category)
            VALUES (?, ?, ?)
        """, (name, category, main_category))
        product_id = cursor.lastrowid
        conn.commit()

    conn.close()
    return product_id


def update_product_stats(product_id: int):
    """제품의 avg_rating과 review_count 업데이트."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE products SET
            avg_rating = COALESCE((SELECT AVG(rating) FROM reviews WHERE product_id = ? AND rating IS NOT NULL), 0),
            review_count = (SELECT COUNT(*) FROM reviews WHERE product_id = ?)
        WHERE id = ?
    """, (product_id, product_id, product_id))

    conn.commit()
    conn.close()


def get_product_by_name(name: str) -> Optional[ProductRecord]:
    """제품명으로 제품 조회."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return ProductRecord(
            id=row["id"],
            name=row["name"],
            category=row["category"] or "",
            main_category=row["main_category"] or "",
            avg_rating=row["avg_rating"] or 0.0,
            review_count=row["review_count"] or 0,
            created_at=row["created_at"]
        )
    return None


def get_all_products() -> list[ProductRecord]:
    """모든 제품 조회."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products ORDER BY review_count DESC")
    rows = cursor.fetchall()
    conn.close()

    return [
        ProductRecord(
            id=row["id"],
            name=row["name"],
            category=row["category"] or "",
            main_category=row["main_category"] or "",
            avg_rating=row["avg_rating"] or 0.0,
            review_count=row["review_count"] or 0,
            created_at=row["created_at"]
        )
        for row in rows
    ]


# =============================================================================
# 리뷰 CRUD
# =============================================================================

def add_review(product_id: int, text: str, sentiment: str, aspects: list[dict],
               rating: Optional[int] = None, created_at: Optional[str] = None) -> int:
    """리뷰 추가."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reviews (product_id, text, sentiment, aspects, rating, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        product_id,
        text,
        sentiment,
        json.dumps(aspects, ensure_ascii=False),
        rating,
        created_at or datetime.now().isoformat()
    ))

    review_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # 제품 통계 업데이트
    update_product_stats(product_id)

    return review_id


def get_reviews_by_product(product_name: str, sentiment: Optional[str] = None) -> list[dict]:
    """제품명으로 리뷰 조회."""
    conn = get_connection()
    cursor = conn.cursor()

    # 제품 ID 조회
    cursor.execute("SELECT id FROM products WHERE name = ?", (product_name,))
    product_row = cursor.fetchone()

    if not product_row:
        conn.close()
        return []

    product_id = product_row["id"]

    if sentiment and sentiment != "전체":
        cursor.execute(
            "SELECT * FROM reviews WHERE product_id = ? AND sentiment = ? ORDER BY created_at DESC",
            (product_id, sentiment)
        )
    else:
        cursor.execute(
            "SELECT * FROM reviews WHERE product_id = ? ORDER BY created_at DESC",
            (product_id,)
        )

    rows = cursor.fetchall()
    conn.close()

    reviews = []
    for row in rows:
        reviews.append({
            "id": row["id"],
            "product_id": row["product_id"],
            "text": row["text"],
            "sentiment": row["sentiment"],
            "aspects": json.loads(row["aspects"]) if row["aspects"] else [],
            "rating": row["rating"],
            "created_at": row["created_at"]
        })

    return reviews


def get_review_count(product_name: str) -> dict:
    """제품별 리뷰 수 통계."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM products WHERE name = ?", (product_name,))
    product_row = cursor.fetchone()

    if not product_row:
        conn.close()
        return {"긍정": 0, "중립": 0, "부정": 0, "total": 0}

    product_id = product_row["id"]

    cursor.execute("""
        SELECT sentiment, COUNT(*) as count
        FROM reviews
        WHERE product_id = ?
        GROUP BY sentiment
    """, (product_id,))

    rows = cursor.fetchall()
    conn.close()

    stats = {"긍정": 0, "중립": 0, "부정": 0, "total": 0}
    for row in rows:
        stats[row["sentiment"]] = row["count"]
        stats["total"] += row["count"]

    return stats


def get_review_aspects_by_text(text: str) -> list[dict]:
    """리뷰 텍스트로 aspects 조회.

    Args:
        text: 리뷰 텍스트 (부분 매칭)

    Returns:
        aspects 리스트 (없으면 빈 리스트)
    """
    conn = get_connection()
    cursor = conn.cursor()

    # 텍스트 앞부분으로 검색 (100자까지)
    search_text = text[:100] if len(text) > 100 else text

    cursor.execute(
        "SELECT aspects FROM reviews WHERE text LIKE ? LIMIT 1",
        (f"{search_text}%",)
    )
    row = cursor.fetchone()
    conn.close()

    if row and row["aspects"]:
        try:
            return json.loads(row["aspects"])
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def delete_review(review_id: int) -> bool:
    """리뷰 삭제."""
    conn = get_connection()
    cursor = conn.cursor()

    # 삭제 전 product_id 조회
    cursor.execute("SELECT product_id FROM reviews WHERE id = ?", (review_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return False

    product_id = row["product_id"]

    cursor.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    # 제품 통계 업데이트
    if deleted:
        update_product_stats(product_id)

    return deleted


# =============================================================================
# AIHub JSON 마이그레이션
# =============================================================================

def convert_score_to_rating(score: int) -> int:
    """ReviewScore를 5점 만점으로 변환.

    AI Hub 데이터는 두 가지 형식이 혼재:
    - 5점 척도: 1, 2, 3, 4, 5
    - 100점 만점: 10, 20, ..., 100

    5 이하면 이미 5점 척도로 간주, 5 초과면 100점 만점으로 환산.
    """
    if score <= 5:
        # 이미 5점 척도
        return max(1, min(5, score))
    else:
        # 100점 만점 → 5점 환산
        if score >= 80:
            return 5
        elif score >= 60:
            return 4
        elif score >= 40:
            return 3
        elif score >= 20:
            return 2
        else:
            return 1


def migrate_aihub_json(json_dir: str = None) -> dict:
    """AIHub JSON 파일들을 DB로 마이그레이션."""
    if json_dir is None:
        json_dir = Path(__file__).parent.parent / "data" / "aihub_merged"
    else:
        json_dir = Path(json_dir)

    if not json_dir.exists():
        return {"products": 0, "reviews": 0, "error": "디렉토리 없음"}

    polarity_map = {"1": "긍정", "0": "중립", "-1": "부정", 1: "긍정", 0: "중립", -1: "부정"}

    conn = get_connection()
    cursor = conn.cursor()

    # 이미 데이터가 있는지 확인
    cursor.execute("SELECT COUNT(*) FROM reviews")
    existing_count = cursor.fetchone()[0]
    if existing_count > 0:
        conn.close()
        return {"products": 0, "reviews": 0, "message": "이미 마이그레이션됨"}

    product_count = 0
    review_count = 0
    product_cache = {}  # name -> product_id

    # JSON 파일들 순회
    for json_file in json_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                reviews_data = json.load(f)

            for review_data in reviews_data:
                product_name = review_data.get("ProductName", "")
                if not product_name:
                    continue

                # 제품 조회 또는 생성
                if product_name not in product_cache:
                    cursor.execute("SELECT id FROM products WHERE name = ?", (product_name,))
                    row = cursor.fetchone()

                    if row:
                        product_cache[product_name] = row["id"]
                    else:
                        cursor.execute("""
                            INSERT INTO products (name, category, main_category)
                            VALUES (?, ?, ?)
                        """, (
                            product_name,
                            review_data.get("Domain", ""),
                            review_data.get("MainCategory", "")
                        ))
                        product_cache[product_name] = cursor.lastrowid
                        product_count += 1

                product_id = product_cache[product_name]

                # 리뷰 추가
                polarity = review_data.get("GeneralPolarity", "0")
                sentiment = polarity_map.get(polarity, polarity_map.get(int(polarity), "중립"))

                # 점수 변환 (100점 → 5점)
                score = int(review_data.get("ReviewScore", 50))
                rating = convert_score_to_rating(score)

                # 날짜 변환 (20211010 → 2021-10-10)
                rdate = review_data.get("RDate", "")
                if rdate and len(rdate) == 8:
                    created_at = f"{rdate[:4]}-{rdate[4:6]}-{rdate[6:8]}"
                else:
                    created_at = None

                cursor.execute("""
                    INSERT INTO reviews (product_id, text, sentiment, aspects, rating, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    product_id,
                    review_data.get("RawText", ""),
                    sentiment,
                    json.dumps(review_data.get("Aspects", []), ensure_ascii=False),
                    rating,
                    created_at
                ))
                review_count += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    conn.commit()

    # 모든 제품의 통계 업데이트
    for product_id in product_cache.values():
        cursor.execute("""
            UPDATE products SET
                avg_rating = COALESCE((SELECT AVG(rating) FROM reviews WHERE product_id = ? AND rating IS NOT NULL), 0),
                review_count = (SELECT COUNT(*) FROM reviews WHERE product_id = ?)
            WHERE id = ?
        """, (product_id, product_id, product_id))

    conn.commit()
    conn.close()

    return {"products": product_count, "reviews": review_count}


# =============================================================================
# 기존 호환성 (app.py에서 사용)
# =============================================================================

def migrate_aihub_product(product) -> int:
    """기존 Product 객체에서 마이그레이션 (호환성 유지)."""
    # JSON 마이그레이션이 이미 되어있으면 스킵
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM products WHERE name = ?", (product.name,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return 0  # 이미 있음

    # 없으면 개별 마이그레이션
    polarity_map = {1: "긍정", 0: "중립", -1: "부정"}
    product_id = get_or_create_product(
        name=product.name,
        category=product.category,
        main_category=product.main_category
    )

    count = 0
    for review in product.reviews:
        sentiment = polarity_map.get(review.general_polarity, "중립")
        add_review(
            product_id=product_id,
            text=review.raw_text,
            sentiment=sentiment,
            aspects=review.aspects if review.aspects else [],
            rating=None  # 기존 데이터는 rating 없음
        )
        count += 1

    return count
