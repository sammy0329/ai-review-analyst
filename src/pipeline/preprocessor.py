"""
리뷰 데이터 전처리 파이프라인.

텍스트 정제, 청킹, 메타데이터 추출, 중복 제거 등의 전처리 기능을 제공합니다.
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.crawler.base import Review


# 이모지 패턴 (Unicode Emoji ranges)
# 주의: 범위가 한글 영역(U+AC00-U+D7A3)과 겹치지 않도록 분리
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U000024FF"  # enclosed alphanumerics (subset, avoiding Korean range)
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002300-\U000023FF"  # misc technical
    "\U0001F200-\U0001F251"  # enclosed ideographic supplement
    "]+",
    flags=re.UNICODE,
)

# 특수문자 패턴 (한글, 영문, 숫자, 기본 문장부호 제외)
SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9.,!?~\-()\"':\n]")

# 반복 문자 패턴 (예: ㅋㅋㅋㅋㅋ, ㅎㅎㅎㅎ, !!!!)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")

# 연속 공백 패턴
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")

# HTML 엔티티 패턴
HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#\d+;")


@dataclass
class TextChunk:
    """텍스트 청크 데이터 구조."""

    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedReview:
    """전처리된 리뷰 데이터 구조."""

    original_text: str
    cleaned_text: str
    chunks: list[TextChunk]
    text_hash: str
    rating: float | None
    date: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class TextCleaner:
    """텍스트 정제 클래스."""

    def __init__(
        self,
        remove_emojis: bool = False,
        remove_special_chars: bool = True,
        normalize_repeated_chars: bool = True,
        max_repeated_chars: int = 2,
        remove_html_entities: bool = True,
        normalize_whitespace: bool = True,
        lowercase: bool = False,
    ):
        """
        초기화.

        Args:
            remove_emojis: 이모지 제거 여부 (기본값: False, 감정 분석에 유용할 수 있음)
            remove_special_chars: 특수문자 제거 여부
            normalize_repeated_chars: 반복 문자 정규화 여부 (ㅋㅋㅋㅋ → ㅋㅋ)
            max_repeated_chars: 최대 허용 반복 문자 수
            remove_html_entities: HTML 엔티티 제거 여부
            normalize_whitespace: 공백 정규화 여부
            lowercase: 소문자 변환 여부 (한글에는 영향 없음)
        """
        self.remove_emojis = remove_emojis
        self.remove_special_chars = remove_special_chars
        self.normalize_repeated_chars = normalize_repeated_chars
        self.max_repeated_chars = max_repeated_chars
        self.remove_html_entities = remove_html_entities
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase

    def clean(self, text: str) -> str:
        """
        텍스트 정제.

        Args:
            text: 원본 텍스트

        Returns:
            정제된 텍스트
        """
        if not text:
            return ""

        result = text

        # 1. HTML 엔티티 제거
        if self.remove_html_entities:
            result = self._remove_html_entities(result)

        # 2. Unicode 정규화 (NFC: 한글 자모 결합)
        result = unicodedata.normalize("NFC", result)

        # 3. 이모지 처리
        if self.remove_emojis:
            result = self._remove_emojis(result)

        # 4. 특수문자 제거
        if self.remove_special_chars:
            result = self._remove_special_chars(result)

        # 5. 반복 문자 정규화
        if self.normalize_repeated_chars:
            result = self._normalize_repeated_chars(result)

        # 6. 공백 정규화
        if self.normalize_whitespace:
            result = self._normalize_whitespace(result)

        # 7. 소문자 변환
        if self.lowercase:
            result = result.lower()

        return result.strip()

    def _remove_html_entities(self, text: str) -> str:
        """HTML 엔티티 제거."""
        # 일반적인 HTML 엔티티를 실제 문자로 변환
        replacements = {
            "&nbsp;": " ",
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&apos;": "'",
            "&#39;": "'",
        }
        for entity, char in replacements.items():
            text = text.replace(entity, char)

        # 나머지 HTML 엔티티 제거
        return HTML_ENTITY_PATTERN.sub("", text)

    def _remove_emojis(self, text: str) -> str:
        """이모지 제거."""
        return EMOJI_PATTERN.sub("", text)

    def _remove_special_chars(self, text: str) -> str:
        """특수문자 제거 (이모지 보존 옵션 고려)."""
        if not self.remove_emojis:
            # 이모지를 보존해야 하는 경우, 이모지를 임시로 치환 후 복원
            emoji_placeholder = {}
            counter = 0

            def save_emoji(match):
                nonlocal counter
                # 알파벳으로만 구성된 플레이스홀더 사용 (특수문자 패턴에 걸리지 않음)
                key = f"EMOJIPLACEHOLDER{counter}END"
                emoji_placeholder[key] = match.group(0)
                counter += 1
                return key

            text = EMOJI_PATTERN.sub(save_emoji, text)
            text = SPECIAL_CHAR_PATTERN.sub(" ", text)

            # 이모지 복원
            for key, emoji in emoji_placeholder.items():
                text = text.replace(key, emoji)
            return text
        else:
            return SPECIAL_CHAR_PATTERN.sub(" ", text)

    def _normalize_repeated_chars(self, text: str) -> str:
        """반복 문자 정규화."""
        return REPEATED_CHAR_PATTERN.sub(r"\1" * self.max_repeated_chars, text)

    def _normalize_whitespace(self, text: str) -> str:
        """공백 정규화."""
        # 연속 공백을 단일 공백으로
        text = MULTIPLE_SPACES_PATTERN.sub(" ", text)
        # 줄바꿈 전후 공백 제거
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)


class TextChunker:
    """텍스트 청킹 클래스."""

    # 한국어 문장 종결 패턴
    SENTENCE_ENDINGS = re.compile(r"([.!?~]+)\s*")

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        split_by_sentence: bool = True,
    ):
        """
        초기화.

        Args:
            chunk_size: 청크 최대 크기 (문자 수)
            chunk_overlap: 청크 간 오버랩 크기
            min_chunk_size: 최소 청크 크기 (이보다 작으면 이전 청크에 병합)
            split_by_sentence: 문장 단위 분할 여부 (True면 문장 경계에서 분할)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.split_by_sentence = split_by_sentence

    def chunk(self, text: str) -> list[TextChunk]:
        """
        텍스트를 청크로 분할.

        Args:
            text: 분할할 텍스트

        Returns:
            TextChunk 리스트
        """
        if not text:
            return []

        # 텍스트가 청크 크기보다 작으면 그대로 반환
        if len(text) <= self.chunk_size:
            return [
                TextChunk(
                    text=text,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        if self.split_by_sentence:
            return self._chunk_by_sentence(text)
        else:
            return self._chunk_by_size(text)

    def _chunk_by_sentence(self, text: str) -> list[TextChunk]:
        """문장 단위로 청킹."""
        # 문장으로 분리
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for sentence in sentences:
            # 현재 청크에 문장 추가 시 크기 확인
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                # 현재 청크가 최소 크기 이상이면 저장
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(
                        TextChunk(
                            text=current_chunk.strip(),
                            chunk_index=chunk_index,
                            start_char=current_start,
                            end_char=current_start + len(current_chunk),
                        )
                    )
                    chunk_index += 1

                    # 오버랩 계산
                    overlap_text = current_chunk[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + sentence
                else:
                    # 최소 크기 미달이면 계속 추가
                    current_chunk += sentence

        # 마지막 청크 처리
        if current_chunk.strip():
            # 마지막 청크가 너무 작으면 이전 청크에 병합
            if len(current_chunk) < self.min_chunk_size and chunks:
                last_chunk = chunks[-1]
                chunks[-1] = TextChunk(
                    text=last_chunk.text + " " + current_chunk.strip(),
                    chunk_index=last_chunk.chunk_index,
                    start_char=last_chunk.start_char,
                    end_char=current_start + len(current_chunk),
                )
            else:
                chunks.append(
                    TextChunk(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                    )
                )

        return chunks

    def _chunk_by_size(self, text: str) -> list[TextChunk]:
        """크기 기반 청킹 (문장 무시)."""
        chunks = []
        chunk_index = 0
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # 단어 경계에서 자르기 시도
            if end < len(text):
                # 공백을 찾아서 거기서 자르기
                last_space = text.rfind(" ", start, end)
                if last_space > start + self.min_chunk_size:
                    end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1

            # 다음 시작 위치 (오버랩 적용)
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """텍스트를 문장으로 분리."""
        # 문장 종결 부호로 분리
        parts = self.SENTENCE_ENDINGS.split(text)

        sentences = []
        i = 0
        while i < len(parts):
            sentence = parts[i]
            # 종결 부호가 있으면 붙이기
            if i + 1 < len(parts) and self.SENTENCE_ENDINGS.match(parts[i + 1]):
                sentence += parts[i + 1]
                i += 2
            else:
                i += 1

            if sentence.strip():
                sentences.append(sentence)

        return sentences


class DuplicateFilter:
    """중복 리뷰 필터링 클래스."""

    def __init__(
        self,
        use_fuzzy: bool = False,
        fuzzy_threshold: float = 0.9,
    ):
        """
        초기화.

        Args:
            use_fuzzy: 퍼지 매칭 사용 여부 (유사 텍스트 중복 감지)
            fuzzy_threshold: 퍼지 매칭 임계값 (0.0 ~ 1.0)
        """
        self.use_fuzzy = use_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self._seen_hashes: set[str] = set()
        self._seen_texts: list[str] = []  # 퍼지 매칭용

    def reset(self) -> None:
        """필터 상태 초기화."""
        self._seen_hashes.clear()
        self._seen_texts.clear()

    def is_duplicate(self, text: str) -> bool:
        """
        중복 여부 확인.

        Args:
            text: 확인할 텍스트

        Returns:
            True면 중복
        """
        text_hash = self._compute_hash(text)

        # 정확히 동일한 텍스트
        if text_hash in self._seen_hashes:
            return True

        # 퍼지 매칭
        if self.use_fuzzy:
            for seen_text in self._seen_texts:
                if self._similarity(text, seen_text) >= self.fuzzy_threshold:
                    return True

        return False

    def add(self, text: str) -> None:
        """
        텍스트를 필터에 추가.

        Args:
            text: 추가할 텍스트
        """
        text_hash = self._compute_hash(text)
        self._seen_hashes.add(text_hash)

        if self.use_fuzzy:
            self._seen_texts.append(text)

    def filter(self, texts: list[str]) -> list[str]:
        """
        중복 제거된 텍스트 리스트 반환.

        Args:
            texts: 텍스트 리스트

        Returns:
            중복이 제거된 텍스트 리스트
        """
        result = []
        for text in texts:
            if not self.is_duplicate(text):
                self.add(text)
                result.append(text)
        return result

    @staticmethod
    def _compute_hash(text: str) -> str:
        """텍스트 해시 계산."""
        # 공백 정규화 후 해시
        normalized = " ".join(text.split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _similarity(text1: str, text2: str) -> float:
        """
        두 텍스트의 유사도 계산 (Jaccard 유사도).

        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        # 단어 집합으로 변환
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard 유사도
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class ReviewPreprocessor:
    """리뷰 전처리 파이프라인 클래스."""

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        chunker: TextChunker | None = None,
        duplicate_filter: DuplicateFilter | None = None,
        min_text_length: int = 10,
        max_text_length: int = 10000,
    ):
        """
        초기화.

        Args:
            cleaner: 텍스트 정제기 (None이면 기본값 사용)
            chunker: 텍스트 청커 (None이면 기본값 사용)
            duplicate_filter: 중복 필터 (None이면 기본값 사용)
            min_text_length: 최소 텍스트 길이 (이보다 짧으면 필터링)
            max_text_length: 최대 텍스트 길이 (이보다 길면 잘라냄)
        """
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or TextChunker()
        self.duplicate_filter = duplicate_filter or DuplicateFilter()
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

    def process(self, review: Review, skip_duplicate_check: bool = False) -> ProcessedReview | None:
        """
        단일 리뷰 전처리.

        Args:
            review: 원본 리뷰
            skip_duplicate_check: 중복 검사 스킵 여부

        Returns:
            전처리된 리뷰 (필터링된 경우 None)
        """
        original_text = review.text

        # 1. 텍스트 길이 검사
        if len(original_text) < self.min_text_length:
            return None

        # 2. 최대 길이 제한
        if len(original_text) > self.max_text_length:
            original_text = original_text[: self.max_text_length]

        # 3. 텍스트 정제
        cleaned_text = self.cleaner.clean(original_text)

        # 정제 후 길이 재검사
        if len(cleaned_text) < self.min_text_length:
            return None

        # 4. 중복 검사
        if not skip_duplicate_check:
            if self.duplicate_filter.is_duplicate(cleaned_text):
                return None
            self.duplicate_filter.add(cleaned_text)

        # 5. 청킹
        chunks = self.chunker.chunk(cleaned_text)

        # 6. 해시 계산
        text_hash = DuplicateFilter._compute_hash(cleaned_text)

        # 7. 메타데이터 추출
        metadata = self._extract_metadata(review)

        return ProcessedReview(
            original_text=review.text,
            cleaned_text=cleaned_text,
            chunks=chunks,
            text_hash=text_hash,
            rating=review.rating,
            date=review.date,
            metadata=metadata,
        )

    def process_batch(
        self,
        reviews: list[Review],
        skip_duplicate_check: bool = False,
    ) -> list[ProcessedReview]:
        """
        리뷰 배치 전처리.

        Args:
            reviews: 원본 리뷰 리스트
            skip_duplicate_check: 중복 검사 스킵 여부

        Returns:
            전처리된 리뷰 리스트
        """
        results = []
        for review in reviews:
            processed = self.process(review, skip_duplicate_check)
            if processed:
                results.append(processed)
        return results

    def iter_process(
        self,
        reviews: Iterator[Review],
        skip_duplicate_check: bool = False,
    ) -> Iterator[ProcessedReview]:
        """
        리뷰 이터레이터 전처리.

        Args:
            reviews: 원본 리뷰 이터레이터
            skip_duplicate_check: 중복 검사 스킵 여부

        Yields:
            전처리된 리뷰
        """
        for review in reviews:
            processed = self.process(review, skip_duplicate_check)
            if processed:
                yield processed

    def _extract_metadata(self, review: Review) -> dict[str, Any]:
        """
        리뷰에서 메타데이터 추출.

        Args:
            review: 원본 리뷰

        Returns:
            추출된 메타데이터
        """
        metadata = dict(review.metadata) if review.metadata else {}

        # 기본 메타데이터 추가
        if review.author:
            metadata["author"] = review.author

        if review.option:
            metadata["option"] = review.option

        if review.helpful_count is not None:
            metadata["helpful_count"] = review.helpful_count

        if review.verified_purchase:
            metadata["verified_purchase"] = review.verified_purchase

        if review.images:
            metadata["has_images"] = True
            metadata["image_count"] = len(review.images)

        # 텍스트 통계
        metadata["text_length"] = len(review.text)
        metadata["word_count"] = len(review.text.split())

        return metadata

    def reset(self) -> None:
        """파이프라인 상태 초기화 (중복 필터 리셋)."""
        self.duplicate_filter.reset()

    def get_statistics(self, processed_reviews: list[ProcessedReview]) -> dict[str, Any]:
        """
        전처리 결과 통계 반환.

        Args:
            processed_reviews: 전처리된 리뷰 리스트

        Returns:
            통계 정보
        """
        if not processed_reviews:
            return {"total": 0}

        total = len(processed_reviews)
        total_chunks = sum(len(r.chunks) for r in processed_reviews)
        text_lengths = [len(r.cleaned_text) for r in processed_reviews]
        ratings = [r.rating for r in processed_reviews if r.rating is not None]

        return {
            "total_reviews": total,
            "total_chunks": total_chunks,
            "avg_chunks_per_review": total_chunks / total,
            "avg_text_length": sum(text_lengths) / total,
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
            "avg_rating": sum(ratings) / len(ratings) if ratings else None,
            "reviews_with_rating": len(ratings),
        }


def create_default_preprocessor(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    remove_emojis: bool = False,
    use_fuzzy_dedup: bool = False,
) -> ReviewPreprocessor:
    """
    기본 설정의 전처리기 생성.

    Args:
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        remove_emojis: 이모지 제거 여부
        use_fuzzy_dedup: 퍼지 중복 제거 사용 여부

    Returns:
        ReviewPreprocessor 인스턴스
    """
    cleaner = TextCleaner(
        remove_emojis=remove_emojis,
        remove_special_chars=True,
        normalize_repeated_chars=True,
    )

    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_by_sentence=True,
    )

    duplicate_filter = DuplicateFilter(
        use_fuzzy=use_fuzzy_dedup,
        fuzzy_threshold=0.9,
    )

    return ReviewPreprocessor(
        cleaner=cleaner,
        chunker=chunker,
        duplicate_filter=duplicate_filter,
    )


def main():
    """테스트 실행."""
    from pathlib import Path

    from src.pipeline.aihub_loader import AIHubDataLoader

    # AI Hub 데이터 로드
    data_dir = Path(__file__).parent.parent.parent / "data" / "aihub_data"

    if not data_dir.exists():
        print(f"데이터 디렉토리가 없습니다: {data_dir}")
        print("샘플 데이터로 테스트합니다.")

        # 샘플 리뷰로 테스트
        sample_reviews = [
            Review(
                text="이 제품 정말 좋아요ㅋㅋㅋㅋㅋ 배송도 빨라서 만족합니다!!! 👍👍",
                rating=5.0,
                date="2024-01-15",
            ),
            Review(
                text="품질이 기대에 못 미치네요... 가격 대비 별로입니다.",
                rating=2.0,
                date="2024-01-14",
            ),
            Review(
                text="이 제품 정말 좋아요 배송도 빨라서 만족합니다!",  # 유사 중복
                rating=5.0,
                date="2024-01-13",
            ),
        ]
    else:
        loader = AIHubDataLoader(data_dir)
        sample_reviews = loader.load_reviews(limit=10, as_project_format=True)

    # 전처리기 생성
    preprocessor = create_default_preprocessor(
        chunk_size=300,
        chunk_overlap=30,
        remove_emojis=True,
        use_fuzzy_dedup=True,
    )

    # 전처리 실행
    print("=== 전처리 테스트 ===\n")

    processed = preprocessor.process_batch(sample_reviews)

    print(f"원본 리뷰 수: {len(sample_reviews)}")
    print(f"전처리 후 리뷰 수: {len(processed)}")

    if processed:
        print(f"\n=== 전처리 결과 샘플 ===\n")
        for i, p in enumerate(processed[:3], 1):
            print(f"[{i}] 원본: {p.original_text[:50]}...")
            print(f"    정제: {p.cleaned_text[:50]}...")
            print(f"    청크 수: {len(p.chunks)}")
            print(f"    평점: {p.rating}")
            print(f"    해시: {p.text_hash[:16]}...")
            print()

        # 통계 출력
        stats = preprocessor.get_statistics(processed)
        print("=== 전처리 통계 ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
