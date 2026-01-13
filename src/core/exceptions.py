"""
커스텀 예외 클래스 모듈.

프로젝트에서 사용하는 예외 클래스들을 정의합니다.
"""

from typing import Optional


class ReviewAnalystError(Exception):
    """AI Review Analyst 기본 예외 클래스."""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """
        Args:
            message: 에러 메시지
            details: 상세 정보
            suggestion: 해결 방법 제안
        """
        super().__init__(message)
        self.message = message
        self.details = details
        self.suggestion = suggestion

    def __str__(self) -> str:
        parts = [self.message]
        if self.details:
            parts.append(f"상세: {self.details}")
        if self.suggestion:
            parts.append(f"해결: {self.suggestion}")
        return " | ".join(parts)


# =============================================================================
# API 관련 예외
# =============================================================================


class APIError(ReviewAnalystError):
    """API 호출 관련 예외."""

    def __init__(
        self,
        message: str = "API 호출에 실패했습니다.",
        status_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class RateLimitError(APIError):
    """API Rate Limit 초과 예외."""

    def __init__(
        self,
        message: str = "API 요청 한도를 초과했습니다.",
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            status_code=429,
            suggestion="잠시 후 다시 시도해주세요.",
            **kwargs,
        )
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """API 인증 실패 예외."""

    def __init__(
        self,
        message: str = "API 인증에 실패했습니다.",
        **kwargs,
    ):
        super().__init__(
            message,
            status_code=401,
            suggestion=".env 파일의 API 키를 확인해주세요.",
            **kwargs,
        )


# =============================================================================
# 데이터 관련 예외
# =============================================================================


class DataLoadError(ReviewAnalystError):
    """데이터 로드 실패 예외."""

    def __init__(
        self,
        message: str = "데이터를 로드할 수 없습니다.",
        file_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.file_path = file_path
        if file_path:
            self.details = f"파일: {file_path}"


class DataValidationError(ReviewAnalystError):
    """데이터 유효성 검사 실패 예외."""

    def __init__(
        self,
        message: str = "데이터 형식이 올바르지 않습니다.",
        field: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.field = field
        if field:
            self.details = f"필드: {field}"


# =============================================================================
# 크롤링 관련 예외
# =============================================================================


class CrawlerError(ReviewAnalystError):
    """크롤링 실패 예외."""

    def __init__(
        self,
        message: str = "크롤링에 실패했습니다.",
        url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.url = url


class CrawlerTimeoutError(CrawlerError):
    """크롤링 타임아웃 예외."""

    def __init__(
        self,
        message: str = "크롤링 시간이 초과되었습니다.",
        timeout: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            suggestion="네트워크 상태를 확인하거나 타임아웃 설정을 늘려주세요.",
            **kwargs,
        )
        self.timeout = timeout


class CrawlerBlockedError(CrawlerError):
    """크롤링 차단 예외."""

    def __init__(
        self,
        message: str = "크롤링이 차단되었습니다.",
        **kwargs,
    ):
        super().__init__(
            message,
            suggestion="잠시 후 다시 시도하거나 다른 방법을 사용해주세요.",
            **kwargs,
        )


# =============================================================================
# 임베딩/벡터DB 관련 예외
# =============================================================================


class EmbeddingError(ReviewAnalystError):
    """임베딩 생성 실패 예외."""

    def __init__(
        self,
        message: str = "임베딩 생성에 실패했습니다.",
        **kwargs,
    ):
        super().__init__(message, **kwargs)


class VectorDBError(ReviewAnalystError):
    """벡터 DB 작업 실패 예외."""

    def __init__(
        self,
        message: str = "벡터 데이터베이스 작업에 실패했습니다.",
        collection: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.collection = collection


# =============================================================================
# RAG 관련 예외
# =============================================================================


class RAGError(ReviewAnalystError):
    """RAG 처리 실패 예외."""

    def __init__(
        self,
        message: str = "RAG 처리에 실패했습니다.",
        query: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.query = query


class RetrievalError(RAGError):
    """문서 검색 실패 예외."""

    def __init__(
        self,
        message: str = "관련 문서를 검색할 수 없습니다.",
        **kwargs,
    ):
        super().__init__(message, **kwargs)


class GenerationError(RAGError):
    """응답 생성 실패 예외."""

    def __init__(
        self,
        message: str = "응답을 생성할 수 없습니다.",
        **kwargs,
    ):
        super().__init__(message, **kwargs)
