"""
로깅 설정 모듈.

프로젝트 전체에서 사용하는 로깅 설정을 제공합니다.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# 로그 포맷
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 로그 레벨 매핑
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# 기본 로그 디렉토리
DEFAULT_LOG_DIR = Path("logs")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> None:
    """
    로깅 설정 초기화.

    Args:
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일명 (None이면 파일 로깅 비활성화)
        log_dir: 로그 디렉토리 경로
    """
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))

    # 기존 핸들러 제거
    root_logger.handlers.clear()

    # 포맷터 생성
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)

    # 파일 핸들러 추가 (옵션)
    if log_file:
        log_path = (log_dir or DEFAULT_LOG_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
        root_logger.addHandler(file_handler)

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 반환.

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        logging.Logger 인스턴스

    Example:
        ```python
        from src.core.logging import get_logger

        logger = get_logger(__name__)
        logger.info("작업 시작")
        logger.error("오류 발생", exc_info=True)
        ```
    """
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """
    예외를 로깅.

    Args:
        logger: 로거 인스턴스
        error: 예외 객체
        context: 컨텍스트 설명
    """
    if context:
        logger.error(f"{context}: {type(error).__name__}: {error}", exc_info=True)
    else:
        logger.error(f"{type(error).__name__}: {error}", exc_info=True)
