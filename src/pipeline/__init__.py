# Pipeline module

# AIHub loader는 playwright 의존성이 있으므로 선택적으로 import
try:
    from .aihub_loader import AIHubDataLoader, AIHubReview, Product
except ImportError:
    AIHubDataLoader = None  # type: ignore
    AIHubReview = None  # type: ignore
    Product = None  # type: ignore

from .aspect_extractor import (
    AspectCategory,
    AspectExtractor,
    AspectResult,
    Sentiment,
    create_aspect_extractor,
)
from .embedder import (
    CollectionManager,
    EmbedderConfig,
    ReviewEmbedder,
    SearchResult,
    create_embedder,
)
from .preprocessor import (
    DuplicateFilter,
    ProcessedReview,
    ReviewPreprocessor,
    TextChunk,
    TextChunker,
    TextCleaner,
    create_default_preprocessor,
)

__all__ = [
    # AI Hub Data Loader
    "AIHubDataLoader",
    "AIHubReview",
    "Product",
    # Aspect Extractor
    "AspectCategory",
    "AspectExtractor",
    "AspectResult",
    "Sentiment",
    "create_aspect_extractor",
    # Preprocessor
    "TextCleaner",
    "TextChunker",
    "TextChunk",
    "DuplicateFilter",
    "ProcessedReview",
    "ReviewPreprocessor",
    "create_default_preprocessor",
    # Embedder
    "EmbedderConfig",
    "SearchResult",
    "ReviewEmbedder",
    "CollectionManager",
    "create_embedder",
]
