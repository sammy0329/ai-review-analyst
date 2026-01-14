# API Documentation

> AI Review Analyst 핵심 모듈 API 레퍼런스

---

## Overview

AI Review Analyst는 다음 핵심 모듈을 제공합니다:

| 모듈 | 설명 |
|------|------|
| `src.agents` | LangGraph 기반 멀티 에이전트 시스템 |
| `src.chains` | RAG Chain 구현 |
| `src.pipeline` | 데이터 파이프라인 (임베딩, 전처리) |
| `src.prompts` | 프롬프트 템플릿 |
| `src.database` | SQLite 데이터베이스 관리 |

---

## 1. Agents 모듈

### 1.1 Quick Start

```python
from src.agents import create_review_agent_graph, create_initial_state, IntentType
from src.chains import create_rag_chain
from src.pipeline.embedder import create_embedder

# RAG Chain 생성
embedder = create_embedder(collection_name="my_reviews")
rag_chain = create_rag_chain(embedder=embedder)

# 에이전트 그래프 생성
graph = create_review_agent_graph(rag_chain)

# 질문 처리
state = create_initial_state(
    query="이 제품의 배송은 어떤가요?",
    product_name="테스트 제품"
)
result = graph.invoke(state)

print(result["intent"])      # IntentType.QA
print(result["response"])    # AI 응답
```

### 1.2 AgentState

에이전트 간 공유되는 상태 타입입니다.

```python
from typing import TypedDict

class AgentState(TypedDict, total=False):
    query: str                    # 사용자 질문 (필수)
    product_name: str | None      # 제품명
    intent: IntentType            # 분류된 의도
    response: str                 # 생성된 응답
    source_documents: list[dict]  # 참조 문서
    metadata: dict                # 메타데이터
    error: str | None             # 에러 메시지
```

### 1.3 IntentType

의도 분류 타입입니다.

```python
from enum import Enum

class IntentType(str, Enum):
    QA = "qa"           # 질의응답
    SUMMARY = "summary" # 요약
    UNKNOWN = "unknown" # 미분류
```

### 1.4 create_initial_state()

초기 상태 생성 헬퍼 함수입니다.

```python
def create_initial_state(
    query: str,
    product_name: str | None = None,
    metadata: dict | None = None,
) -> AgentState:
    """
    초기 에이전트 상태 생성.

    Args:
        query: 사용자 질문
        product_name: 제품명 (선택)
        metadata: 추가 메타데이터 (선택)

    Returns:
        AgentState 딕셔너리
    """
```

### 1.5 create_review_agent_graph()

멀티 에이전트 그래프를 생성합니다.

```python
def create_review_agent_graph(
    rag_chain: ReviewRAGChain,
    llm: ChatOpenAI | None = None,
) -> StateGraph:
    """
    리뷰 분석 에이전트 그래프 생성.

    그래프 구조:
        START → Intent Classifier → Router → [QA|Summary] → END

    Args:
        rag_chain: ReviewRAGChain 인스턴스
        llm: ChatOpenAI 인스턴스 (Intent Classifier용, 선택)

    Returns:
        컴파일된 StateGraph
    """
```

### 1.6 개별 에이전트

#### QAAgent
질의응답 에이전트입니다.

```python
from src.agents import QAAgent, create_qa_agent

# 클래스 사용
agent = QAAgent(rag_chain)
result = agent.invoke(state)

# 노드 함수 생성
node = create_qa_agent(rag_chain)
```

#### SummarizeAgent
요약 에이전트입니다.

```python
from src.agents import SummarizeAgent, create_summarize_agent

# 클래스 사용 (top_k 설정 가능)
agent = SummarizeAgent(rag_chain, top_k=10)
result = agent.invoke(state)

# 노드 함수 생성
node = create_summarize_agent(rag_chain, top_k=10)
```

### 1.7 Intent Classifier

의도 분류기입니다. 규칙 기반 + LLM 하이브리드 방식을 사용합니다.

```python
from src.agents import classify_intent, IntentClassification

# 규칙 기반 분류
result: IntentClassification = classify_intent("이 제품 리뷰 요약해줘")
print(result.intent)      # IntentType.SUMMARY
print(result.confidence)  # 1.0
print(result.method)      # "rule_based"
```

**키워드 패턴:**
| 의도 | 키워드 |
|------|--------|
| SUMMARY | 요약, 정리, 종합, 전체적, 총평 |

---

## 2. Chains 모듈

### 2.1 RAGConfig

RAG Chain 설정 클래스입니다.

```python
from dataclasses import dataclass

@dataclass
class RAGConfig:
    # LLM 설정
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    streaming: bool = True

    # 검색 설정
    top_k: int = 5
    search_type: str = "similarity"  # "similarity" or "mmr"

    # 프롬프트 설정
    system_prompt: str = "..."
    user_prompt_template: str = "..."
```

### 2.2 RAGResponse

RAG 응답 데이터 구조입니다.

```python
@dataclass
class RAGResponse:
    answer: str                              # 생성된 답변
    source_documents: list[Document] = []   # 참조 문서
    metadata: dict = {}                      # 메타데이터
```

### 2.3 ReviewRAGChain

RAG Chain 메인 클래스입니다.

```python
class ReviewRAGChain:
    def __init__(
        self,
        embedder: ReviewEmbedder | None = None,
        config: RAGConfig | None = None,
        openai_api_key: str | None = None,
    ):
        """
        초기화.

        Args:
            embedder: ReviewEmbedder 인스턴스 (None이면 새로 생성)
            config: RAG 설정 (None이면 기본값 사용)
            openai_api_key: OpenAI API 키
        """
```

#### 메서드

**query()**
```python
def query(self, question: str) -> RAGResponse:
    """
    질문에 대한 답변 생성.

    Args:
        question: 사용자 질문

    Returns:
        RAGResponse 객체
    """
```

**query_with_sources()**
```python
def query_with_sources(self, question: str) -> dict[str, Any]:
    """
    출처와 함께 답변 반환.

    Args:
        question: 사용자 질문

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "metadata": dict
        }
    """
```

**stream()**
```python
def stream(self, question: str) -> Iterator[str]:
    """
    스트리밍 방식으로 답변 생성.

    Args:
        question: 사용자 질문

    Yields:
        답변 텍스트 청크
    """
```

**astream()**
```python
async def astream(self, question: str) -> AsyncIterator[str]:
    """
    비동기 스트리밍 방식으로 답변 생성.
    """
```

**update_config()**
```python
def update_config(self, **kwargs) -> None:
    """
    설정 업데이트.

    Args:
        **kwargs: 업데이트할 설정 값들
            - model_name, temperature, max_tokens, streaming
            - top_k, search_type
    """
```

**set_prompt()**
```python
def set_prompt(self, prompt_name: str) -> None:
    """
    프롬프트 템플릿 변경.

    Args:
        prompt_name: "qa", "summary", "sentiment"
    """
```

### 2.4 create_rag_chain()

RAG Chain 생성 헬퍼 함수입니다.

```python
def create_rag_chain(
    embedder: ReviewEmbedder | None = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_k: int = 5,
    streaming: bool = True,
    openai_api_key: str | None = None,
) -> ReviewRAGChain:
    """
    RAG Chain 생성 헬퍼 함수.

    Args:
        embedder: ReviewEmbedder 인스턴스
        model_name: LLM 모델명
        temperature: 온도 설정
        top_k: 검색할 문서 수
        streaming: 스트리밍 사용 여부
        openai_api_key: OpenAI API 키

    Returns:
        ReviewRAGChain 인스턴스
    """
```

---

## 3. Pipeline 모듈

### 3.1 ReviewEmbedder

리뷰 임베딩 및 벡터 DB 관리 클래스입니다.

```python
from src.pipeline.embedder import create_embedder

# 생성
embedder = create_embedder(
    collection_name="product_reviews",
    persist_directory="./data/chroma_db",
    openai_api_key="...",
)

# 리뷰 추가
embedder.add_reviews(processed_reviews)

# Retriever 획득
retriever = embedder.get_retriever(top_k=5)

# 직접 검색
docs = embedder.search("배송이 빠른가요?", k=5)

# 컬렉션 관리
embedder.reset_collection()
embedder.delete_collection()
```

### 3.2 AIHubDataLoader

AI Hub 데이터 로더입니다.

```python
from src.pipeline.aihub_loader import AIHubDataLoader

# 로더 생성
loader = AIHubDataLoader(data_dir="./data/aihub")

# 제품 목록 조회
products = loader.get_products()

# 특정 제품 리뷰 조회
reviews = loader.get_reviews_by_product("제품명")

# 카테고리별 조회
reviews = loader.get_reviews_by_category("패션")

# 통계 조회
stats = loader.get_statistics()
```

---

## 4. Prompts 모듈

### 4.1 사용 가능한 프롬프트

| 이름 | 용도 |
|------|------|
| `qa` | 질의응답 |
| `summary` | 리뷰 요약 |
| `sentiment` | 감성 분석 |

### 4.2 get_prompt()

```python
from src.prompts.templates import get_prompt, PromptTemplate

prompt: PromptTemplate = get_prompt("qa")
print(prompt.system_prompt)
print(prompt.user_prompt_template)
```

---

## 5. 전체 사용 예시

### 5.1 기본 RAG 질의응답

```python
from dotenv import load_dotenv
from src.chains import create_rag_chain
from src.pipeline.embedder import create_embedder

load_dotenv()

# 임베더 생성 (기존 데이터 사용)
embedder = create_embedder(
    collection_name="product_reviews",
    persist_directory="./data/chroma_db",
)

# RAG Chain 생성
rag_chain = create_rag_chain(embedder=embedder, top_k=5)

# 질문
result = rag_chain.query_with_sources("이 제품 배송 빠른가요?")

print(f"답변: {result['answer']}")
print(f"참조 리뷰 수: {len(result['sources'])}")
```

### 5.2 멀티 에이전트 사용

```python
from src.agents import create_review_agent_graph, create_initial_state

# 그래프 생성
graph = create_review_agent_graph(rag_chain)

# 여러 유형의 질문 처리
queries = [
    ("이 제품 리뷰 요약해줘", None),           # SUMMARY
    ("배송은 빠른가요?", None),                 # QA
]

for query, product_name in queries:
    state = create_initial_state(query=query, product_name=product_name)
    result = graph.invoke(state)

    print(f"질문: {query}")
    print(f"의도: {result['intent']}")
    print(f"답변: {result['response'][:100]}...")
    print("-" * 50)
```

### 5.3 스트리밍 응답

```python
# Streamlit에서 스트리밍
import streamlit as st

def stream_response(question: str):
    for chunk in rag_chain.stream(question):
        yield chunk

st.write_stream(stream_response("이 제품 추천하시나요?"))
```

---

## 6. 환경 변수

| 변수 | 필수 | 설명 |
|------|------|------|
| `OPENAI_API_KEY` | O | OpenAI API 키 |
| `CHROMA_DB_PATH` | X | ChromaDB 저장 경로 (기본: `./data/chroma_db`) |

---

## 7. Database 모듈

SQLite 기반 리뷰 데이터 저장 및 조회를 위한 모듈입니다.

### 7.1 Quick Start

```python
from src.database import (
    get_products,
    get_reviews_by_product,
    search_products,
    get_representative_reviews,
)

# 전체 제품 목록 조회
products = get_products()

# 특정 제품 리뷰 조회
reviews = get_reviews_by_product(product_id=1)

# 제품 검색
results = search_products(query="운동화")

# 대표 리뷰 조회
rep_reviews = get_representative_reviews(product_id=1, limit=3)
```

### 7.2 주요 함수

#### get_connection()
```python
def get_connection() -> sqlite3.Connection:
    """
    SQLite 데이터베이스 연결 반환.

    Returns:
        sqlite3.Connection 객체
    """
```

#### get_products()
```python
def get_products(
    category: str | None = None,
    subcategory: str | None = None,
) -> list[dict]:
    """
    제품 목록 조회.

    Args:
        category: 대분류 필터 (선택)
        subcategory: 소분류 필터 (선택)

    Returns:
        제품 정보 딕셔너리 리스트
        [{"id": 1, "name": "...", "category": "...", ...}]
    """
```

#### get_reviews_by_product()
```python
def get_reviews_by_product(
    product_id: int,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """
    특정 제품의 리뷰 조회.

    Args:
        product_id: 제품 ID
        limit: 최대 조회 수 (선택)
        offset: 시작 위치 (기본: 0)

    Returns:
        리뷰 정보 딕셔너리 리스트
    """
```

#### search_products()
```python
def search_products(
    query: str,
    category: str | None = None,
) -> list[dict]:
    """
    제품명으로 검색.

    Args:
        query: 검색어
        category: 카테고리 필터 (선택)

    Returns:
        매칭된 제품 리스트
    """
```

#### get_representative_reviews()
```python
def get_representative_reviews(
    product_id: int,
    limit: int = 3,
) -> list[dict]:
    """
    제품의 대표 리뷰 조회.

    가중치 기반 선정:
    - 텍스트 길이 (100~500자 선호)
    - 감정 다양성 (긍정/중립/부정 균형)
    - 유용성 점수

    Args:
        product_id: 제품 ID
        limit: 반환할 리뷰 수 (기본: 3)

    Returns:
        대표 리뷰 리스트
    """
```

#### randomize_review_dates()
```python
def randomize_review_dates(start_days_ago: int = 365) -> int:
    """
    모든 리뷰의 created_at을 임의 날짜로 업데이트.

    Args:
        start_days_ago: 시작 날짜 (현재로부터 몇 일 전, 기본: 365)

    Returns:
        업데이트된 리뷰 수

    Example:
        # 최근 1년 내 임의 날짜로 설정
        updated = randomize_review_dates(365)
        print(f"{updated}개 리뷰 날짜 업데이트됨")
    """
```

### 7.3 데이터베이스 스키마

```sql
-- products 테이블
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    subcategory TEXT,
    avg_rating REAL,
    review_count INTEGER
);

-- reviews 테이블
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    text TEXT NOT NULL,
    rating REAL,
    sentiment TEXT,
    created_at TEXT,
    is_suspicious INTEGER DEFAULT 0
);
```

---

## 8. 에러 처리

모든 에이전트는 에러 발생 시 `AgentState`의 `error` 필드에 에러 메시지를 설정합니다.

```python
result = graph.invoke(state)

if result.get("error"):
    print(f"에러 발생: {result['error']}")
else:
    print(f"응답: {result['response']}")
```

---

*본 문서는 프로젝트 업데이트에 따라 지속적으로 갱신됩니다.*
