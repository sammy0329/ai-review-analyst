# Tech Stack Documentation

> AI Review Analyst 프로젝트에서 사용하는 기술 스택 상세 문서

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TECH STACK OVERVIEW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                        PRESENTATION LAYER                        │   │
│   │                         [ Streamlit ]                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                       APPLICATION LAYER                          │   │
│   │     [ LangChain ]  ←→  [ LangGraph ]  ←→  [ OpenAI API ]        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                          DATA LAYER                              │   │
│   │   [ Playwright ]  →  [ BeautifulSoup ]  →  [ ChromaDB/SQLite ]  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      INFRASTRUCTURE LAYER                        │   │
│   │              [ Docker ]  ←→  [ AWS EC2 (Free Tier) ]            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Core Language

### Python 3.9+

| 항목 | 내용 |
|------|------|
| **버전** | 3.9 이상 (권장: 3.11) |
| **선택 이유** | AI/ML 생태계 지원, 풍부한 라이브러리, 빠른 프로토타이핑 |
| **공식 문서** | https://docs.python.org/3/ |

**주요 사용 라이브러리:**
```
python-dotenv    # 환경 변수 관리
pydantic         # 데이터 검증 및 설정 관리
asyncio          # 비동기 처리
typing           # 타입 힌팅
```

---

## 2. LLM Framework

### 2.1 LangChain

| 항목 | 내용 |
|------|------|
| **버전** | 0.1.x (최신 안정 버전) |
| **역할** | LLM 애플리케이션 오케스트레이션 프레임워크 |
| **선택 이유** | Chain/Agent 추상화, 풍부한 통합 생태계, 활발한 커뮤니티 |
| **공식 문서** | https://python.langchain.com/docs/ |

**프로젝트 내 활용:**
```python
# RAG Chain 구성 예시
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = Chroma(collection_name="reviews").as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True  # 출처 표기를 위한 원문 반환
)
```

**핵심 모듈:**
| 모듈 | 용도 |
|------|------|
| `langchain-core` | 기본 추상화 (Runnable, Chain) |
| `langchain-openai` | OpenAI 모델 통합 |
| `langchain-chroma` | ChromaDB 벡터스토어 통합 |
| `langchain-community` | 커뮤니티 통합 (Playwright 등) |

---

### 2.2 LangGraph

| 항목 | 내용 |
|------|------|
| **버전** | 0.0.x (최신) |
| **역할** | 복잡한 에이전트 워크플로우 제어 |
| **선택 이유** | 상태 기반 그래프 구조, 조건부 라우팅, 순환 처리 가능 |
| **공식 문서** | https://langchain-ai.github.io/langgraph/ |

**프로젝트 내 활용:**
```python
# Multi-Agent 라우팅 예시
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class AgentState(TypedDict):
    query: str
    intent: Literal["summarize", "qa", "compare"]
    response: str

def router(state: AgentState) -> str:
    """사용자 의도에 따라 적절한 에이전트로 라우팅"""
    intent = state["intent"]
    if intent == "summarize":
        return "summarize_agent"
    elif intent == "qa":
        return "qa_agent"
    else:
        return "compare_agent"

# 그래프 구성
workflow = StateGraph(AgentState)
workflow.add_node("classifier", classify_intent)
workflow.add_node("summarize_agent", summarize_reviews)
workflow.add_node("qa_agent", answer_question)
workflow.add_node("compare_agent", compare_products)

workflow.add_conditional_edges("classifier", router)
workflow.set_entry_point("classifier")
```

**LangGraph vs LangChain Agent:**
| 비교 항목 | LangChain Agent | LangGraph |
|-----------|-----------------|-----------|
| 구조 | 단일 루프 | 그래프 기반 |
| 상태 관리 | 제한적 | 명시적 State 정의 |
| 조건 분기 | Tool 선택 의존 | 명시적 Edge 정의 |
| 적합한 경우 | 단순 도구 호출 | 복잡한 멀티스텝 워크플로우 |

---

## 3. AI Model

### OpenAI GPT-4o-mini

| 항목 | 내용 |
|------|------|
| **모델명** | gpt-4o-mini |
| **선택 이유** | 비용 효율성 + 충분한 성능 (GPT-4 대비 약 10배 저렴) |
| **토큰 제한** | 입력 128K, 출력 16K |
| **공식 문서** | https://platform.openai.com/docs/models |

**비용 비교:**
| 모델 | Input (1M tokens) | Output (1M tokens) |
|------|-------------------|-------------------|
| GPT-4o | $5.00 | $15.00 |
| **GPT-4o-mini** | **$0.15** | **$0.60** |
| GPT-3.5 Turbo | $0.50 | $1.50 |

**프로젝트 내 활용:**
```python
from langchain_openai import ChatOpenAI

# 메인 LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,        # 일관된 출력을 위해 0 설정
    max_tokens=2048,
    streaming=True        # UX 향상을 위한 스트리밍
)

# 임베딩 모델 (별도)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

**임베딩 모델 선택:**
| 모델 | 차원 | 비용 (1M tokens) | 특징 |
|------|------|------------------|------|
| text-embedding-3-small | 1536 | $0.02 | 경량, 빠름 |
| text-embedding-3-large | 3072 | $0.13 | 고성능, 정확 |

→ MVP 단계에서는 **text-embedding-3-small** 사용 권장

---

## 4. Vector Database

### ChromaDB

| 항목 | 내용 |
|------|------|
| **버전** | 0.4.x |
| **역할** | 리뷰 텍스트 임베딩 저장 및 시맨틱 검색 |
| **선택 이유** | 로컬 개발 용이, 설치 간편, Python 네이티브 |
| **공식 문서** | https://docs.trychroma.com/ |

**vs 다른 Vector DB:**
| DB | 장점 | 단점 | 적합한 경우 |
|----|------|------|-------------|
| **ChromaDB** | 설치 간편, 로컬 개발 | 대용량 한계 | MVP, 프로토타입 |
| Pinecone | 관리형, 확장성 | 비용 발생 | 프로덕션 |
| Weaviate | 기능 풍부 | 러닝커브 | 복잡한 요구사항 |
| FAISS | 고성능 | 관리 필요 | 대용량 검색 |

**프로젝트 내 활용:**
```python
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ChromaDB 클라이언트 설정
client = chromadb.PersistentClient(path="./data/chroma_db")

# LangChain 통합
vectorstore = Chroma(
    client=client,
    collection_name="product_reviews",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# 리뷰 저장
vectorstore.add_texts(
    texts=["맛있어요! 배송도 빨랐습니다.", "포장이 조금 아쉬웠어요"],
    metadatas=[
        {"product_id": "123", "rating": 5, "date": "2024-01-10"},
        {"product_id": "123", "rating": 3, "date": "2024-01-09"}
    ]
)

# 시맨틱 검색
results = vectorstore.similarity_search(
    query="배송이 빠른가요?",
    k=5  # 상위 5개 결과
)
```

---

## 5. Metadata Database

### SQLite

| 항목 | 내용 |
|------|------|
| **역할** | 상품 메타데이터, 크롤링 히스토리 저장 |
| **선택 이유** | 별도 서버 불필요, Python 표준 라이브러리 |
| **공식 문서** | https://docs.python.org/3/library/sqlite3.html |

**스키마 예시:**
```sql
-- 상품 정보
CREATE TABLE products (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    name TEXT,
    price INTEGER,
    total_reviews INTEGER,
    avg_rating REAL,
    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 크롤링 히스토리
CREATE TABLE crawl_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT,
    status TEXT,  -- 'success', 'failed', 'pending'
    reviews_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id)
);
```

---

## 6. Web Crawling

### 6.1 Playwright

| 항목 | 내용 |
|------|------|
| **버전** | 1.40.x |
| **역할** | 동적 웹페이지(JS 렌더링) 리뷰 크롤링 |
| **선택 이유** | Headless 브라우저, 안정적인 자동화, 다양한 브라우저 지원 |
| **공식 문서** | https://playwright.dev/python/ |

**vs 다른 크롤링 도구:**
| 도구 | JS 렌더링 | 속도 | 적합한 경우 |
|------|-----------|------|-------------|
| **Playwright** | O | 중간 | 동적 SPA |
| Selenium | O | 느림 | 레거시 지원 |
| Requests | X | 빠름 | 정적 페이지 |
| Scrapy | X | 빠름 | 대용량 크롤링 |

**프로젝트 내 활용:**
```python
from playwright.async_api import async_playwright

async def crawl_reviews(url: str) -> list[dict]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # User-Agent 설정 (차단 우회)
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."
        })

        await page.goto(url)

        # 리뷰 영역이 로드될 때까지 대기
        await page.wait_for_selector(".review-item", timeout=10000)

        # 무한 스크롤 처리
        for _ in range(10):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)

        # 리뷰 데이터 추출
        reviews = await page.evaluate("""
            () => Array.from(document.querySelectorAll('.review-item')).map(el => ({
                text: el.querySelector('.review-text')?.innerText,
                rating: el.querySelector('.rating')?.getAttribute('data-score'),
                date: el.querySelector('.date')?.innerText
            }))
        """)

        await browser.close()
        return reviews
```

---

### 6.2 BeautifulSoup4

| 항목 | 내용 |
|------|------|
| **버전** | 4.12.x |
| **역할** | HTML 파싱 및 데이터 추출 |
| **선택 이유** | 직관적인 API, 유연한 파서 선택 |
| **공식 문서** | https://www.crummy.com/software/BeautifulSoup/bs4/doc/ |

**프로젝트 내 활용:**
```python
from bs4 import BeautifulSoup

def parse_review_html(html: str) -> dict:
    soup = BeautifulSoup(html, 'lxml')

    return {
        "title": soup.select_one("h1.product-title").get_text(strip=True),
        "price": soup.select_one(".price").get_text(strip=True),
        "reviews": [
            {
                "text": review.select_one(".text").get_text(strip=True),
                "rating": int(review.get("data-rating", 0))
            }
            for review in soup.select(".review-item")
        ]
    }
```

---

## 7. Frontend

### Streamlit

| 항목 | 내용 |
|------|------|
| **버전** | 1.30.x |
| **역할** | MVP 웹 대시보드 UI |
| **선택 이유** | Python 전용, 빠른 프로토타이핑, 데이터 시각화 내장 |
| **공식 문서** | https://docs.streamlit.io/ |

**프로젝트 내 활용:**
```python
import streamlit as st

st.set_page_config(page_title="AI Review Analyst", layout="wide")

st.title("AI Review Analyst")
st.markdown("리뷰를 읽는 시간 30분을 30초로 단축하다.")

# URL 입력
url = st.text_input("분석할 상품 URL을 입력하세요")

if st.button("분석 시작"):
    with st.spinner("리뷰를 분석하고 있습니다..."):
        # 크롤링 및 분석 로직
        result = analyze_reviews(url)

    # 결과 표시
    col1, col2 = st.columns(2)
    with col1:
        st.metric("긍정 리뷰", f"{result['positive_ratio']}%")
    with col2:
        st.metric("부정 리뷰", f"{result['negative_ratio']}%")

    # 차트
    st.bar_chart(result['keyword_stats'])

# 채팅 인터페이스
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("리뷰에 대해 질문하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # LLM 응답 (Streaming)
    with st.chat_message("assistant"):
        response = st.write_stream(get_ai_response(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## 8. Deployment

### 8.1 Docker

| 항목 | 내용 |
|------|------|
| **역할** | 애플리케이션 컨테이너화 |
| **선택 이유** | 환경 일관성, 배포 간편화 |
| **공식 문서** | https://docs.docker.com/ |

**Dockerfile 예시:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 (Playwright용)
RUN apt-get update && apt-get install -y \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright 브라우저 설치
RUN playwright install chromium

# 소스 코드
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data  # ChromaDB 영속성
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

---

### 8.2 AWS EC2 (Free Tier)

| 항목 | 내용 |
|------|------|
| **인스턴스** | t2.micro (1 vCPU, 1GB RAM) |
| **스토리지** | 30GB EBS (Free Tier) |
| **비용** | 월 750시간 무료 (1년) |

**배포 스크립트:**
```bash
#!/bin/bash
# deploy.sh

# EC2 접속
ssh -i "key.pem" ec2-user@your-ec2-ip << 'EOF'
    cd ~/ai-review-analyst
    git pull origin main
    docker-compose down
    docker-compose up -d --build
EOF
```

---

## 9. Dependencies Summary

### requirements.txt
```
# Core
python-dotenv==1.0.0
pydantic==2.5.0

# LLM Framework
langchain==0.1.0
langchain-openai==0.0.5
langchain-chroma==0.0.1
langgraph==0.0.20

# Vector DB
chromadb==0.4.22

# Web Crawling
playwright==1.40.0
beautifulsoup4==4.12.2
lxml==5.1.0

# Frontend
streamlit==1.30.0

# Utilities
tenacity==8.2.3      # 재시도 로직
tiktoken==0.5.2      # 토큰 카운팅
```

---

## 10. 기술 스택 선택 근거 요약

| 레이어 | 선택 기술 | 핵심 선택 이유 |
|--------|-----------|----------------|
| **Language** | Python 3.11 | AI/ML 생태계, 생산성 |
| **LLM Orchestration** | LangChain + LangGraph | 유연한 체인 구성 + 상태 기반 에이전트 |
| **AI Model** | GPT-4o-mini | 비용 효율성 (GPT-4 대비 10배 저렴) |
| **Vector DB** | ChromaDB | 로컬 개발 용이, 설치 간편 |
| **Crawler** | Playwright | 동적 SPA 렌더링 지원 |
| **Frontend** | Streamlit | 빠른 MVP 개발 |
| **Deployment** | Docker + EC2 | 비용 최소화 (Free Tier) |

---

## 참고 자료

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Playwright Python](https://playwright.dev/python/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

*본 문서는 프로젝트 진행에 따라 업데이트됩니다.*
