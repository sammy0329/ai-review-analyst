# AI Review Analyst: Smart Buying Decision Agent

> **"리뷰를 읽는 시간 30분을 30초로 단축하다."**
>
> RAG(검색 증강 생성) 기반 이커머스 리뷰 분석 및 구매 의사결정 지원 에이전트

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-v0.3-green?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-blue?style=flat-square)
![AIHub](https://img.shields.io/badge/Dataset-AI_Hub_250K-yellow?style=flat-square)

---

## Project Overview

**AI Review Analyst**는 이커머스 플랫폼(올웨이즈, 쿠팡 등)의 방대한 리뷰 데이터를 분석하여, 사용자가 **구매 고민(Hesitation)**을 끝내고 **구매 확신(Conviction)**을 갖도록 돕는 AI 에이전트입니다.

단순한 '3줄 요약'을 넘어, 사용자의 상황(육아, 자취, 특정 취향)에 맞는 정보를 **팩트 기반(Fact-based)**으로 답변하여 **이탈률 감소**와 **구매 전환율(CVR) 증대**를 목표로 합니다.

---

## The Problem (Why I built this)

초저가/발견형 커머스 사용자들은 다음과 같은 **Pain Point**를 겪고 있습니다.

| Problem | Description |
|---------|-------------|
| **Information Overload** | 인기 상품의 수천 개 리뷰를 일일이 읽기엔 시간이 부족함 |
| **Trust Issues** | 광고성 리뷰와 진성 리뷰가 섞여 있어 품질을 확신하기 어려움 |
| **Lack of Context** | 별점 4.5점이라도, "내 상황(예: 매운 걸 못 먹음)"에 맞는지 판단 불가 |

---

## The Solution

**RAG(Retrieval-Augmented Generation)** 기술을 활용해 비정형 리뷰 데이터를 구조화하고, LLM이 '나만의 쇼핑 비서'처럼 행동합니다.

| Feature | Description | Tech Key |
|:--------|:------------|:---------|
| **쇼핑몰 스타일 UI** | 제품 목록 → 제품 상세 → 리뷰 분석 흐름의 직관적 인터페이스 | `Streamlit`, `UX Design` |
| **팩트 기반 요약** | 긍/부정 비율 시각화 및 키워드별 장단점(맛, 배송, 가성비) 분석 | `Prompt Engineering` |
| **시맨틱 Q&A** | "이거 3살 아기가 먹어도 돼?" 같은 자연어 질문에 리뷰 근거로 답변 | `RAG`, `Vector DB` |
| **비교 분석** | A상품(가성비) vs B상품(고품질) 중 내게 맞는 상품 추천 | `Multi-Agent`, `Reasoning` |
| **속성 추출** | Raw 리뷰에서 가격/디자인/품질/배송 등 속성별 감정 자동 분석 | `LLM Prompting`, `Structured Output` |
| **제품별 RAG Q&A** | 선택한 제품의 리뷰만을 기반으로 질문에 답변 | `Scoped RAG`, `Context Filtering` |
| **할루시네이션 방지** | 답변 생성 시 참고한 실제 리뷰 원문(출처) 표기 | `Source Citation` |

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Core Logic** | Python, LangChain (Orchestration), LangGraph (Flow Control) |
| **AI Model** | OpenAI GPT-4o-mini (Cost-effective reasoning) |
| **Database** | ChromaDB (Vector Store for semantic search) |
| **Data Pipeline** | Playwright (Dynamic crawling), BeautifulSoup |
| **UI/UX** | Streamlit (Rapid MVP prototyping) |
| **Deployment** | Docker, AWS EC2 |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [User Input]                                                           │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────┐     ┌─────────────────────────────────────────┐       │
│   │  Streamlit  │────▶│           LangChain Orchestration        │       │
│   │     UI      │     │                                          │       │
│   └─────────────┘     │   ┌─────────────────────────────────┐   │       │
│                       │   │         LangGraph Router         │   │       │
│   ┌─────────────┐     │   └──────────┬──────────┬───────────┘   │       │
│   │  Playwright │     │              │          │               │       │
│   │   Crawler   │─────┤              ▼          ▼               │       │
│   └─────────────┘     │   ┌──────────────┐ ┌──────────────┐     │       │
│                       │   │   Summarize  │ │    Q&A RAG   │     │       │
│   ┌─────────────┐     │   │    Agent     │ │    Agent     │     │       │
│   │  ChromaDB   │◀────┤   └──────────────┘ └──────────────┘     │       │
│   │(Vector Store)│    │              │          │               │       │
│   └─────────────┘     │              ▼          ▼               │       │
│                       │        ┌─────────────────────┐          │       │
│   ┌─────────────┐     │        │    GPT-4o-mini      │          │       │
│   │   OpenAI    │◀────┤        │    (Generation)     │          │       │
│   │     API     │     │        └─────────────────────┘          │       │
│   └─────────────┘     └─────────────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Source:** AI Hub 속성기반 감정분석 데이터 (250K+ 이커머스 리뷰)
2. **Embedding:** 텍스트 청킹(Chunking) 후 ChromaDB에 벡터 저장
3. **Retrieval:** 사용자 질문(Query)과 유사한 리뷰 맥락 검색
4. **Generation:** 검색된 맥락을 바탕으로 LLM이 답변 및 근거 생성

### Data Source

| Source | Reviews | Category | Status |
|--------|---------|----------|--------|
| **AI Hub 속성기반 감정분석** | 250,312 | 패션/화장품/가전/IT/생활용품 | 🔄 신청 예정 |
| Coupang Crawler | - | - | ⚠️ 봇 탐지로 제한 |
| 11st/Naver Crawler | - | - | ⚠️ 봇 탐지로 제한 |

> **Note:** 주요 이커머스 플랫폼의 봇 탐지 시스템으로 인해 [AI Hub 속성기반 감정분석 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71603)를 활용하여 RAG 파이프라인 및 에이전트 시스템을 개발합니다.

---

## Project Structure

```
ai-review-analyst/
├── README.md
├── docs/
│   ├── PRD.md                 # Product Requirements Document
│   ├── Tasks.md               # 작업 관리 문서
│   └── TechStack.md           # 기술 스택 문서
├── data/
│   └── sample_reviews.json    # 테스트용 샘플 데이터
├── src/
│   ├── __init__.py
│   ├── config.py              # 설정 모듈
│   ├── crawler/               # Web scraping module
│   │   ├── __init__.py
│   │   ├── base.py            # 기본 크롤러 클래스
│   │   ├── coupang.py         # 쿠팡 크롤러
│   │   └── always.py          # 올웨이즈 크롤러
│   ├── pipeline/              # Data processing
│   │   ├── __init__.py
│   │   ├── aihub_loader.py    # AI Hub 데이터 로더
│   │   ├── aspect_extractor.py # LLM 속성 추출기
│   │   ├── preprocessor.py
│   │   └── embedder.py
│   ├── prompts/               # 프롬프트 템플릿
│   │   ├── __init__.py
│   │   └── templates.py       # Q&A, 요약, 비교, 감성분석 프롬프트
│   ├── agents/                # LangGraph agents
│   │   ├── __init__.py
│   │   ├── summarize_agent.py
│   │   ├── qa_agent.py
│   │   └── compare_agent.py
│   ├── chains/                # LangChain chains
│   │   ├── __init__.py
│   │   └── rag_chain.py
│   ├── ui/                    # Streamlit UI 모듈
│   │   ├── __init__.py
│   │   └── app.py             # 쇼핑몰 스타일 대시보드
│   └── utils/
│       └── __init__.py
├── tests/
│   └── test_crawler.py        # 크롤러 테스트
├── scripts/
│   └── debug_crawl.py         # 디버그 스크립트
├── app.py                     # Streamlit application
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API Key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-review-analyst.git
cd ai-review-analyst

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Playwright browsers
playwright install chromium

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running the Application

```bash
# Run Streamlit app (쇼핑몰 스타일 UI)
streamlit run src/ui/app.py

# 또는 루트의 엔트리포인트 사용
streamlit run app.py
```

---

## Key Features Demo

### 1. Product Explorer (쇼핑몰 스타일 UI)
```
[제품 목록 화면]
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ 제품 A      │ │ 제품 B      │ │ 제품 C      │
│ ⭐ 4.2/5.0  │ │ ⭐ 3.8/5.0  │ │ ⭐ 4.5/5.0  │
│ 리뷰 128개  │ │ 리뷰 89개   │ │ 리뷰 256개  │
│ 긍정 75%    │ │ 긍정 62%    │ │ 긍정 88%    │
└─────────────┘ └─────────────┘ └─────────────┘

[제품 상세 화면]
- 요약 탭: 평점, 감정 분포, 키워드
- 속성 분석 탭: 가격/품질/배송 등 속성별 감정
- Q&A 탭: 해당 제품 리뷰 기반 RAG 질의응답
- 리뷰 목록 탭: 전체 리뷰 브라우징
```

### 2. Fact-based Summarization
```
Input: 상품 URL 입력
Output:
- 긍정/부정 리뷰 비율: 82% / 18%
- 주요 키워드: 배송(빠름), 맛(달콤함), 가성비(좋음)
- 3줄 요약: "대부분의 리뷰어가 배송 속도와 맛에 만족..."
```

### 3. Semantic Q&A (제품별 RAG)
```
[특정 제품 상세 페이지에서]
User: "이거 3살 아기가 먹어도 괜찮아?"
Agent: "12개의 리뷰에서 '아이', '유아' 관련 언급을 찾았습니다.
       8개 리뷰가 긍정적이며, 주요 내용은..."
       [참고 리뷰 원문 보기]

→ 해당 제품의 리뷰만 검색하여 정확한 답변 제공
```

### 4. Product Comparison
```
Input: 상품A URL, 상품B URL
Output:
- A상품: 맛 우수(87%), 배송 느림(평균 5일)
- B상품: 맛 보통(65%), 로켓배송(1일)
- 추천: "빠른 배송을 원하시면 B상품을 추천드립니다."
```

### 5. Aspect Extraction (속성 추출)
```
Input: "가격은 좀 비싸지만 소재가 정말 좋아요. 배송도 빨랐습니다."
Output:
- 가격: 부정 (비싸다)
- 소재/품질: 긍정 (좋다)
- 배송: 긍정 (빠르다)

→ 제품별 속성 감정 분포 차트 시각화
→ 여러 제품 속성별 비교 분석
```

---

## Development Roadmap

### Phase 1: Foundation (100% 완료)
- [x] Project setup & documentation
- [x] Playwright crawler implementation (봇 탐지로 제한적)
- [x] AI Hub 공개 데이터셋 통합 (225K+ 이커머스 리뷰)
- [x] Data preprocessing pipeline

### Phase 2: Core RAG (100% 완료)
- [x] ChromaDB integration & embedding pipeline
- [x] LangChain RAG chain
- [x] Prompt engineering (Q&A, 요약, 비교, 감성분석)
- [x] LLM 기반 속성 추출 시스템 (27개 테스트 통과)

### Phase 3: Dashboard & UI (진행 중)
- [x] 기본 Streamlit 대시보드
- [x] 쇼핑몰 스타일 UI 개편 (제품 목록 → 상세 → Q&A)
- [x] 제품별 RAG Q&A 기능
- [ ] 속성 분석 시각화 개선

### Phase 4-5: Agent & Deployment
- [ ] LangGraph multi-agent system
- [ ] Docker & AWS EC2 deployment

---

## Target Position

이 프로젝트는 [레브잇 Problem Solver AI Agent 인턴](https://www.wanted.co.kr/wd/308575) 포지션 지원을 위해 개발되었습니다.

### Demonstrated Skills

| Skill | Implementation |
|-------|----------------|
| **LLM/RAG** | LangChain + ChromaDB 기반 검색 증강 생성 |
| **Agent Development** | LangGraph를 활용한 멀티 에이전트 시스템 |
| **Prompt Engineering** | 팩트 기반 요약 및 출처 명시 프롬프트 설계 |
| **Web Crawling** | Playwright + Stealth 기반 동적 페이지 크롤링 |
| **Data Pipeline** | HuggingFace 데이터셋 통합 및 벡터 임베딩 |
| **Problem Solving** | 크롤링 한계 → 공개 데이터셋 활용 전략 수립 |
| **Rapid Prototyping** | Streamlit MVP 개발 |

---

## Documentation

- [PRD (Product Requirements Document)](./docs/PRD.md)
- [Tech Stack (기술 스택 상세)](./docs/TechStack.md)
- [Tasks (작업 관리)](./docs/Tasks.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **GitHub:** [@yourusername](https://github.com/yourusername)
- **Email:** your.email@example.com

---

*Built with passion for AI-powered problem solving*
