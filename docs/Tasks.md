# Project Tasks

> AI Review Analyst 프로젝트 작업 관리 문서

**범례:**
- [ ] 미완료
- [x] 완료
- Phase > Epic > Task 계층 구조

---

## Phase 1: Foundation (프로젝트 기반 구축)

### Epic 1.1: 프로젝트 환경 설정

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 1.1.1 | [x] Python 가상환경 생성 | 완료 | `python -m venv venv` |
| 1.1.2 | [x] requirements.txt 작성 | 완료 | 의존성 패키지 정의 |
| 1.1.3 | [x] .env.example 생성 | 완료 | 환경 변수 템플릿 |
| 1.1.4 | [x] .gitignore 설정 | 완료 | venv, .env, __pycache__ 등 |
| 1.1.5 | [x] 프로젝트 디렉토리 구조 생성 | 완료 | src/, tests/, data/ 등 |
| 1.1.6 | [x] config.py 설정 모듈 작성 | 완료 | pydantic-settings 활용 |

**완료 기준:** `pip install -r requirements.txt` 실행 시 에러 없이 설치 완료

---

### Epic 1.2: 웹 크롤러 개발

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 1.2.1 | [x] Playwright 설치 및 브라우저 설정 | 완료 | `playwright install chromium` |
| 1.2.2 | [x] 기본 크롤러 클래스 구현 | 완료 | `src/crawler/base.py` |
| 1.2.3 | [x] 올웨이즈 리뷰 크롤러 구현 | 완료 | 동적 페이지 스크롤 처리 |
| 1.2.4 | [x] 쿠팡 리뷰 크롤러 구현 | 완료 | 플랫폼별 파서 분리 |
| 1.2.5 | [x] 크롤링 에러 핸들링 | 완료 | 재시도, 타임아웃 처리 |
| 1.2.6 | [x] User-Agent 로테이션 구현 | 완료 | 차단 우회 전략 |
| 1.2.7 | [x] 크롤러 단위 테스트 작성 | 완료 | `tests/test_crawler.py` |
| 1.2.8 | [x] playwright-stealth 통합 | 완료 | 봇 탐지 우회 강화 |
| 1.2.9 | [x] 인간적 행동 패턴 구현 | 완료 | 랜덤 딜레이, 스크롤 |

**완료 기준:** URL 입력 시 리뷰 데이터 JSON 형태로 추출 성공

**⚠️ 알려진 한계점:**
- **쿠팡**: 강력한 봇 탐지 시스템(Akamai) 사용으로 headless 브라우저 차단
  - 프록시 서비스 또는 CAPTCHA 해결 서비스 필요
  - 프로덕션 환경에서는 공식 API 또는 파트너십 고려 권장
- **올웨이즈**: 모바일 앱 중심 서비스로 웹 버전 제한적
- **11번가/네이버쇼핑**: 봇 탐지로 접속 차단

**💡 대안:** 크롤링 한계로 인해 **AI Hub 공개 데이터셋**을 활용하여 RAG 파이프라인 및 에이전트 시스템 개발에 집중

---

### Epic 1.3: 공개 데이터셋 통합 (AI Hub) ✅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 1.3.1 | [x] AI Hub 회원가입 및 데이터 신청 | 완료 | 승인 완료 |
| 1.3.2 | [x] 데이터셋 다운로드 | 완료 | `data/aihub_data/` |
| 1.3.3 | [x] 데이터 포맷 변환 | 완료 | `src/pipeline/aihub_loader.py` |
| 1.3.4 | [x] 카테고리별 데이터 분리 | 완료 | 패션/화장품/가전/IT/생활용품 |
| 1.3.5 | [x] 샘플 데이터 생성 | 완료 | `data/aihub_sample*.json` |

**데이터셋 정보:**
- **출처:** [AI Hub - 속성기반 감정분석 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71603)
- **실제 규모:** 225,285건 (쇼핑몰 90% + SNS 10%)
- **카테고리:** 패션, 화장품, 가전, IT기기, 생활용품 (각 약 20%)
- **라벨:** 속성별 감정 태깅 (가격, 디자인, 기능, 품질, 색상 등)
- **형식:** JSON
- **활용:** RAG 학습, 속성별 감성 분석, 상품 비교

**생성된 파일:**
- `src/pipeline/aihub_loader.py`: AI Hub 데이터 로더 클래스
- `data/aihub_sample.json`: 통합 샘플 (1,000건)
- `data/aihub_sample_{카테고리}.json`: 카테고리별 샘플 (각 200건)

---

### Epic 1.4: 데이터 전처리 파이프라인 ✅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 1.4.1 | [x] 리뷰 텍스트 정제 함수 | 완료 | 특수문자, 이모지 처리 |
| 1.4.2 | [x] 텍스트 청킹(Chunking) 로직 | 완료 | 의미 단위 분할 |
| 1.4.3 | [x] 메타데이터 추출 | 완료 | 별점, 날짜, 구매옵션 |
| 1.4.4 | [x] 중복 리뷰 필터링 | 완료 | 해시 기반 중복 제거 |
| 1.4.5 | [x] 전처리 파이프라인 클래스 | 완료 | `src/pipeline/preprocessor.py` |
| 1.4.6 | [x] 전처리 단위 테스트 | 완료 | `tests/test_preprocessor.py` |

**완료 기준:** 원본 리뷰 → 정제된 청크 리스트 변환 정상 동작

**구현 내용:**
- `TextCleaner`: 텍스트 정제 클래스 (이모지, 특수문자, 반복문자, HTML 엔티티 처리)
- `TextChunker`: 텍스트 청킹 클래스 (문장 단위 분할, 오버랩 지원)
- `DuplicateFilter`: 중복 필터링 클래스 (해시 기반 + 퍼지 매칭)
- `ReviewPreprocessor`: 통합 파이프라인 클래스
- 39개 단위 테스트 전체 통과

---

## Phase 2: Core RAG (검색 증강 생성 구현)

### Epic 2.1: Vector Database 구축 ✅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 2.1.1 | [x] ChromaDB 설치 및 초기화 | 완료 | 영속적 저장소 설정 |
| 2.1.2 | [x] OpenAI 임베딩 연동 | 완료 | text-embedding-3-small |
| 2.1.3 | [x] 임베딩 파이프라인 구현 | 완료 | `src/pipeline/embedder.py` |
| 2.1.4 | [x] 컬렉션 관리 유틸리티 | 완료 | 생성, 삭제, 조회 |
| 2.1.5 | [x] 메타데이터 필터링 구현 | 완료 | 별점, 날짜 기반 필터 |
| 2.1.6 | [x] 벡터 검색 테스트 | 완료 | 유사도 검색 정확도 확인 |

**완료 기준:** 자연어 쿼리로 관련 리뷰 Top-K 검색 성공

**구현 내용:**
- `ReviewEmbedder`: 리뷰 임베딩 및 벡터 검색 클래스
- `CollectionManager`: ChromaDB 컬렉션 관리 유틸리티
- `SearchResult`: 검색 결과 데이터 구조
- 메타데이터 필터링 (평점, 날짜) 지원
- LangChain Retriever 통합
- 16개 단위 테스트 통과

---

### Epic 2.2: LangChain RAG Chain 구현 ✅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 2.2.1 | [x] LangChain 기본 설정 | 완료 | ChatOpenAI 초기화 |
| 2.2.2 | [x] Retriever 구성 | 완료 | ChromaDB 연동 |
| 2.2.3 | [x] 프롬프트 템플릿 설계 | 완료 | 시스템/유저 프롬프트 |
| 2.2.4 | [x] RetrievalQA Chain 구현 | 완료 | `src/chains/rag_chain.py` |
| 2.2.5 | [x] 출처(Source) 반환 로직 | 완료 | 참고 리뷰 원문 표기 |
| 2.2.6 | [x] Chain 스트리밍 설정 | 완료 | 실시간 응답 출력 |
| 2.2.7 | [x] RAG Chain 통합 테스트 | 완료 | E2E 질의응답 테스트 |

**완료 기준:** "배송이 빠른가요?" 질문에 리뷰 근거 기반 답변 생성

**구현 내용:**
- `ReviewRAGChain`: RAG 체인 클래스 (query, stream, astream 메서드)
- `RAGConfig`: 설정 데이터클래스 (모델, 온도, top_k 등)
- `RAGResponse`: 응답 데이터클래스 (답변, 출처, 메타데이터)
- 시스템 프롬프트: 리뷰 분석 전문가 역할, 근거 기반 답변 원칙
- 스트리밍 지원 (동기/비동기)
- 23개 단위 테스트 통과

---

### Epic 2.3: 프롬프트 엔지니어링 ✅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 2.3.1 | [x] Q&A 시스템 프롬프트 작성 | 완료 | 역할, 제약조건 정의 |
| 2.3.2 | [x] 요약 프롬프트 작성 | 완료 | 수치화 근거 포함 지시 |
| 2.3.3 | [x] 비교 분석 프롬프트 작성 | 완료 | 장단점 비교 포맷 |
| 2.3.4 | [x] 할루시네이션 방지 지시문 | 완료 | "모르면 모른다고 답변" |
| 2.3.5 | [x] Few-shot 예시 추가 | 완료 | 답변 품질 향상 |
| 2.3.6 | [x] 프롬프트 버전 관리 | 완료 | prompts/ 디렉토리 구성 |

**완료 기준:** 일관된 포맷과 품질의 답변 생성

**구현 내용:**
- `src/prompts/templates.py`: 프롬프트 템플릿 모듈
- `PromptTemplate`: 프롬프트 데이터 구조 (시스템/유저 프롬프트, Few-shot 예시)
- `PromptManager`: 프롬프트 관리 클래스 (등록, 조회, 타입별 필터)
- 4가지 프롬프트: Q&A, 요약, 비교분석, 감성분석
- 할루시네이션 방지 지시문 (ANTI_HALLUCINATION_INSTRUCTIONS)
- RAG Chain 통합 (set_prompt, set_prompt_template 메서드)
- 29개 단위 테스트 통과

---

### Epic 2.4: LLM 기반 속성 추출 시스템

> **목표:** 실제 프로덕션 환경처럼 라벨이 없는 Raw 리뷰에서 LLM을 활용해 속성(Aspect)을 자동 추출하는 시스템

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 2.4.1 | [x] AspectExtractor 클래스 구현 | 완료 | `src/pipeline/aspect_extractor.py` |
| 2.4.2 | [x] 속성 추출 프롬프트 설계 | 완료 | 11개 카테고리 정의 |
| 2.4.3 | [x] 속성별 감정 분석 | 완료 | Structured Output 활용 |
| 2.4.4 | [x] 배치 처리 및 캐싱 | 완료 | MD5 해시 기반 캐싱 |
| 2.4.5 | [x] UI: 속성별 차트 시각화 | 완료 | 속성별 감정 분포 차트 |
| 2.4.6 | [x] UI: 속성 필터 기능 | 완료 | 감정 토글 필터 (긍정/부정/중립) |
| 2.4.7 | [x] UI: 제품 비교 뷰 | 완료 | 속성별 제품 간 비교 |
| 2.4.8 | [x] UI: 사용자 리뷰 추가 기능 | 완료 | Raw 리뷰 → LLM 분석 → DB 저장 |

**완료 기준:** Raw 리뷰 입력 시 속성별 감정 분석 결과 및 시각화 출력

**구현 내용 (Epic 2.4 전체 완료 ✅):**
- `src/pipeline/aspect_extractor.py`: 속성 추출기 모듈
- `AspectExtractor`: LLM 기반 속성 추출 클래스
- `AspectResult`: 추출 결과 데이터 구조
- `Sentiment`, `AspectCategory`: Enum 정의
- Pydantic Structured Output 활용
- 28개 속성 카테고리 지원 (확장됨)
- MD5 해시 기반 캐싱 시스템
- 27개 단위 테스트 통과 (`tests/test_aspect_extractor.py`)

**UI 구현 내용 (2.4.5~2.4.8 완료):**
- **제품 비교 뷰**: 2-4개 제품 동시 비교, 속성별 감정 비교 차트
- **사용자 리뷰 추가**: 별점 입력, LLM 분석, 결과 저장
- `src/pipeline/user_review_store.py`: 사용자 리뷰 저장소 모듈
- `UserReview`, `UserReviewStore` 클래스

**배경:**
- 실제 프로덕션 환경에서는 AI Hub처럼 사전 라벨링된 Aspects 데이터가 없음
- Raw 리뷰 텍스트만으로 속성을 추출하고 감정을 분석해야 함
- 이 시스템으로 어떤 이커머스 리뷰든 분석 가능

**추출 대상 속성 카테고리:**
- 가격/가성비
- 디자인/외관
- 사이즈/치수
- 소재/품질
- 배송/포장
- 색상
- 기능/성능
- 착용감/편안함
- 내구성
- 서비스/응대
- 기타

---

## Phase 3: Agent System (멀티 에이전트 구현)

### Epic 3.1: LangGraph 기반 에이전트 구조 ✅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 3.1.1 | [x] AgentState 정의 | 완료 | `src/agents/state.py` |
| 3.1.2 | [x] Intent Classifier 노드 | 완료 | `src/agents/intent_classifier.py` |
| 3.1.3 | [x] Conditional Router 구현 | 완료 | `src/agents/graph.py` |
| 3.1.4 | [x] StateGraph 구성 | 완료 | `src/agents/graph.py` |
| 3.1.5 | [x] 그래프 시각화 테스트 | 완료 | `get_graph_visualization()` |

**완료 기준:** 질문 유형에 따라 올바른 에이전트로 라우팅

**구현 내용:**
- `AgentState`: TypedDict 상태 스키마 (query, intent, response 등)
- `IntentType`: QA, SUMMARY, COMPARE, UNKNOWN Enum
- Intent Classifier: 하이브리드 방식 (규칙 기반 + LLM 기반)
- StateGraph: START → Intent Classifier → Router → [QA|Summary|Compare] → END

---

### Epic 3.2: 개별 에이전트 구현 ✅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 3.2.1 | [x] Summarize Agent 구현 | 완료 | `src/agents/summarize_agent.py` |
| 3.2.2 | [x] Q&A Agent 구현 | 완료 | `src/agents/qa_agent.py` |
| 3.2.3 | [x] Compare Agent 구현 | 완료 | `src/agents/compare_agent.py` |
| 3.2.4 | [x] 에이전트 공통 인터페이스 | 완료 | `src/agents/base.py` |
| 3.2.5 | [x] 에이전트 단위 테스트 | 완료 | `tests/test_agents.py` (8개 테스트) |

**완료 기준:** 3가지 에이전트가 독립적으로 동작

**구현 내용:**
- `BaseAgent`: 추상 기본 클래스 (RAG Chain 래핑, 프롬프트 설정)
- `QAAgent`: RAG 기반 질의응답 (qa 프롬프트)
- `SummarizeAgent`: 리뷰 요약 (summary 프롬프트, top_k=10)
- `CompareAgent`: 제품 비교 분석 (compare 프롬프트)

---

### Epic 3.3: 가짜 리뷰 필터링 ✅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 3.3.1 | [x] 어뷰징 패턴 정의 | 완료 | 과도한 칭찬, 스팸, 반복 등 |
| 3.3.2 | [x] LLM 기반 분류기 구현 | 완료 | 하이브리드 탐지 (규칙+LLM) |
| 3.3.3 | [x] 필터링 임계값 설정 | 완료 | 가중치 계산 로직 |
| 3.3.4 | [x] 테스트 작성 및 검증 | 완료 | 15개 테스트 통과 |

**완료 기준:** 의심 리뷰 필터링 또는 가중치 감소 적용

**구현 내용:**
- `src/pipeline/fake_review_filter.py`: 가짜 리뷰 필터 모듈
- `FakeReviewFilter`: 하이브리드 탐지 클래스 (규칙 기반 + LLM)
- `FakeReviewReason`: 의심 사유 Enum (7가지)
- `FakeReviewResult`: 판정 결과 데이터클래스

**탐지 패턴:**
- 과도한 칭찬 (인생템, 최고의 제품, 무조건 사세요 등)
- 스팸/광고 키워드 (협찬, 체험단, 카톡 문의 등)
- 반복 패턴 (동일 단어 3회 이상)
- 너무 짧은 리뷰 (10자 미만)
- 구체성 부족 (제품 특성 언급 없음)
- 평점-내용 불일치 (높은 평점 + 부정 내용)

**가중치 계산:**
- 정상 리뷰: 1.0
- 의심 리뷰: confidence에 따라 0.2~0.75
- 스팸 리뷰: 추가 50% 감소

---

## Phase 4: UI & Polish (사용자 인터페이스)

### Epic 4.1: Streamlit 대시보드 ✅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 4.1.1 | [x] 기본 레이아웃 구성 | 완료 | 헤더, 사이드바, 메인 |
| 4.1.2 | [x] 데이터 로더 컴포넌트 | 완료 | AI Hub 데이터 로드 |
| 4.1.3 | [x] 로딩 상태 표시 | 완료 | 스피너, 프로그레스 바 |
| 4.1.4 | [x] 요약 리포트 카드 | 완료 | 긍/부정 비율, 메트릭 |
| 4.1.5 | [x] 차트 시각화 | 완료 | bar chart (평점/감성 분포) |
| 4.1.6 | [x] 채팅 인터페이스 | 완료 | st.chat_message 활용 |
| 4.1.7 | [x] 세션 상태 관리 | 완료 | 대화 히스토리 유지 |

**완료 기준:** 데이터 로드 → 요약 → 채팅 플로우 정상 동작

**버그 수정 (Issue #19):**
- ReviewScore 문자열/None/빈값 파싱 처리
- SNS 데이터 GeneralPolarity 기반 평점 추정
- original_text 전체 저장 (500자 제한 제거)
- RAG 소스에서 original_text 우선 사용
- API 호출 파라미터 수정 (카테고리명, category= 파라미터)

---

### Epic 4.4: 쇼핑몰 스타일 UI 개편 ✅ (Issue #21)

> **목표:** 제품 목록 → 제품 상세 → 리뷰 분석/Q&A 형태의 직관적인 쇼핑몰 UX

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 4.4.1 | [x] Product 데이터클래스 추가 | 완료 | 제품 정보 구조화 |
| 4.4.2 | [x] get_products() 메서드 | 완료 | 제품별 리뷰 그룹화 |
| 4.4.3 | [x] 제품 목록 페이지 | 완료 | 카드 그리드, 필터/정렬 |
| 4.4.4 | [x] 제품 상세 페이지 | 완료 | 4개 탭 구성 |
| 4.4.5 | [x] 제품별 RAG Q&A | 완료 | 해당 제품 리뷰만으로 RAG |

**완료 기준:** 제품 선택 → 상세 분석 → Q&A 플로우 정상 동작

**구현 내용:**
- `src/pipeline/aihub_loader.py`: Product 클래스, get_products() 추가
- `src/ui/app.py`: 쇼핑몰 스타일 UI로 전면 개편

**제품 목록 페이지:**
- 카드 그리드 형태 제품 표시 (3열)
- 카테고리 필터링
- 검색 및 정렬 (리뷰 많은순, 평점 높은순)
- 제품별 평점, 리뷰 수, 감정 요약 표시
- 주요 속성 태그 표시
- ✅ 페이지네이션 (12개/페이지, 상하단 네비게이션)

**제품 상세 페이지 (4개 탭):**
- 📊 요약: 감정 분포 차트, 주요 속성, 자동 인사이트
- 🏷️ 속성 분석: AI Hub 라벨 기반 속성별 감정 분석
  - ✅ 전체 리뷰 텍스트 + 속성 하이라이트 (감정별 색상)
  - ✅ 감정 토글 필터 (긍정/부정/중립)
  - ✅ 페이지네이션 (10개/페이지)
- 💬 Q&A: 해당 제품 리뷰만으로 RAG 기반 질의응답
- 📋 리뷰 목록: 감정 필터링/정렬 가능

**실행:** `streamlit run src/ui/app.py`

---

### Epic 4.2: UX 개선

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 4.2.1 | [ ] 스트리밍 응답 구현 | 대기 | st.write_stream 활용 |
| 4.2.2 | [ ] 에러 메시지 사용자화 | 대기 | 친절한 에러 안내 |
| 4.2.3 | [ ] 출처 리뷰 펼쳐보기 | 대기 | expander 컴포넌트 |
| 4.2.4 | [ ] 분석 결과 다운로드 | 대기 | JSON/CSV 내보내기 |
| 4.2.5 | [ ] 반응형 디자인 조정 | 대기 | 모바일 대응 |
| 4.2.6 | [ ] 다크모드 지원 | 대기 | 테마 설정 |

**완료 기준:** 사용자 피드백 기반 UX 개선 완료

---

### Epic 4.3: 에러 핸들링 및 로깅

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 4.3.1 | [ ] 로깅 설정 | 대기 | logging 모듈 구성 |
| 4.3.2 | [ ] 예외 클래스 정의 | 대기 | 커스텀 Exception |
| 4.3.3 | [ ] API 에러 처리 | 대기 | OpenAI rate limit 등 |
| 4.3.4 | [ ] 크롤링 실패 처리 | 대기 | 재시도, 폴백 |
| 4.3.5 | [ ] 사용자 알림 시스템 | 대기 | st.error, st.warning |

**완료 기준:** 모든 예상 에러 케이스에 대한 graceful 처리

---

## Phase 5: Deployment (배포)

### Epic 5.1: 컨테이너화

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 5.1.1 | [ ] Dockerfile 작성 | 대기 | Python + Playwright |
| 5.1.2 | [ ] docker-compose.yml 작성 | 대기 | 환경 변수 주입 |
| 5.1.3 | [ ] .dockerignore 설정 | 대기 | 불필요 파일 제외 |
| 5.1.4 | [ ] 로컬 Docker 빌드 테스트 | 대기 | `docker-compose up` |
| 5.1.5 | [ ] 볼륨 마운트 설정 | 대기 | ChromaDB 영속성 |

**완료 기준:** 로컬에서 Docker 컨테이너 정상 실행

---

### Epic 5.2: AWS EC2 배포

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 5.2.1 | [ ] EC2 인스턴스 생성 | 대기 | t2.micro (Free Tier) |
| 5.2.2 | [ ] 보안 그룹 설정 | 대기 | 포트 8501 오픈 |
| 5.2.3 | [ ] Docker 설치 | 대기 | EC2에 Docker 세팅 |
| 5.2.4 | [ ] 소스 코드 배포 | 대기 | git clone 또는 SCP |
| 5.2.5 | [ ] 환경 변수 설정 | 대기 | .env 파일 생성 |
| 5.2.6 | [ ] 컨테이너 실행 | 대기 | docker-compose up -d |
| 5.2.7 | [ ] 도메인 연결 (선택) | 대기 | Route53 또는 외부 DNS |

**완료 기준:** 퍼블릭 IP로 서비스 접속 가능

---

### Epic 5.3: 운영 및 모니터링

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 5.3.1 | [ ] 헬스체크 엔드포인트 | 대기 | 서비스 상태 확인 |
| 5.3.2 | [ ] 로그 수집 설정 | 대기 | CloudWatch 또는 파일 |
| 5.3.3 | [ ] 자동 재시작 설정 | 대기 | restart: unless-stopped |
| 5.3.4 | [ ] 백업 스크립트 작성 | 대기 | ChromaDB 데이터 백업 |
| 5.3.5 | [ ] 배포 자동화 (선택) | 대기 | GitHub Actions CI/CD |

**완료 기준:** 안정적인 서비스 운영 환경 구축

---

## Phase 6: Testing & Documentation (테스트 및 문서화)

### Epic 6.1: 테스트

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 6.1.1 | [ ] pytest 설정 | 대기 | pytest.ini, conftest.py |
| 6.1.2 | [ ] 단위 테스트 작성 | 대기 | 각 모듈별 테스트 |
| 6.1.3 | [ ] 통합 테스트 작성 | 대기 | E2E 시나리오 테스트 |
| 6.1.4 | [ ] 테스트 커버리지 측정 | 대기 | pytest-cov |
| 6.1.5 | [ ] RAG 품질 평가 | 대기 | Retrieval Accuracy 측정 |

**완료 기준:** 테스트 커버리지 80% 이상

---

### Epic 6.2: 문서화

| Task ID | Task | 상태 | 설명 |
|---------|------|------|------|
| 6.2.1 | [x] README.md 작성 | 완료 | 프로젝트 소개 |
| 6.2.2 | [x] PRD.md 작성 | 완료 | 요구사항 문서 |
| 6.2.3 | [x] TechStack.md 작성 | 완료 | 기술 스택 문서 |
| 6.2.4 | [x] Tasks.md 작성 | 완료 | 작업 관리 문서 |
| 6.2.5 | [ ] API 문서 작성 | 대기 | 내부 API 명세 |
| 6.2.6 | [ ] 사용자 가이드 작성 | 대기 | 사용법 안내 |

**완료 기준:** 신규 개발자가 문서만으로 프로젝트 이해 가능

---

## Progress Summary

| Phase | Epic 수 | 완료 | 진행률 |
|-------|---------|------|--------|
| Phase 1: Foundation | 4 | 4 | 100% |
| Phase 2: Core RAG | 4 | 4 | 100% |
| Phase 3: Agent System | 3 | 2 | 67% |
| Phase 4: UI & Polish | 4 | 2 | 50% |
| Phase 5: Deployment | 3 | 0 | 0% |
| Phase 6: Testing & Docs | 2 | 1 | 50% |
| **Total** | **20** | **13** | **65%** |

---

## Quick Start Checklist

프로젝트 시작 시 우선 완료해야 할 최소 태스크:

- [x] 1.1.1 ~ 1.1.6: 환경 설정 ✅
- [x] 1.2.1 ~ 1.2.9: 웹 크롤러 ✅ (봇 탐지로 제한적)
- [x] 1.3.1 ~ 1.3.5: AI Hub 데이터셋 통합 ✅
- [x] 1.4.1 ~ 1.4.6: 데이터 전처리 ✅
- [x] 2.1.1 ~ 2.1.6: 벡터 DB 연동 ✅
- [x] 2.2.1 ~ 2.2.7: RAG Chain ✅
- [x] 4.1.1 ~ 4.1.7: 기본 UI ✅

→ **MVP 데모 가능!** 🎉

**추가 완료 사항:**
- [x] 4.4.1 ~ 4.4.5: 쇼핑몰 스타일 UI ✅ (Issue #21)
- [x] 3.1.1 ~ 3.1.5: LangGraph 에이전트 구조 ✅
- [x] 3.2.1 ~ 3.2.5: 개별 에이전트 구현 ✅

---

## Notes

- 각 Phase는 순차적으로 진행하되, 독립적인 Epic은 병렬 진행 가능
- Task 완료 시 체크박스 업데이트 및 커밋
- 예상치 못한 이슈 발생 시 해당 Epic에 Task 추가

---

*최종 업데이트: 2026-01-12 (Epic 3.1~3.2 완료: LangGraph 멀티 에이전트 시스템)*
