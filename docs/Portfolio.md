# Problem Solver 포트폴리오

> **"비즈니스 임팩트를 만들어내는 AI Problem Solver, 안성재입니다."**

[GitHub](https://github.com/sammy0329) · [Email](mailto:sammy8305@gmail.com) · [Blog](https://velog.io/@sammy0329/series)

---

## 1. Core Identity

| 키워드 | 설명 |
|--------|------|
| **Execution** | 아이디어 구상부터 MVP 배포까지 **4일 만에 완성** |
| **AI Native** | LangChain + LangGraph 기반 멀티 에이전트 시스템 구축 |
| **Vibe Coding** | Phase → Epic → Task 구조로 Claude AI와 협업, 체계적 프로젝트 진행 |
| **Problem Solving** | "리뷰 30분 → 30초" 문제를 RAG 파이프라인으로 해결 |

---

## 2. Key Project: AI Review Analyst

> **"리뷰를 읽는 시간 30분을 30초로 단축하다."**

- **GitHub**: [ai-review-analyst](https://github.com/sammy0329/ai-review-analyst)
- **Live Demo**: [AI Review Analyst](http://ec2-3-38-252-10.ap-northeast-2.compute.amazonaws.com:8501/)
- **Video Demo**: [YouTube에서 보기](https://youtu.be/WWrGmhjnR6Q)

---

### 2.1 Problem (문제 정의)

초저가/발견형 커머스(올웨이즈, 쿠팡) 사용자들이 겪는 핵심 불편함:

| Problem | Description | 영향 |
|---------|-------------|------|
| **Information Overload** | 인기 상품 리뷰 500~2,000개, 일일이 읽기 불가능 | 구매 결정 지연 |
| **Trust Issues** | 광고성 리뷰 vs 진성 리뷰 구분 어려움 | 구매 전환율 하락 |
| **Lack of Context** | 별점 4.5점이어도 "내 상황"에 맞는지 판단 불가 | 이탈률 증가 |

**데이터 기반 관찰:**
- **올웨이즈:** 초저가 + 발견형 쇼핑 → 처음 보는 브랜드/상품이라 리뷰 의존도 높음
- **일반 커머스:** 베스트셀러 평균 리뷰 1,200개, 사용자는 15~20개만 확인 (1.5%)
- **결론:** 98.5%의 정보가 구매 결정에 반영되지 않음

---

### 2.2 Solution (핵심 기능)

#### F1. 시맨틱 Q&A 챗봇 (RAG Chatbot)

사용자가 **"민감성 피부인데 자극 있나요?"** 라고 물으면, 1,200개 리뷰에서 관련 리뷰를 벡터 검색하여 답변.

| 기존 방식 | AI Agent 해결책 |
|-----------|-----------------|
| 키워드 필터링 (정확도 낮음) | LLM 기반 **시맨틱 검색**으로 문맥 이해 |
| 일반적 요약 | **사용자 질문에 맞춤형** 답변 |
| 출처 불명확 | **[참고한 리뷰 원문 보기]** 제공 → 할루시네이션 불안 해소 |

#### F2. 팩트 기반 리뷰 요약 (Fact-based Summarization)

단순히 "좋아요"가 아닌, **수치화된 근거(Quantitative Evidence)** 제시.

> _"리뷰의 82%가 보습력이 좋다고 언급했습니다"_

#### F3. 가짜 리뷰 필터링 (Anti-Spam Filter)

7가지 패턴(과도한 칭찬, 스팸 키워드, 평점-내용 불일치 등)을 탐지하여 가중치 감소.
- 삭제하지 않고 **[의심] 라벨**로 투명하게 표시

#### F4. 속성별 감정 분석 (Aspect Extraction)

Raw 리뷰에서 속성(보습력, 자극, 향 등)을 자동 추출하고 감정 분석.

```
입력: "보습력은 좋은데 향이 너무 강해요. 가격은 적당해요."
출력: 보습력(긍정), 향(부정), 가격(중립)
```

#### F5. 쇼핑몰 스타일 제품 탐색

실제 쇼핑몰처럼 **제품 목록 → 제품 상세 → AI 분석/Q&A** 형태의 직관적 UX.

---

### 2.3 Process (실행 과정)

#### MVP 개발 타임라인 (4일)

| Day | Task | Output |
|-----|------|--------|
| 1 | 크롤링 시도 → 봇 탐지 차단 → **피봇 결정** | AI Hub 225K 리뷰 확보 |
| 2 | RAG 파이프라인 + Streamlit UI + 속성 추출 | **핵심 기능 완성** |
| 3 | LangGraph 멀티 에이전트 시스템 | 의도 분류 → 에이전트 라우팅 |
| 4 | 가짜 리뷰 필터링 + UX 개선 + 에러 핸들링 | **MVP 배포 완료** |

**피봇 의사결정:** 쿠팡/네이버 봇 탐지 차단 → 같은 날 AI Hub 공개 데이터셋으로 전환 결정.
오히려 속성별 감정 라벨링이 완료된 더 좋은 데이터 확보.

---

### 2.4 Result (검증 결과)

| 지표 | Before | After | 개선율 |
|------|--------|-------|--------|
| 정보 탐색 시간 | 30분 (20개 리뷰 수동) | 30초 (Q&A 1회) | **98% 감소** |
| 검색 범위 | 1.5% (20/1,200개) | 100% (전체 벡터 검색) | **66배 확장** |
| 구매 결정 근거 | 주관적 인상 | 팩트 기반 + 출처 | 신뢰도 향상 |

**품질 검증:** 253개 테스트 100% 통과 → 안정적인 프로덕션 배포 가능

---

## 3. Tech Stack

| Category | Technology | 선택 이유 |
|----------|------------|-----------|
| **LLM** | GPT-4o-mini | 비용 효율성과 속도의 균형 |
| **Framework** | LangChain + LangGraph | Agent & Chain 관리, State 기반 흐름 제어 |
| **Vector DB** | ChromaDB | 경량화, 로컬 개발 용이 |
| **Embedding** | text-embedding-3-small | 비용 효율적 임베딩 |
| **Backend** | Python 3.12, SQLite | 빠른 프로토타이핑 |
| **Frontend** | Streamlit | MVP용 Rapid Prototyping |
| **Deployment** | AWS EC2 | Free Tier 활용 |

---

## 4. Why Levit?

### 레브잇 문화와 나의 핏(Fit)

| 레브잇 문화 | 나의 경험 |
|-------------|----------|
| **Aim Higher and Find a Way** | 크롤링 차단 → 같은 날 AI Hub로 피봇, **오히려 더 좋은 데이터 확보** |
| **Move Fast and Learn** | 아이디어 → MVP 배포까지 **4일** |
| **Problem Solver** | "리뷰 30분" 문제 → RAG 에이전트로 **30초**로 해결 |

### 레브잇에서 하고 싶은 일

**목표:** 올웨이즈 사용자의 "구매 고민(Hesitation)"을 끝내는 AI 에이전트 개발

**기여:**
1. 이미 검증된 RAG 파이프라인 → 올웨이즈 데이터에 적용
2. 가설 → MVP → 검증 → 피봇 사이클에 익숙
3. 4일 만에 MVP 완성 가능한 실행력

> 올웨이즈 회원이 **"이거 사도 될까?"** 고민하는 시간을 없애고 싶습니다.

---

*"아이디어는 누구나 가질 수 있지만, 실행은 아무나 못 합니다. 저는 실행하는 사람입니다."*
