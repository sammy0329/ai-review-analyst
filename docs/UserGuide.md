# User Guide

> AI Review Analyst 사용자 가이드

---

## 1. 소개

AI Review Analyst는 이커머스 리뷰를 AI로 분석하여 구매 의사결정을 도와주는 도구입니다.

### 주요 기능
- **리뷰 요약**: 수백 개의 리뷰를 한눈에 파악
- **Q&A**: 자연어로 궁금한 점 질문
- **비교 분석**: 장단점 분석
- **가짜 리뷰 필터링**: 신뢰할 수 있는 리뷰만 분석

---

## 2. 설치 방법

### 2.1 사전 요구사항

- Python 3.9 이상
- OpenAI API Key

### 2.2 설치

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/ai-review-analyst.git
cd ai-review-analyst

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY 설정
```

### 2.3 환경 변수

`.env` 파일에 다음 설정이 필요합니다:

```
OPENAI_API_KEY=sk-your-api-key-here
```

---

## 3. 애플리케이션 실행

### 3.1 Streamlit 앱 실행

```bash
streamlit run src/ui/app.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

### 3.2 Docker로 실행

```bash
# Docker Compose 사용
docker-compose up -d

# 접속
# http://localhost:8501
```

---

## 4. 사용법

### 4.1 제품 탐색

1. **카테고리 선택**: 사이드바에서 대분류(패션, 화장품 등) 선택
2. **소분류 선택**: 세부 카테고리 선택 (선택사항)
3. **검색**: 제품명으로 검색
4. **정렬**: 리뷰 많은순, 평점 높은순 등으로 정렬

### 4.2 제품 상세 보기

제품 카드를 클릭하면 상세 페이지로 이동합니다.

**탭 구성:**
| 탭 | 설명 |
|---|---|
| 요약 | 평점, 감정 분포, 전체 요약 |
| 속성 분석 | 배송/품질/가격 등 속성별 감정 |
| Q&A | AI에게 질문하기 |
| 리뷰 목록 | 개별 리뷰 확인 |

### 4.3 AI Q&A 사용

Q&A 탭에서 자연어로 질문할 수 있습니다.

**예시 질문:**
- "배송은 빠른가요?"
- "사이즈가 작은 편인가요?"
- "가성비가 좋나요?"
- "이 제품 장단점 알려줘"
- "3살 아이가 먹어도 괜찮을까요?"

### 4.4 질문 유형별 처리

AI가 질문 의도를 자동으로 파악합니다:

| 질문 유형 | 예시 | 처리 방식 |
|----------|------|----------|
| 질의응답 | "배송 빠른가요?" | 관련 리뷰 검색 후 답변 |
| 요약 | "리뷰 요약해줘" | 전체 리뷰 종합 분석 |
| 비교 | "장단점 비교해줘" | 장점/단점 분류 분석 |

---

## 5. 데이터 준비

### 5.1 AI Hub 데이터 사용

기본적으로 AI Hub 속성기반 감정분석 데이터를 사용합니다.

**데이터 다운로드:**
1. [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71603) 접속
2. 데이터셋 신청 및 다운로드
3. `data/aihub/` 디렉토리에 압축 해제

**지원 카테고리:**
- 패션 (여성의류, 남성의류, 잡화 등)
- 화장품 (스킨케어, 메이크업 등)
- 가전 (생활가전, 계절가전 등)
- IT기기 (스마트폰, 태블릿 등)
- 생활용품 (주방용품, 생활잡화 등)

### 5.2 커스텀 데이터 사용

자체 리뷰 데이터를 사용할 수도 있습니다.

```python
from src.crawler.base import Review
from src.pipeline.preprocessor import create_default_preprocessor
from src.pipeline.embedder import create_embedder

# 1. 리뷰 데이터 준비
reviews = [
    Review(text="좋은 제품입니다.", rating=5.0, date="2024-01-01"),
    Review(text="배송이 느렸어요.", rating=2.0, date="2024-01-02"),
]

# 2. 전처리
preprocessor = create_default_preprocessor()
processed = preprocessor.process_batch(reviews)

# 3. 임베딩 저장
embedder = create_embedder(collection_name="my_reviews")
embedder.add_reviews(processed)
```

---

## 6. 고급 기능

### 6.1 프롬프트 커스터마이징

기본 프롬프트를 수정하여 응답 스타일을 변경할 수 있습니다.

```python
from src.chains import create_rag_chain

rag_chain = create_rag_chain(embedder=embedder)

# 프롬프트 변경
rag_chain.set_prompt("summary")  # 요약 모드
rag_chain.set_prompt("sentiment")  # 감성 분석 모드
```

### 6.2 검색 설정 변경

```python
# 검색 결과 수 조정
rag_chain.update_config(top_k=10)

# 검색 방식 변경 (MMR - 다양성 확보)
rag_chain.update_config(search_type="mmr")
```

### 6.3 스트리밍 응답

실시간으로 응답을 받을 수 있습니다.

```python
for chunk in rag_chain.stream("이 제품 추천하시나요?"):
    print(chunk, end="", flush=True)
```

---

## 7. 트러블슈팅

### 7.1 API 키 오류

```
openai.OpenAIError: The api_key client option must be set
```

**해결:**
1. `.env` 파일에 `OPENAI_API_KEY` 설정 확인
2. 환경변수가 로드되었는지 확인

```python
from dotenv import load_dotenv
load_dotenv()
```

### 7.2 ChromaDB 오류

```
chromadb.errors.InvalidCollectionException
```

**해결:**
1. 컬렉션 재생성

```python
embedder.reset_collection()
embedder.add_reviews(reviews)
```

### 7.3 메모리 부족

대량의 리뷰 처리 시 메모리가 부족할 수 있습니다.

**해결:**
1. 배치 크기 조정

```python
# 작은 배치로 분할 처리
for batch in batched(reviews, 100):
    embedder.add_reviews(batch)
```

### 7.4 응답이 느린 경우

**해결:**
1. `top_k` 값 줄이기 (기본값: 5)
2. 스트리밍 모드 사용
3. 더 빠른 모델 사용 (`gpt-4o-mini` 권장)

---

## 8. FAQ

### Q: 어떤 언어를 지원하나요?
A: 현재 한국어만 지원합니다. 한국어 리뷰 데이터에 최적화되어 있습니다.

### Q: API 비용은 얼마나 드나요?
A: GPT-4o-mini 기준 약 $0.15/1M 입력 토큰입니다. 일반적인 사용 시 월 몇 달러 수준입니다.

### Q: 자체 리뷰 데이터를 사용할 수 있나요?
A: 네, `Review` 객체로 변환하여 사용할 수 있습니다. [5.2 커스텀 데이터 사용](#52-커스텀-데이터-사용) 참고.

### Q: 응답 품질이 좋지 않은 경우?
A: `top_k` 값을 늘리거나 (더 많은 리뷰 참조), 더 구체적인 질문을 해보세요.

### Q: 오프라인에서 사용 가능한가요?
A: LLM 응답 생성에는 OpenAI API가 필요합니다. 임베딩 검색만은 로컬에서 가능합니다.

---

## 9. 피드백 및 문의

- **GitHub Issues**: [https://github.com/yourusername/ai-review-analyst/issues](https://github.com/yourusername/ai-review-analyst/issues)
- **Email**: your.email@example.com

---

*본 가이드는 프로젝트 업데이트에 따라 지속적으로 갱신됩니다.*
