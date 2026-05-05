# 🛒 Optima — 쿠팡 리뷰 불만 분석 대시보드

> 쿠팡 상품 리뷰를 자동 수집·분류하고, 판매자가 즉시 액션을 취할 수 있는 리스크 조기경보 대시보드

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://optima-dashboard-uyusympgpfjzb7hiszwtx8.streamlit.app)

---

## 📌 프로젝트 개요

이커머스 플랫폼 판매자는 수백~수천 개의 리뷰를 수동으로 확인해야 합니다.
**Optima**는 리뷰 데이터를 자동으로 수집·분류하고, 불만 유형별 트렌드와 리스크 지표를 대시보드로 제공합니다.

| 항목 | 내용 |
|------|------|
| 데이터 | 쿠팡 5개 카테고리 리뷰 **99,850건** |
| 기간 | 2022.01 ~ 2025.12 |
| 카테고리 | Appliances · Baby · Beauty · Pet · Toys |
| 분류 클래스 | 결함/불량 · 배송/CS · 사용성 · 적합성 · 해당없음 |

---

## 🎯 핵심 기능

### 1. 자동 불만 분류
키워드 기반 룰 엔진으로 리뷰를 5개 불만 유형으로 자동 분류

```
결함/불량  → 파손, 불량, 효과없음, 내구성 불량 등
배송/CS    → 배송지연, 오배송, 환불 문제 등
사용성     → 설치어려움, 작동불량, 매뉴얼 부재 등
적합성     → 사이즈 불일치, 피부 트러블, 연령 부적합 등
해당없음   → 일반 리뷰
```

### 2. 리뷰 신호 지표 (REM Score)
불만율, 저평점, 리뷰량 변동, 불만 유형 전환 등을 종합한 **0~100점 리스크 지표**

```
위험 (0~33)   → 즉각 대응 필요
주의 (33~66)  → 모니터링 강화
정상 (66~100) → 안정적 상태
```

### 3. 상품별 추천 액션
신호 점수와 불만 유형을 기반으로 판매자에게 **구체적인 대응 액션 자동 생성**

---

## 🏗️ 파이프라인

```
[쿠팡 리뷰 크롤링]          [전처리 & 분류]           [지표 계산]
Selenium + ChromeDriver  →  키워드 분류기 (5-class)  →  일별 집계
                             ↓                          REM Score 산출
                         [모델 비교 실험]                    ↓
                         KoELECTRA / KoBERT          [대시보드]
                         KcELECTRA                   Streamlit Cloud
```

---

## 📊 대시보드 구성

| 탭 | 내용 |
|----|------|
| 📊 전체 현황 | KPI 지표, 카테고리별 신호 점수, 불만 히트맵 |
| 📈 일별 추이 | 불만율 추이, 유형별 일별 건수, 신호 지표 변화 |
| ☁️ 키워드 분석 | 불만 유형별 워드클라우드, 상위 키워드 빈도 |
| 🔍 상품별 리뷰 | 상품 선택 → 신호 점수, 키워드, 리뷰 테이블 |
| 💡 추천 액션 | 상품별 리스크 요약 및 구체적 대응 가이드 |

---

## 🛠️ 기술 스택

| 분야 | 기술 |
|------|------|
| 크롤링 | Python, Selenium, undetected-chromedriver |
| 데이터 처리 | Pandas, NumPy |
| NLP / 분류 | 키워드 룰 엔진, KoELECTRA, KoBERT, KcELECTRA (Hugging Face) |
| 데이터베이스 | MySQL (로컬), SQLAlchemy |
| 시각화 | Streamlit, Matplotlib, WordCloud |
| 배포 | GitHub, Streamlit Cloud |

---

## 🧪 모델 비교 실험

키워드 기반 분류기와 사전학습 언어모델 3종을 비교 (Macro F1 기준)

| 모델 | Macro F1 |
|------|----------|
| Keyword Baseline | ~0.32 |
| KoELECTRA-base | ~0.33 |
| KoBERT | ~0.32 |
| KcELECTRA-base | ~0.31 |

> 소규모 라벨 데이터(3,000건) 환경에서 딥러닝 모델이 키워드 기반과 유사한 성능 → **키워드 고도화** 채택

---

## 📁 프로젝트 구조

```
optima-dashboard/
├── 01_crawling/          # 쿠팡 리뷰 크롤러
├── 02_preprocessing/     # 분류기, 모델 비교 실험
├── 03_metrics/           # REM 지표 계산
├── 04_eda/               # 탐색적 데이터 분석
├── 05_dashboard/         # Streamlit 대시보드
│   ├── streamlit_app.py
│   └── requirements.txt
├── data/processed/       # 전처리 완료 데이터
├── fonts/                # 한국어 폰트 (NanumGothic)
└── db/                   # DB 스키마 & import 스크립트
```

---

## 🚀 로컬 실행

```bash
# 의존성 설치
pip install -r 05_dashboard/requirements.txt

# 대시보드 실행
streamlit run 05_dashboard/streamlit_app.py
```

---

## 🔗 링크

- **라이브 데모**: https://optima-dashboard-uyusympgpfjzb7hiszwtx8.streamlit.app
- **GitHub**: https://github.com/eve29th/optima-dashboard
