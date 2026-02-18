# 🌐 Global Multi-Asset Risk Scoring Model
**글로벌 멀티에셋 리스크 스코어링 모델**

> IGIS Asset Management Intern Project
> Bloomberg Terminal 데이터를 활용한 **오늘의** Risk On/Off 레짐 판별 모델

---

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [주요 기능](#2-주요-기능)
3. [환경 설정 및 설치](#3-환경-설정-및-설치)
4. [파일 구조](#4-파일-구조)
5. [모델 설계 상세](#5-모델-설계-상세)
   - 5.1 [캐너리 자산 유니버스](#51-캐너리-자산-유니버스)
   - 5.2 [매크로 필터](#52-매크로-필터)
   - 5.3 [SKEW 패널티 로직](#53-skew-패널티-로직)
   - 5.4 [레짐 분류 기준](#54-레짐-분류-기준)
6. [클래스 구조 및 메서드](#6-클래스-구조-및-메서드)
7. [사용 방법](#7-사용-방법)
8. [출력 결과물](#8-출력-결과물)
9. [설계 의도 및 주요 고려사항](#9-설계-의도-및-주요-고려사항)
10. [알려진 데이터 이슈](#10-알려진-데이터-이슈)
11. [한계 및 향후 개선 방향](#11-한계-및-향후-개선-방향)

---

## 1. 프로젝트 개요

글로벌 금융시장의 **리스크 온(Risk On) / 리스크 오프(Risk Off)** 레짐을 **오늘** 기준으로 수치화하는 복합 스코어링 모델입니다.

FX, 신흥국 채권/통화, 금리, 주식, 원자재, 변동성, 신용, 거시 지표까지 **총 15개 시그널**을 블룸버그 터미널에서 직접 수집하여 가중 합산합니다.
과거 데이터는 이동평균(MA) 계산과 백분위 임계값 보정에만 활용되며, **핵심 출력은 오늘의 현황**입니다.

```
[Bloomberg Terminal]
        │
        ▼
   fetch_data()          ← 오늘까지 2년치 데이터 수집 (MA 워밍업 포함)
        │
        ▼
 calculate_signals()     ← 이동평균·임계값 기반 가중 시그널 생성
        │
        ▼
   get_regime()          ← 복합 스코어 합산 → 백분위 기준 레짐 분류
        │
        ├──▶ daily_snapshot()   ← 오늘의 현황 (레짐·스코어·시그널·vs 과거 비교)
        │
        ├──▶ plot_results()     ← 3패널 차트 (SPX / 스코어 / 시그널 기여도)
        │
        └──▶ to_excel()         ← 보고용 Excel (3시트: 오늘 현황 / 이력 / 모델 정의)
```

---

## 2. 주요 기능

| 기능 | 설명 |
|---|---|
| **Bloomberg 데이터 수집** | `xbbg.blp.bdh()`로 일별·월별 데이터 자동 수집, 오늘 날짜 자동 적용 |
| **이동평균 기반 시그널** | 자산별 10일/60일 MA 크로스오버로 Risk On/Off 판단 |
| **SKEW 패널티** | SKEW Index > 140 시 스코어에서 2점 차감 (꼬리 리스크 반영) |
| **매크로 필터** | PMI, CPI, 실업률 월별 지표를 일별로 ffill 후 스코어에 반영 |
| **백분위 레짐 분류** | 표본 전체의 p70 / p30 기준으로 레짐 자동 결정 |
| **오늘의 리스크 현황** | 레짐·스코어·시그널별 상태·30d/90d 대비 변화를 한눈에 출력 |
| **3패널 시각화** | SPX 오버레이, 복합 스코어, 시그널 기여도 스택 차트 |
| **보고용 Excel (3시트)** | 오늘 현황 / 이력 비교 / 모델 정의를 색상 코딩된 엑셀로 저장 |
| **데모 모드** | Bloomberg 터미널 없이도 합성 데이터로 전체 파이프라인 검증 가능 |

---

## 3. 환경 설정 및 설치

### 필수 조건
- Bloomberg Terminal 설치 및 로그인 상태
- Python 3.10 이상

### 패키지 설치

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
xbbg>=0.3.7      # Bloomberg BDH 래퍼 (터미널 필수)
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
openpyxl>=3.1    # Excel 보고서 생성
```

> Bloomberg Terminal 없이 테스트하려면 그냥 실행하면 됩니다.
> `xbbg` import 실패 시 자동으로 **데모 모드(합성 데이터)**로 전환됩니다.

---

## 4. 파일 구조

```
canaria_risk_score/
├── risk_scoring_model.py   # 메인 모델 클래스 (RiskScoringModel)
├── requirements.txt        # 패키지 의존성
└── README.md               # 이 문서
```

실행 후 생성되는 파일:
```
canaria_risk_score/
├── risk_score_chart.png      # 3패널 시각화 차트 (자동 저장)
└── risk_score_report.xlsx    # 보고용 Excel (3시트, 자동 저장)
```

---

## 5. 모델 설계 상세

### 5.1 캐너리 자산 유니버스

시그널 로직: 가격이 **이동평균(MA) 대비** 특정 방향에 있을 때 **Risk On = +가중치**, 아닐 때 = 0.

| 분류 | 자산명 | 블룸버그 티커 | 필드 | MA 기간 | 가중치 | Risk On 조건 | 비고 |
|---|---|---|---|---|---|---|---|
| **FX** | 달러 인덱스 | `DXY Curncy` | `PX_LAST` | 10일 | **2.0** | 가격 < MA | 고감도 |
| **EM** | EM 채권 | `EMB US Equity` | `PX_LAST` | 10일 | 1.0 | 가격 > MA | |
| **EM** | EM 통화 | `CEW US Equity` | `PX_LAST` | 10일 | 1.0 | 가격 > MA | |
| **금리** | 미국채 종합 | `BND US Equity` | `PX_LAST` | 10일 | 1.0 | 가격 < MA | 금리 상승(채권 하락) → 위험선호 맥락 |
| **금리** | TIPS | `TIP US Equity` | `PX_LAST` | 10일 | 1.0 | 가격 < MA | |
| **주식** | 선진국 주식 | `VEA US Equity` | `PX_LAST` | 10일 | 1.0 | 가격 > MA | |
| **원자재** | 금 | `GLD US Equity` | `PX_LAST` | 10일 | 1.0 | 가격 < MA | 안전자산 자금 유출 |
| **원자재** | 원자재 | `DBC US Equity` | `PX_LAST` | 10일 | 1.0 | 가격 > MA | |
| **변동성** | VIX | `VIX Index` | `PX_LAST` | 60일 | **2.0** | 지수 < MA | 고감도 |
| **변동성** | VVIX | `VVIX Index` | `PX_LAST` | 60일 | 1.0 | 지수 < MA | 변동성의 변동성 |
| **꼬리리스크** | SKEW | `SKEW Index` | `PX_LAST` | — | **−2.0** | 값 > 140 | 패널티 (↓ 참조) |
| **신용** | 미국 HY OAS | `LF98TRUU Index` | `OAS_SPREAD_BID`¹ | 60일 | **2.0** | 스프레드 < MA | 스프레드 축소 = Risk On |

> ¹ OAS 필드명은 라이선스에 따라 다릅니다. 자동 폴백 순서: `OAS_SPREAD_BID → OAS_BID → OAS_MID → OAS_SPREAD → OAS → Z_SPREAD_MID → OPTION_ADJ_SPREAD`

**이론적 최대 스코어**: 2+1+1+1+1+1+1+1+2+1+2+1+1+1 = **17점**
**이론적 최솟값**: 0 − 2 (SKEW 패널티) = **−2점**

---

### 5.2 매크로 필터

월별/분기별로 발표되는 지표를 `ffill()`로 일별 값에 매핑합니다.

| 지표 | 티커 | Risk On 조건 | 가중치 |
|---|---|---|---|
| ISM 제조업 PMI | `NAPMPMI Index` | 값 > 50 | +1.0 |
| CPI YoY | `CPI YOY Index` | 값 < 3.0% | +1.0 |
| 실업률 | `USURTOT Index` | 값 < 4.0% | +1.0 |

> Bloomberg BDH는 월별 지표를 발표일에만 값을 반환합니다(sparse).
> 코드 내부에서 일별 인덱스로 재색인 후 `ffill()`로 다음 발표일까지 최근 값을 유지합니다.
> `bfill()`은 Look-ahead bias 방지를 위해 **의도적으로 사용하지 않습니다.**

---

### 5.3 SKEW 패널티 로직

SKEW Index는 S&P 500 옵션 시장의 꼬리 리스크 수요를 나타냅니다.

```
SKEW > 140  →  스코어에서 2점 차감  (SKEW_PENALTY = −2.0)
SKEW ≤ 140  →  기여값 = 0
```

- 단순 MA 크로스오버가 아닌 **절대 임계값(140)** 기반
- 다른 지표가 모두 Risk On을 가리켜도 꼬리 리스크가 높으면 스코어를 강제로 하향

---

### 5.4 레짐 분류 기준

고정 임계값 대신 **표본 전체의 백분위**를 사용합니다.
날짜 범위가 달라져도 자동으로 적응합니다.

```
복합 스코어 ≥ p70  →  Risk On      (시장 친화적 환경)
p30 < 스코어 < p70  →  Transition  (중립 / 관망)
복합 스코어 ≤ p30  →  Risk Off     (방어적 환경)
```

---

## 6. 클래스 구조 및 메서드

```python
class RiskScoringModel:
```

### 클래스 속성 (설정값)

| 속성 | 설명 | 기본값 |
|---|---|---|
| `PRICE_ASSETS` | 가격 기반 자산 설정 딕셔너리 | 10개 자산 |
| `SKEW_TICKER / SKEW_THRESHOLD / SKEW_PENALTY` | SKEW 패널티 파라미터 | `140` / `−2.0` |
| `SPREAD_ASSETS` | OAS 스프레드 자산 설정 | LF98TRUU |
| `_OAS_FALLBACK_FIELDS` | OAS 필드명 자동 폴백 목록 | 7개 필드명 |
| `MACRO_ASSETS` | 거시 지표 설정 | PMI / CPI / 실업률 |
| `SPX_TICKER` | 비교용 벤치마크 티커 | `SPX Index` |
| `RISK_ON_PCT / RISK_OFF_PCT` | 레짐 분류 백분위 기준 | `70` / `30` |
| `MA_WARMUP_DAYS` | MA 초기화용 추가 히스토리 일수 | `120`일 |
| `PALETTE` | 차트 색상 딕셔너리 | — |

### 인스턴스 메서드

#### `fetch_data(start_date, end_date)`
Bloomberg Terminal에서 모든 데이터를 수집합니다.

- `start_date` 기준 120 캘린더일 이전부터 수집하여 MA 워밍업 수행
- 내부적으로 4개의 별도 BDH 호출:
  1. 가격 자산 (PX_LAST, 일괄 조회)
  2. 신용 스프레드 (OAS, 7개 필드명 자동 폴백)
  3. 매크로 지표 (PX_LAST, 희소 일별 → ffill)
  4. SPX 벤치마크

#### `calculate_signals()`
이동평균 및 임계값 기반 가중 시그널을 계산합니다.

- MA 계산은 워밍업 포함 전체 히스토리에서 수행 후 `[start_date, end_date]`로 트리밍
- Look-ahead bias 방지를 위해 `bfill()` 미사용, `ffill()`만 적용
- 반환: `pd.DataFrame` (열 = 시그널명, 행 = 날짜)

#### `get_regime()`
복합 스코어를 합산하고 레짐을 분류합니다.

- `self.signals.sum(axis=1, min_count=1)` → 일별 복합 스코어
- 백분위 기준 레짐 라벨 부여
- 반환: `pd.DataFrame` — 컬럼 `["score", "regime"]`

#### `daily_snapshot(date=None)` ⭐ 핵심 출력
**오늘(또는 지정 날짜)의 리스크 현황**을 콘솔에 출력하고 dict로 반환합니다.

| 출력 항목 | 내용 |
|---|---|
| 레짐 & 스코어 | 오늘의 레짐 배지 + 스코어 / 최대값 + 막대 시각화 |
| 역사적 백분위 | 전체 구간에서 오늘 스코어가 몇 % 수준인가 |
| 30일 / 90일 대비 | 단기·중기 추세 방향 확인 |
| 시그널 분류 | Risk On ✔ / Risk Off ✘ / Penalty ⚠ 그룹으로 정리 |

#### `plot_results(figsize, save_path)`
3패널 차트를 생성하고 저장합니다.

| 패널 | 내용 |
|---|---|
| 상단 | S&P 500 가격 + 레짐별 배경 색상 |
| 중단 | 복합 스코어 + 20일 MA + p70/p30 임계선 + 구간 음영 |
| 하단 | 시그널별 기여도 스택 면적 차트 (SKEW 패널티 = 아래 방향 빨강) |

#### `run(start_date=None, end_date=None, plot=True)`
전체 파이프라인을 한 번에 실행합니다.

- `end_date` 기본값 = **오늘**
- `start_date` 기본값 = **오늘로부터 2년 전**
- 날짜 지정 없이 `model.run()` 만으로 최신 데이터 기준 분석

#### `to_excel(path="risk_score_report.xlsx")`
보고용 Excel 파일을 3개 시트로 저장합니다.

| 시트 | 내용 |
|---|---|
| **오늘의 리스크 현황** | 레짐 배지 + 스코어 바 + 30d/90d 비교 + 시그널별 상태 |
| **이력 비교** | 레짐 분포 요약 + 전체 일별 스코어 테이블 (자동 필터, 틀 고정) |
| **모델 정의** | 자산 스펙 + 매크로 필터 + SKEW 패널티 + 레짐 분류 기준 |

#### `signal_summary(date=None)`
특정 날짜의 개별 시그널 가중치·값·상태를 표로 출력합니다.

---

## 7. 사용 방법

### 기본 실행 — 오늘 기준 (권장)

```python
from risk_scoring_model import RiskScoringModel

model = RiskScoringModel()
results = model.run()          # end_date=오늘, start_date=2년 전 자동 적용

# 오늘의 리스크 현황 출력
model.daily_snapshot()

# Excel 보고서 저장
model.to_excel("risk_score_report.xlsx")
```

### 콘솔 출력 예시 (`daily_snapshot`)

```
════════════════════════════════════════════════════════════════
  오늘의 리스크 현황   2025-02-19 (Wed)
════════════════════════════════════════════════════════════════

  🔴  레짐 : [ Risk Off ]
     스코어 : 2.0 / 17   [████░░░░░░░░░░░░░░░░]
     역사적 백분위 : 15.3%  (하위권)
     30일 평균 대비 : ▼ 1.50점   (30d avg = 3.5)
     90일 평균 대비 : ▼ 2.10점   (90d avg = 4.1)

  ✔  RISK ON  시그널
  ────────────────────────────────────────────────────────────
     ✔  US Treasury Agg       +1.0
     ✔  TIPS                  +1.0
     ✔  Commodity             +1.0
     ✔  CPI YoY               +1.0

  ✘  RISK OFF 시그널
  ────────────────────────────────────────────────────────────
     ✘  Dollar Index           0.0
     ✘  EM Bond                0.0
     ...

  ⚠  PENALTY
  ────────────────────────────────────────────────────────────
     ⚠  발동 중!  SKEW Penalty    -2.0

════════════════════════════════════════════════════════════════
  복합 스코어 합계 : 2.00   →   [ Risk Off ]
════════════════════════════════════════════════════════════════
```

### 특정 날짜 지정 실행

```python
model = RiskScoringModel()
results = model.run(start_date="2020-01-01", end_date="2024-12-31")

# 특정 날짜의 현황
model.daily_snapshot("2022-06-15")
```

### 단계별 실행

```python
model = RiskScoringModel()

# 1. 데이터 수집 (오늘까지 2년치)
model.fetch_data("2023-01-01", "2025-02-19")

# 2. 시그널 계산
signals = model.calculate_signals()

# 3. 레짐 분류
results = model.get_regime()     # pd.DataFrame ["score", "regime"]

# 4. 오늘의 현황 출력
snapshot = model.daily_snapshot()
print(snapshot["regime"])        # "Risk Off"
print(snapshot["pct_rank"])      # 15.3 (역사적 백분위)

# 5. 차트 + Excel 저장
model.plot_results()
model.to_excel()
```

### Risk Off 구간 분석

```python
risk_off = results[results["regime"] == "Risk Off"]
print(f"Risk Off 일수: {len(risk_off)}일")
print(risk_off.sort_values("score").head(10))   # 가장 극단적인 날
```

### 파라미터 커스터마이징

```python
model = RiskScoringModel()
model.RISK_ON_PCT    = 75      # 더 엄격한 Risk On 기준
model.RISK_OFF_PCT   = 25
model.SKEW_THRESHOLD = 135     # SKEW 패널티 민감도 상향
model.SKEW_PENALTY   = -3.0    # 패널티 강도 증가

results = model.run()
```

---

## 8. 출력 결과물

### 콘솔 출력 순서

```
① Bloomberg 데이터 수집 로그  (fetch 단계)
② 레짐 요약 통계              (get_regime 단계)
③ 오늘의 리스크 현황          (daily_snapshot)
④ 최근 10 영업일 이력         (results.tail)
⑤ Excel 저장 확인             (to_excel)
```

### `risk_score_chart.png` — 3패널 차트

```
┌─────────────────────────────────────────────────────────┐
│ Panel 1: S&P 500                                        │
│   - SPX 가격선 + 레짐별 배경 색상                          │
│     (녹색=Risk On / 주황=Transition / 빨강=Risk Off)       │
├─────────────────────────────────────────────────────────┤
│ Panel 2: Composite Risk Score                           │
│   - 일별 복합 스코어 (진한 선)                             │
│   - 20일 이동평균 (보라 점선)                              │
│   - p70 / p30 임계선 (녹색/빨강 점선)                     │
│   - 임계값 초과 구간 음영 강조                              │
├─────────────────────────────────────────────────────────┤
│ Panel 3: Signal Contributions                           │
│   - 자산별 가중 기여도 스택 면적 차트                        │
│   - SKEW 패널티는 빨간색으로 아래 방향 표시                  │
└─────────────────────────────────────────────────────────┘
```

### `risk_score_report.xlsx` — 3시트 Excel 보고서

| 시트 | 탭 색상 | 주요 내용 |
|---|---|---|
| **오늘의 리스크 현황** | 오늘 레짐 색 | 레짐 배지(대형) · 스코어 바 · 백분위 · 30d/90d 비교 · 시그널별 상태 |
| **이력 비교** | 파랑 | 레짐 분포 요약 · 전체 일별 테이블 (자동필터 ▼, 틀 고정, 오늘 행 굵게) |
| **모델 정의** | 네이비 | 자산 유니버스 스펙 · 매크로 필터 · SKEW 패널티 규칙 · 레짐 분류 기준 |

---

## 9. 설계 의도 및 주요 고려사항

### 오늘 데이터 우선 설계
`run()`의 기본값이 **오늘**입니다. 과거 2년치 데이터는 MA 계산과 백분위 보정에만 활용합니다.
매일 `python risk_scoring_model.py` 한 번으로 최신 시장 레짐을 즉시 확인할 수 있습니다.

### MA 워밍업 처리
60일 MA가 제대로 초기화되려면 `start_date` 이전 데이터가 필요합니다.
`fetch_data()`는 내부적으로 120 캘린더일 이전부터 데이터를 수집하고,
`calculate_signals()`에서 MA를 계산한 뒤 요청 구간만 잘라냅니다.
`start_date` 첫날부터 신뢰할 수 있는 MA값이 보장됩니다.

### Look-ahead Bias 방지 (`bfill()` 미사용)
시그널 결합 후 `bfill()`(역방향 채움)을 사용하지 않습니다.
`ffill()`만 사용하여 미래 데이터가 과거 행에 유입되는 오염을 차단합니다.
120일 워밍업 버퍼가 충분하기 때문에 초기 NaN 문제가 발생하지 않습니다.

### OAS 필드명 자동 폴백 (7단계)
`LF98TRUU Index`의 OAS 필드명은 블룸버그 터미널 라이선스에 따라 다릅니다.
```
OAS_SPREAD_BID → OAS_BID → OAS_MID → OAS_SPREAD → OAS → Z_SPREAD_MID → OPTION_ADJ_SPREAD
```
각 필드가 실패할 때마다 콘솔에 실패 이유를 출력한 뒤 다음 필드를 시도합니다.

### 월별 매크로 데이터의 일별 변환
Bloomberg는 PMI, CPI, 실업률 등 월별 지표를 **발표일에만** 값을 반환합니다.
코드에서는 일별 인덱스로 재색인한 후 `ffill()`로 다음 발표일까지 값을 유지합니다.

### 백분위 기반 레짐 임계값
고정 임계값(예: 스코어 > 10 = Risk On) 대신 표본 백분위를 사용합니다.
분석 기간이 달라져도 레짐 분류 비율이 안정적으로 유지됩니다.

### Bloomberg 터미널 없이도 테스트 가능
`xbbg` import 실패 시 자동으로 **데모 모드**로 전환됩니다.
VIX, SKEW, HY OAS 등에 현실적인 범위의 합성 데이터를 생성해 전체 파이프라인을 검증할 수 있습니다.

---

## 10. 알려진 데이터 이슈

| 티커 / 필드 | 현상 | 대응 |
|---|---|---|
| `LF98TRUU Index` OAS | 모든 OAS 필드명 실패 시 신용 시그널(가중치 2.0) 제외됨 | 7개 필드 순차 시도; 지속 실패 시 콘솔 경고 출력 및 스킵 |
| `CPI YOY Index` | 일부 터미널에서 `CPI YOY Index` 대신 별도 코드 사용 | 실패 시 `[WARN]` 출력 후 매크로 시그널에서 제외 |
| `xbbg` MultiIndex 컬럼 | BDH 반환값의 컬럼이 `(ticker, field)` 형태인 경우 | `_flatten_bbg_columns()`에서 ticker 레벨로 자동 축소 |
| `datetime.date` 인덱스 | xbbg가 `pd.Timestamp` 대신 `datetime.date` 반환 시 `.loc[]` 슬라이싱 오류 발생 | 동일 메서드에서 `pd.to_datetime()` 변환으로 처리 |

---

## 11. 한계 및 향후 개선 방향

| 구분 | 내용 |
|---|---|
| **생존 편향** | 분석 기간 전체의 백분위로 레짐을 분류하므로 실시간 운용 시 미래 정보가 반영됨. 실전 적용 시 expanding window 방식으로 변경 필요 |
| **가중치 고정** | 현재 가중치는 정성적으로 설정됨. PCA, 회귀분석, 또는 최적화로 데이터 기반 가중치 산출 가능 |
| **시그널 상관관계** | 일부 자산(예: BND↓와 TIP↓)은 높은 상관관계를 가질 수 있음. 상관 조정 가중치 고려 가능 |
| **레짐 전환 지연** | MA 기반 시그널은 후행적. 시장 급변 시(예: VIX 스파이크) 반응 속도 개선 여지 있음 |
| **매크로 발표 시차** | CPI, PMI 등은 발표일과 실제 측정 기간 사이에 시차 존재. Look-ahead bias 주의 필요 |
| **LF98TRUU OAS 의존성** | 블룸버그 라이선스에 따라 OAS 필드가 없을 수 있음. HYG ETF 가격 기반 대체 시그널 도입 검토 가능 |
| **향후 기능 추가** | 레짐별 자산배분 백테스트, 웹 대시보드(Streamlit), 일별 스코어 이메일 알림 등 확장 가능 |

---

*Bloomberg, xbbg, pandas, numpy, matplotlib, openpyxl 기반으로 작성됨*
*IGIS Asset Management — Quant Research*
