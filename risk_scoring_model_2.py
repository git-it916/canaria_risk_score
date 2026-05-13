"""
Global Multi-Asset Risk Scoring Model — v2 (Yahoo Finance + FRED edition)
==========================================================================
risk_scoring_model.py 와 동일한 시그널 로직을 사용하되, 데이터 소스를
Bloomberg Terminal(xbbg) 대신 무료 API 로 교체:

    • Yahoo Finance (yfinance)  — 가격/지수/변동성 (12개 티커)
    • FRED          (fredapi)   — HY OAS 및 미국 매크로 지표 (4개 시리즈)

운영 환경
--------
매일 한국시간 08:30 자동 실행을 가정.
이 시점의 데이터 가용성:

    KST 08:30  ≡  EDT(US Eastern) 전일 19:30
    └─ 미국 정규장 마감(16:00 EDT) 후 약 3시간 30분 경과
       → yfinance 에서 "전일(미국 날짜)" 종가 사용 가능

FRED 발표 지연 (look-ahead bias 방지)
-----------------------------------
FRED 의 월간 시계열은 *참조 월 1일* 로 인덱싱되지만, 실제 발표는
다음 달 초중순임. Bloomberg BDH 는 *발표일* 기준으로 값을 돌려주므로,
원본 로직과 동일한 동작을 유지하려면 FRED 의 reference-period index 를
release-date 로 직접 시프트해야 함.

각 매크로 시리즈의 발표 지연 (calendar days, month-end 기준):

    NAPM       : ~1일   (다음 달 1영업일)
    CPIAUCSL   : ~15일  (다음 달 10–15일)
    UNRATE     : ~7일   (다음 달 첫 금요일)
    BAMLH0A0HYM2: 일별 (FRED 가 T+1 로 dating; 추가 시프트 불필요)

CPI YoY 계산
-----------
FRED CPIAUCSL 은 지수값(index level)이므로 YoY% 를 12개월 전 대비 변화율로
직접 계산. 계산 후에 release-date 시프트 적용.

사용법
-----
    from risk_scoring_model_2 import RiskScoringModelV2

    model = RiskScoringModelV2()
    results = model.run()
    model.daily_snapshot()
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Windows 콘솔(cp949) 에서 한글/유니코드 출력 안전화
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# 시그널 로직 / 차트 / Excel 출력 등은 전부 부모 클래스에서 그대로 상속받음
from risk_scoring_model import RiskScoringModel, _make_synthetic_prices

warnings.filterwarnings("ignore")


# ── External data libraries ────────────────────────────────────────────────────
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    print("[WARN] yfinance not installed - run: pip install yfinance")

try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False
    print("[WARN] fredapi not installed - run: pip install fredapi")


# ── FRED API key ──────────────────────────────────────────────────────────────
# 환경변수 FRED_API_KEY 로 override 가능. 깃 커밋 시 키를 빼고 싶으면
# 이 상수를 비우고 환경변수만 사용하세요.
FRED_API_KEY: str = os.environ.get(
    "FRED_API_KEY", "7469ba1eab0c6971b8a3802634163bc9"
)


# ══════════════════════════════════════════════════════════════════════════════

class RiskScoringModelV2(RiskScoringModel):
    """
    RiskScoringModel 과 동일한 로직, 데이터 소스만 yfinance + FRED 로 교체.

    상속되는 것 (변경 없음)
    ----------------------
      • PRICE_ASSETS / SPREAD_ASSETS / MACRO_ASSETS 정의
      • SKEW 페널티 임계값, 가중치, MA 기간 등 모든 파라미터
      • calculate_signals()  ← MA / 임계값 / 페널티 로직
      • get_regime()         ← p70 / p30 백분위 분류
      • plot_results(), to_excel(), daily_snapshot()
      • signal_summary(), run() 등 헬퍼

    오버라이드 되는 것
    -----------------
      • fetch_data()          : xbbg 가용성 체크 제거
      • _fetch_prices()       : yfinance.download()
      • _fetch_spreads()      : Fred.get_series()
      • _fetch_macro()        : Fred.get_series() + release-date 시프트
      • _fetch_spx()          : yfinance.download()
    """

    # ── Bloomberg 티커 → Yahoo Finance 티커 매핑 ──────────────────────────────
    BBG_TO_YF: dict[str, str] = {
        "DXY Curncy":    "DX-Y.NYB",    # ICE U.S. Dollar Index
        "EMB US Equity": "EMB",         # iShares JPM USD EM Bond ETF
        "CEW US Equity": "CEW",         # WisdomTree EM Currency Fund
        "BND US Equity": "BND",         # Vanguard Total Bond Market ETF
        "TIP US Equity": "TIP",         # iShares TIPS Bond ETF
        "VEA US Equity": "VEA",         # Vanguard FTSE Developed Markets
        "GLD US Equity": "GLD",         # SPDR Gold Shares
        "DBC US Equity": "DBC",         # Invesco DB Commodity Index
        "VIX Index":     "^VIX",        # CBOE VIX
        "VVIX Index":    "^VVIX",       # CBOE VVIX (vol-of-vol)
        "SKEW Index":    "^SKEW",       # CBOE SKEW
        "SPX Index":     "^GSPC",       # S&P 500
    }

    # ── Bloomberg 티커 → FRED 시리즈 ID 매핑 ──────────────────────────────────
    # NAPM 은 FRED 에서 2016년 라이선스 회수로 단종된 상태. 시리즈 자체가
    # 존재하지 않아 fetch_macro 가 자동으로 스킵 (시그널 제외) 함.
    # CPI 는 NSA (CPIAUCNS) 사용 — Bloomberg `CPI YOY Index` 가 NSA 기준 YoY 이므로 매칭.
    BBG_TO_FRED: dict[str, str] = {
        "LF98TRUU Index": "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS (일별, % 단위)
        "NAPMPMI Index":  "NAPM",          # ISM PMI - FRED 2016년 중단. fetch 실패 시 자동 스킵
        "CPI YOY Index":  "CPIAUCNS",      # US CPI Urban NSA (index level → YoY% 직접 계산)
        "USURTOT Index":  "UNRATE",        # US Civilian Unemployment Rate (SA, %)
    }

    # ── FRED 월간 시리즈 발표 지연 (참조월 말일 + N일) ─────────────────────────
    # 이 값을 사용해 FRED 의 reference-period index 를 release-date 로 시프트.
    # 동일 월의 다른 매크로가 같은 날짜에 들어가지 않도록 적절히 분산.
    FRED_RELEASE_LAG_DAYS: dict[str, int] = {
        "NAPM":         2,   # ISM PMI : 다음 달 1영업일
        "CPIAUCNS":    15,   # CPI NSA : 다음 달 10–15일
        "CPIAUCSL":    15,   # CPI SA  : (fallback, 동일한 지연)
        "UNRATE":       7,   # 실업률  : 다음 달 첫 금요일
        # 일별 시리즈 (BAMLH0A0HYM2) 는 이미 올바르게 dating 되어 있음 → 시프트 안 함
    }

    # ── FRED API 재시도 설정 ──────────────────────────────────────────────────
    # FRED 가 종종 transient 500 (Internal Server Error) 를 반환함.
    # 지수 백오프로 N 회 재시도.
    FRED_MAX_RETRIES: int = 4
    FRED_RETRY_BACKOFF_SEC: float = 1.5

    # ── 매크로 데이터 신선도 임계값 ────────────────────────────────────────────
    # FRED 의 최신 관측치가 이 일수보다 오래된 경우 경고. NAPM 처럼 단종된
    # 시리즈를 사용하면 시그널이 과거 값으로 ffill 되어 노이즈가 됨.
    MACRO_STALENESS_WARN_DAYS: int = 60

    # ═════════════════════════════════════════════════════════════════════════
    # Constructor
    # ═════════════════════════════════════════════════════════════════════════

    def __init__(self, fred_api_key: str | None = None) -> None:
        super().__init__()
        self._fred_key = fred_api_key or FRED_API_KEY
        if _FRED_AVAILABLE and self._fred_key:
            self._fred: Fred | None = Fred(api_key=self._fred_key)
        else:
            self._fred = None

    # ─────────────────────────────────────────────────────────────────────────
    # Helper — FRED retry wrapper
    # ─────────────────────────────────────────────────────────────────────────
    def _fred_get_series(
        self, series_id: str, **kwargs
    ) -> pd.Series | None:
        """
        Call fred.get_series with retry/backoff.

        FRED API 가 가끔 transient 500 (Internal Server Error) 를 던지는데
        몇 초 후 재시도하면 대부분 성공. "series does not exist" 같은 영구
        에러는 즉시 None 반환 (NAPM 같은 단종 시리즈 처리).
        """
        if self._fred is None:
            return None
        last_exc: Exception | None = None
        for attempt in range(self.FRED_MAX_RETRIES):
            try:
                return self._fred.get_series(series_id, **kwargs)
            except Exception as exc:
                msg = str(exc)
                last_exc = exc
                # 영구 에러 (시리즈 없음 / 잘못된 ID) - 재시도 의미 없음
                if "does not exist" in msg or "Bad Request" in msg:
                    return None
                # transient 에러 - 백오프 후 재시도
                if attempt < self.FRED_MAX_RETRIES - 1:
                    sleep_sec = self.FRED_RETRY_BACKOFF_SEC * (2 ** attempt)
                    time.sleep(sleep_sec)
        # 전부 실패
        print(f"       [WARN] FRED '{series_id}' 모든 재시도 실패: {last_exc}")
        return None

    # ═════════════════════════════════════════════════════════════════════════
    # 1. fetch_data  (override — Bloomberg 가용성 체크 제거)
    # ═════════════════════════════════════════════════════════════════════════

    def fetch_data(self, start_date: str, end_date: str) -> None:
        """Pull data from yfinance + FRED. Same warm-up logic as parent."""
        self._start_date = start_date
        self._end_date   = end_date

        ext_start = (
            datetime.strptime(start_date, "%Y-%m-%d")
            - timedelta(days=self.MA_WARMUP_DAYS)
        ).strftime("%Y-%m-%d")

        _sep = "─" * 62
        print(f"\n{_sep}")
        print("  YFINANCE + FRED DATA FETCH")
        print(_sep)
        print(f"  Requested window : {start_date}  →  {end_date}")
        print(f"  Fetch window     : {ext_start}  →  {end_date}"
              f"  (+{self.MA_WARMUP_DAYS}d MA warm-up)")

        online = _YF_AVAILABLE and _FRED_AVAILABLE and self._fred is not None
        if online:
            self._full_price  = self._fetch_prices(ext_start, end_date)
            self._full_spread = self._fetch_spreads(ext_start, end_date)
            self._full_macro  = self._fetch_macro(ext_start, end_date)
            self.spx_data     = self._fetch_spx(start_date, end_date)
        else:
            missing: list[str] = []
            if not _YF_AVAILABLE:  missing.append("yfinance")
            if not _FRED_AVAILABLE: missing.append("fredapi")
            if self._fred is None:  missing.append("FRED API key")
            print(f"\n  [DEMO] Missing: {missing} — synthetic data 사용.")
            self._load_synthetic_data(ext_start, end_date, start_date)

        print(f"{_sep}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Helper — yfinance 결과에서 (ticker, "Close") 추출
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_close(raw: pd.DataFrame, yf_ticker: str) -> pd.Series | None:
        """
        yfinance.download() 의 결과는 ticker 개수/group_by 옵션에 따라
        구조가 달라짐. (단일/다중, MultiIndex 순서 등) 그것을 흡수.
        """
        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            # group_by='ticker' → (ticker, field), 그 외 → (field, ticker)
            for key in [(yf_ticker, "Close"), ("Close", yf_ticker)]:
                if key in raw.columns:
                    s = raw[key]
                    return s if isinstance(s, pd.Series) else s.iloc[:, 0]
            # MultiIndex 인데 그래도 못 찾으면 Close 컬럼군 첫번째 사용
            if "Close" in raw.columns.get_level_values(0):
                sub = raw["Close"]
                if yf_ticker in sub.columns:
                    return sub[yf_ticker]
            return None

        # 단순 컬럼인 경우
        if "Close" in raw.columns:
            return raw["Close"]
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # 1a. _fetch_prices  (override — Yahoo Finance)
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_prices(self, ext_start: str, end_date: str) -> pd.DataFrame:
        """Fetch daily Close for all price-based assets + SKEW from Yahoo Finance."""
        bbg_tickers = list(self.PRICE_ASSETS.keys()) + [self.SKEW_TICKER]
        yf_tickers = [self.BBG_TO_YF[t] for t in bbg_tickers]
        print(f"\n  [1/4] Price assets  ({len(yf_tickers)} tickers via yfinance) …")

        # yfinance 의 end 는 exclusive — end_date 자체를 포함하려면 +1일
        yf_end = (
            datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        try:
            raw = yf.download(
                tickers=yf_tickers,
                start=ext_start,
                end=yf_end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            print(f"       [WARN] yfinance.download failed: {exc}")
            return pd.DataFrame()

        out = pd.DataFrame()
        for bbg_t in bbg_tickers:
            yf_t = self.BBG_TO_YF[bbg_t]
            s = self._extract_close(raw, yf_t)
            if s is None or s.dropna().empty:
                print(f"       [WARN] No data for {bbg_t} ({yf_t})")
                continue
            out[bbg_t] = s

        out.index = pd.to_datetime(out.index)
        out = out.sort_index().ffill()

        missing = [t for t in bbg_tickers if t not in out.columns]
        if missing:
            print(f"       [WARN] No data returned for: {missing}")
        print(f"       → {out.shape[0]} rows × {out.shape[1]} cols")
        if not out.empty:
            print(f"       last date = {out.index[-1].date()}")
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # 1b. _fetch_spreads  (override — FRED, 일별 시리즈, 시프트 없음)
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_spreads(self, ext_start: str, end_date: str) -> pd.DataFrame:
        """
        Fetch HY OAS from FRED.

        FRED 의 BAMLH0A0HYM2 는 일별 데이터이고 발표는 T+1 영업일 정도지만
        FRED 가 이미 거래일 기준 dating 을 해주므로 별도 시프트 불필요.
        """
        print(f"\n  [2/4] Spread assets ({len(self.SPREAD_ASSETS)} tickers via FRED) …")
        frames: list[pd.DataFrame] = []

        for bbg_t in self.SPREAD_ASSETS.keys():
            fred_id = self.BBG_TO_FRED.get(bbg_t)
            if fred_id is None:
                print(f"       [WARN] No FRED mapping for {bbg_t}")
                continue
            s = self._fred_get_series(
                fred_id,
                observation_start=ext_start,
                observation_end=end_date,
            )
            if s is None or s.empty:
                print(f"       [WARN] {bbg_t} ({fred_id}) returned no data")
                continue

            s.index = pd.to_datetime(s.index)
            s = s.sort_index().ffill().dropna()
            df = pd.DataFrame({bbg_t: s})
            frames.append(df)
            last = df.index[-1].date()
            print(f"       {bbg_t} → FRED '{fred_id}' "
                  f"({df.shape[0]} rows, last={last})")

        if frames:
            return pd.concat(frames, axis=1)
        return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────────────
    # 1c. _fetch_macro  (override — FRED, release-date 시프트)
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_macro(self, ext_start: str, end_date: str) -> pd.DataFrame:
        """
        Fetch monthly macro indicators from FRED, shifted to release dates.

        Look-ahead bias 방지
        ------------------
        FRED 의 월간 데이터는 *참조월 1일* 로 dating 되지만 실제 발표는
        수십일 뒤에 일어난다. Bloomberg 의 BDH 는 발표일 기준으로 값을
        주므로(원본 모델은 이 가정 위에서 ffill), FRED 인덱스도 동일하게
        release-date 로 시프트해야 한다.

            shifted_date = month_end(ref_month) + FRED_RELEASE_LAG_DAYS

        CPI 처리
        --------
        FRED CPIAUCSL 은 지수값(예: 315.6)이므로 YoY% 를 직접 계산.

            YoY% = (level_t / level_{t-12} - 1) × 100

        12개월 lookback 이 필요하므로 fetch window 를 추가로 확장한다.
        """
        tickers = list(self.MACRO_ASSETS.keys())
        print(f"\n  [3/4] Macro assets  ({len(tickers)} tickers via FRED) …")

        # CPI YoY 계산용 추가 lookback 13개월 ≈ 400일
        ext_macro_start = (
            datetime.strptime(ext_start, "%Y-%m-%d") - timedelta(days=400)
        ).strftime("%Y-%m-%d")

        ext_start_ts = pd.Timestamp(ext_start)
        today_ts = pd.Timestamp.utcnow().tz_localize(None).normalize()
        out: dict[str, pd.Series] = {}

        for bbg_t in tickers:
            fred_id = self.BBG_TO_FRED.get(bbg_t)
            if fred_id is None:
                print(f"       [WARN] No FRED mapping for {bbg_t}")
                continue
            s = self._fred_get_series(
                fred_id,
                observation_start=ext_macro_start,
                observation_end=end_date,
            )
            if s is None or s.empty:
                print(f"       [WARN] {bbg_t} ({fred_id}) returned no data "
                      f"(시리즈 단종 또는 fetch 실패)")
                continue

            s.index = pd.to_datetime(s.index)
            s = s.sort_index().dropna()

            # CPI: 지수값 → YoY% 변환 (NSA / SA 둘 다 대응)
            if fred_id in ("CPIAUCNS", "CPIAUCSL"):
                s = (s / s.shift(12) - 1.0) * 100.0
                s = s.dropna()

            if s.empty:
                print(f"       [WARN] {bbg_t} ({fred_id}) empty after YoY calc")
                continue

            # ── 발표일 시프트 (look-ahead bias 방지) ─────────────────────────
            lag = self.FRED_RELEASE_LAG_DAYS.get(fred_id, 30)
            release_idx = (
                s.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)
            )
            # 미래(아직 발표 안 됐을) 데이터는 제거 — 오늘 이후 release 는 제외
            mask_released = release_idx <= today_ts
            s_shifted = pd.Series(
                s.values[mask_released],
                index=release_idx[mask_released],
                name=bbg_t,
            )

            # 워밍업 시작일 이전 데이터는 잘라냄
            s_shifted = s_shifted[s_shifted.index >= ext_start_ts]
            if s_shifted.empty:
                print(f"       [WARN] {bbg_t} ({fred_id}) empty after release-shift")
                continue

            # 신선도 체크 (단종된 시리즈 감지)
            staleness = (today_ts - s_shifted.index[-1]).days
            if staleness > self.MACRO_STALENESS_WARN_DAYS:
                print(
                    f"       [WARN] {bbg_t} ({fred_id}) 마지막 release="
                    f"{s_shifted.index[-1].date()} → {staleness}일 stale. "
                    f"FRED 시리즈가 단종되었을 수 있음."
                )

            out[bbg_t] = s_shifted
            print(f"       {bbg_t} → {fred_id} "
                  f"({len(s_shifted)} releases, +{lag}d release-lag, "
                  f"last={s_shifted.index[-1].date()})")

        if not out:
            return pd.DataFrame()

        df = pd.concat(out, axis=1).sort_index()
        df = df.dropna(how="all")
        print(f"       → {df.shape[0]} observations (sparse, release-dated)")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 1d. _fetch_spx  (override — Yahoo Finance)
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_spx(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch S&P 500 (^GSPC) for the overlay chart."""
        yf_t = self.BBG_TO_YF[self.SPX_TICKER]
        print(f"\n  [4/4] Benchmark     ({self.SPX_TICKER} → {yf_t}) …")
        yf_end = (
            datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")
        try:
            raw = yf.download(
                tickers=yf_t,
                start=start_date,
                end=yf_end,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:
            print(f"       [WARN] SPX fetch failed: {exc}")
            return pd.DataFrame()

        s = self._extract_close(raw, yf_t)
        if s is None or s.dropna().empty:
            print(f"       [WARN] No SPX data returned")
            return pd.DataFrame()

        df = pd.DataFrame({self.SPX_TICKER: s})
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().ffill()
        print(f"       → {df.shape[0]} rows (last={df.index[-1].date()})")
        return df


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 매일 KST 08:30 실행 가정. end_date 는 모델의 run() 기본값(=오늘) 사용.
    today_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join("canaria_risk_score_output", today_str)
    os.makedirs(out_dir, exist_ok=True)

    model = RiskScoringModelV2()

    # 기간 미지정시 run() 이 start=2년 전, end=오늘로 자동 세팅
    results = model.run(plot=False)

    # ── 핵심 출력: 오늘의 리스크 현황 ──────────────────────────────────
    model.daily_snapshot()

    # ── 보조 출력: 최근 10일 이력 ──────────────────────────────────────
    print("최근 10 영업일:")
    print(results.tail(10).to_string())

    # ── 차트 저장 ────────────────────────────────────────────────────────
    chart_path = os.path.join(out_dir, "risk_score_chart.png")
    model.plot_results(save_path=chart_path)

    # ── Excel 보고서 저장 ───────────────────────────────────────────────
    last_str = model.scores.index[-1].strftime("%Y%m%d")
    excel_path = os.path.join(out_dir, f"risk_score_report_{last_str}.xlsx")
    model.to_excel(excel_path)
