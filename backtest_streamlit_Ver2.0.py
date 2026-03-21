import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import requests
from io import StringIO
import quantstats as qs
import plotly.express as px
import seaborn as sns
from datetime import datetime, timedelta
import concurrent.futures
import time
import random
import os

st.set_page_config(page_title="Advanced AI Quant Lab", layout="wide")

# ─────────────────────────────────────────────
# 환경 감지 — 로컬 vs Streamlit Cloud
# ─────────────────────────────────────────────
IS_CLOUD     = os.environ.get("STREAMLIT_SERVER_HEADLESS", "0") == "1"
MAX_WORKERS  = 2          if IS_CLOUD else 8
SLEEP_JITTER = (2.0, 4.0) if IS_CLOUD else (0.5, 1.5)
RETRY_SLEEP  = (4.0, 7.0) if IS_CLOUD else (2.0, 3.5)

TRANSACTION_COST = 0.0008

# ─────────────────────────────────────────────
# FMP stable 엔드포인트
# ─────────────────────────────────────────────
FMP_BASE = "https://financialmodelingprep.com/stable"

# 실제 API 응답 컬럼명 기준으로 검증된 매핑
FMP_INC_MAP = {
    "revenue":                    "Total Revenue",
    "grossProfit":                "Gross Profit",
    "ebit":                       "EBIT",
    "operatingIncome":            "Operating Income",
    "netIncome":                  "Net Income",
    "interestExpense":            "Interest Expense",
    "costOfRevenue":              "Cost Of Revenue",
    "ebitda":                     "EBITDA",
    "depreciationAndAmortization":"Depreciation And Amortization",
}
FMP_BAL_MAP = {
    "totalAssets":                "Total Assets",
    "totalCurrentAssets":         "Total Current Assets",
    "totalCurrentLiabilities":    "Total Current Liabilities",
    "totalStockholdersEquity":    "Stockholders Equity",
    "totalDebt":                  "Total Debt",
    "cashAndCashEquivalents":     "Cash And Cash Equivalents",
    "inventory":                  "Inventory",
}
FMP_CF_MAP = {
    "freeCashFlow":               "Free Cash Flow",
    "depreciationAndAmortization":"Depreciation And Amortization",
    "operatingCashFlow":          "Operating Cash Flow",
}


# ─────────────────────────────────────────────
# FMP API 호출 헬퍼
# ─────────────────────────────────────────────
def _fmp_get(endpoint: str, api_key: str, params: dict = None) -> list:
    try:
        p = {"apikey": api_key}
        if params:
            p.update(params)
        url  = f"{FMP_BASE}/{endpoint}"
        resp = requests.get(url, params=p, timeout=15)

        if resp.status_code == 429:
            raise RuntimeError("FMP_LIMIT_REACHED")

        if resp.status_code in (402, 403):
            return []

        if not resp.text.strip():
            return []

        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            msg = data.get("Error Message", "") or data.get("error", "")
            if "Limit Reach" in msg or "limit" in msg.lower():
                raise RuntimeError("FMP_LIMIT_REACHED")
            return []
        return data if isinstance(data, list) else []

    except RuntimeError:
        raise
    except Exception:
        return []


def _fmp_to_df(records: list, col_map: dict) -> pd.DataFrame:
    """
    FMP JSON → DataFrame 변환
    인덱스: filingDate (SEC 제출일) 기준 — PIT 정확도 향상
    """
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        # stable API는 filingDate 사용 (실제 응답에서 확인됨)
        date_str = r.get("filingDate") or r.get("date", "")
        if not date_str:
            continue
        row = {"_date": pd.to_datetime(date_str)}
        for fmp_col, std_col in col_map.items():
            val = r.get(fmp_col)
            row[std_col] = float(val) if val is not None else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("_date").sort_index(ascending=False)
    df.index.name = None
    return df


# ─────────────────────────────────────────────
# 단일 티커 재무 fetch (FMP stable)
# ─────────────────────────────────────────────
def _fetch_one_ticker(ticker: str, api_key: str) -> tuple[str, dict | None]:
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        try:
            q_p  = {"symbol": ticker, "period": "quarterly", "limit": 40}
            a_p  = {"symbol": ticker, "period": "annual",    "limit": 20}
            pr_p = {"symbol": ticker}

            q_inc_raw = _fmp_get("income-statement",        api_key, q_p)
            q_bal_raw = _fmp_get("balance-sheet-statement", api_key, q_p)
            q_cf_raw  = _fmp_get("cash-flow-statement",     api_key, q_p)
            a_inc_raw = _fmp_get("income-statement",        api_key, a_p)
            a_bal_raw = _fmp_get("balance-sheet-statement", api_key, a_p)
            a_cf_raw  = _fmp_get("cash-flow-statement",     api_key, a_p)
            profile   = _fmp_get("profile",                 api_key, pr_p)

            if not q_inc_raw and not a_inc_raw:
                return ticker, None

            info_raw = profile[0] if profile else {}
            price    = float(info_raw.get("price")     or 1)
            mktcap   = float(info_raw.get("marketCap") or 0)
            shares   = float(info_raw.get("sharesOutstanding") or
                             (mktcap / price if price > 0 else 0))

            return ticker, {
                "q_fin": _fmp_to_df(q_inc_raw, FMP_INC_MAP),
                "q_bal": _fmp_to_df(q_bal_raw, FMP_BAL_MAP),
                "q_cf":  _fmp_to_df(q_cf_raw,  FMP_CF_MAP),
                "a_fin": _fmp_to_df(a_inc_raw, FMP_INC_MAP),
                "a_bal": _fmp_to_df(a_bal_raw, FMP_BAL_MAP),
                "a_cf":  _fmp_to_df(a_cf_raw,  FMP_CF_MAP),
                "info": {
                    "sharesOutstanding": shares,
                    "marketCap":         mktcap,
                    "sector":            info_raw.get("sector",      ""),
                    "industry":          info_raw.get("industry",    ""),
                    "longName":          info_raw.get("companyName", ticker),
                },
            }

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((2 ** (attempt + 1)) + random.uniform(*SLEEP_JITTER))

    return ticker, None

    return ticker, None


# ─────────────────────────────────────────────
# 재무 데이터 수집
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_all_financial_source(tickers: list, api_key: str) -> dict:
    data_cache: dict = {}
    failed:     list = []
    progress_bar = st.progress(0, text="재무 데이터 수집 중 (FMP stable)...")
    total   = len(tickers)
    workers = min(MAX_WORKERS, 3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_fetch_one_ticker, t, api_key): t
            for t in tickers
        }
        done_count = 0
        try:
            for future in concurrent.futures.as_completed(future_map, timeout=600):
                t = future_map[future]
                try:
                    ticker, result = future.result(timeout=60)
                    if result:
                        data_cache[ticker] = result
                    else:
                        failed.append(ticker)
                except Exception as e:
                    if "FMP_LIMIT_REACHED" in str(e):
                        st.error(
                            "⛔ FMP 일일 요청 한도 초과! "
                            "한국 시간 오전 9시(UTC 자정) 이후 다시 시도해주세요."
                        )
                        for f in future_map:
                            f.cancel()
                        break
                    failed.append(t)
                done_count += 1
                progress_bar.progress(
                    int(done_count / total * 80),
                    text=f"재무 데이터 수집 중... ({done_count}/{total})"
                )
        except concurrent.futures.TimeoutError:
            for f, t in future_map.items():
                if not f.done():
                    failed.append(t)
                    f.cancel()

    if failed:
        for idx, ticker in enumerate(failed):
            progress_bar.progress(
                int(80 + idx / max(len(failed), 1) * 18),
                text=f"재시도 중... ({idx+1}/{len(failed)})"
            )
            time.sleep(random.uniform(*RETRY_SLEEP))
            _, result = _fetch_one_ticker(ticker, api_key)
            if result:
                data_cache[ticker] = result

    progress_bar.progress(100, text=f"완료: {len(data_cache)}/{total}개 수집")
    time.sleep(0.5)
    progress_bar.empty()
    return data_cache


# ─────────────────────────────────────────────
# S&P 500 종목 리스트
# ─────────────────────────────────────────────
@st.cache_data
def get_sp500_info():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    df = pd.read_html(StringIO(response.text))[0]
    sectors = sorted(df["GICS Sector"].unique().tolist())
    return df, sectors


# ─────────────────────────────────────────────
# 헬퍼: Series 안전 추출
# ─────────────────────────────────────────────
def _safe_get(series: pd.Series, key: str, default=0):
    val = series.get(key, default)
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


# ─────────────────────────────────────────────
# 헬퍼: Point-in-Time 재무 행 추출
#  ① 반환 시 is_quarterly 플래그도 함께 반환
#     → 연간 데이터 폴백 시 ×4 방지에 사용
# ─────────────────────────────────────────────
def _get_pit_row(
    src: dict, q_key: str, a_key: str, ref_dt: pd.Timestamp
) -> tuple[pd.Series, bool]:
    """
    반환: (row: pd.Series, is_quarterly: bool)
      is_quarterly=True  → 분기 데이터 사용 → 연간화 시 ×4
      is_quarterly=False → 연간 데이터 사용 → 연간화 시 ×1
    """
    q_df = src.get(q_key, pd.DataFrame())
    a_df = src.get(a_key, pd.DataFrame())
    empty = pd.Series(dtype=float)

    def _best_row(df: pd.DataFrame) -> pd.Series | None:
        if df.empty:
            return None
        valid = df[df.index <= (ref_dt + pd.Timedelta(days=45))]
        if valid.empty:
            return None
        idx_min = int(np.abs((valid.index - ref_dt).days).argmin())
        row = valid.iloc[idx_min].copy()
        if row.isna().sum() > len(row) * 0.5:
            filled = df.ffill().bfill()
            if not filled.empty and idx_min < len(filled):
                row = filled.iloc[idx_min]
        return row

    # 분기 우선
    row = _best_row(q_df)
    if row is not None:
        return row, True

    # 연간 폴백
    row = _best_row(a_df)
    if row is not None:
        return row, False

    return empty, False


# ─────────────────────────────────────────────
# ② 성장률 계산 — 분기/연간 타입 통일
#    분기: QoQ(전분기 대비), 연간: YoY(전년 대비)
#    → 같은 타입끼리만 비교
# ─────────────────────────────────────────────
def _get_prev_row(
    src: dict, q_key: str, a_key: str,
    ref_dt: pd.Timestamp, is_quarterly: bool
) -> pd.Series:
    key   = q_key if is_quarterly else a_key
    df    = src.get(key, pd.DataFrame())
    empty = pd.Series(dtype=float)
    if df.empty:
        return empty
    valid = df[df.index <= (ref_dt + pd.Timedelta(days=45))]
    if len(valid) < 2:
        return empty
    return valid.iloc[1]   # 현재 바로 직전 행


# ─────────────────────────────────────────────
# ⑧ 워크포워드용 최적 모델 선택
#    max_depth 후보 3가지를 교차검증으로 비교
# ─────────────────────────────────────────────
def _select_best_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    best_score = -np.inf
    best_depth = 5
    for depth in [5, 10, 15]:
        model = RandomForestRegressor(
            n_estimators=100, random_state=42,
            max_depth=depth, min_samples_leaf=5, n_jobs=-1
        )
        # 데이터가 너무 적으면 cv=2, 충분하면 cv=3
        cv = 2 if len(X) < 30 else 3
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            mean_score = scores.mean()
        except Exception:
            mean_score = -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    best_model = RandomForestRegressor(
        n_estimators=100, random_state=42,
        max_depth=best_depth, min_samples_leaf=5, n_jobs=-1
    )
    best_model.fit(X, y)
    return best_model


# ─────────────────────────────────────────────
# ML 피처 추출 (Point-in-Time)
# ─────────────────────────────────────────────
def fetch_ml_data_optimized_pit(
    tickers, ref_date, full_hist_data, source_cache, is_training=True
):
    features_list = []
    ref_dt        = pd.to_datetime(ref_date)
    all_level0    = set(full_hist_data.columns.get_level_values(0))

    for ticker in tickers:
        try:
            if ticker not in all_level0:
                continue

            ticker_prices = full_hist_data[ticker].dropna(how="all")
            hist = ticker_prices[ticker_prices.index < ref_dt].tail(252)
            if len(hist) < 252:
                continue

            close_now = float(hist["Close"].iloc[-1])
            src       = source_cache.get(ticker, {})
            info      = src.get("info", {})

            # ① is_quarterly 플래그 수령
            cur, is_q_fin = _get_pit_row(src, "q_fin", "a_fin", ref_dt)
            bal, is_q_bal = _get_pit_row(src, "q_bal", "a_bal", ref_dt)
            cf,  is_q_cf  = _get_pit_row(src, "q_cf",  "a_cf",  ref_dt)

            # ① 연간화 계수: 분기=4, 연간=1
            fin_mult = 4 if is_q_fin else 1
            cf_mult  = 4 if is_q_cf  else 1

            # ② 동일 타입 직전 행으로 성장률 계산
            prev = _get_prev_row(src, "q_fin", "a_fin", ref_dt, is_q_fin)

            shares        = float(info.get("sharesOutstanding") or 1)
            mkt_cap       = close_now * shares
            net_income    = _safe_get(cur, "Net Income")
            revenue       = _safe_get(cur, "Total Revenue")
            gross_profit  = _safe_get(cur, "Gross Profit")
            total_assets  = _safe_get(bal, "Total Assets",  1) or 1
            fcf           = _safe_get(cf,  "Free Cash Flow")
            ebit          = _safe_get(cur, "EBIT")
            da            = _safe_get(cf,  "Depreciation And Amortization")
            ebitda        = ebit + da
            # ① 연간화 시 올바른 배수 적용
            annual_ni     = net_income * fin_mult
            annual_rev    = revenue    * fin_mult
            annual_ebitda = ebitda     * fin_mult
            annual_fcf    = fcf        * cf_mult
            total_debt    = _safe_get(bal, "Total Debt")
            cash_eq       = _safe_get(bal, "Cash And Cash Equivalents")
            equity        = _safe_get(bal, "Stockholders Equity", 1) or 1
            ev            = mkt_cap + total_debt - cash_eq

            # ② 동일 타입 직전 데이터로 성장률 계산
            prev_revenue    = _safe_get(prev, "Total Revenue", 1) or 1
            prev_net_income = _safe_get(prev, "Net Income",    1) or 1

            close = hist["Close"]

            data = {
                "Ticker":             ticker,
                # ─ 밸류에이션 (연간화 배수 수정) ─
                "P/E":                mkt_cap / annual_ni        if annual_ni > 0        else 0,
                "P/S":                mkt_cap / annual_rev       if annual_rev > 0       else 0,
                "P/B":                mkt_cap / equity,
                "P/FCF":              mkt_cap / annual_fcf       if annual_fcf > 0       else 0,
                "EV/EBITDA":          ev / annual_ebitda         if annual_ebitda > 0    else 0,
                "FCF_Yield":          annual_fcf / mkt_cap       if mkt_cap > 0          else 0,
                # ─ 수익성 ─
                "ROE":                net_income / equity,
                "ROA":                net_income / total_assets,
                "Gross_Margin":       gross_profit / revenue     if revenue > 0          else 0,
                "Operating_Margin":   ebit / revenue             if revenue > 0          else 0,
                "EBITDA_Margin":      ebitda / revenue           if revenue > 0          else 0,
                "GP_A_Quality":       gross_profit / total_assets,
                "Asset_Turnover":     revenue / total_assets,
                "Inventory_Turnover": (
                    _safe_get(cur, "Cost Of Revenue") / (_safe_get(bal, "Inventory", 1) or 1)
                    if "Inventory" in bal.index else 0
                ),
                # ─ 성장률 (② 동일 타입 직전 비교) ─
                "Revenue_Growth":     (revenue / prev_revenue) - 1,
                "NetIncome_Growth":   (net_income / prev_net_income) - 1,
                # ─ 재무 건전성 ─
                "Debt_Equity":        total_debt / equity,
                "Current_Ratio":      (
                    _safe_get(bal, "Total Current Assets")
                    / (_safe_get(bal, "Total Current Liabilities", 1) or 1)
                    if "Total Current Liabilities" in bal.index else 0
                ),
                "Interest_Coverage":  (
                    ebit / (_safe_get(cur, "Interest Expense", 1) or 1)
                    if "Interest Expense" in cur.index else 0
                ),
                # ─ 모멘텀 ─
                "Mom_1w":             (close_now / close.iloc[-6])   - 1 if len(hist) >= 6   else 0,
                "Mom_1m":             (close_now / close.iloc[-21])  - 1 if len(hist) >= 21  else 0,
                "Mom_6m":             (close_now / close.iloc[-127]) - 1 if len(hist) >= 127 else 0,
                "Mom_12m":            (close_now / close.iloc[-252]) - 1 if len(hist) >= 252 else 0,
                "MA_Convergence":     (
                    (close.rolling(20).mean().iloc[-1] / close.rolling(200).mean().iloc[-1]) - 1
                    if len(hist) >= 200 else 0
                ),
                "MA50_Dist":          close_now / close.rolling(50).mean().iloc[-1]  if len(hist) >= 50  else 1,
                "MA200_Dist":         close_now / close.rolling(200).mean().iloc[-1] if len(hist) >= 200 else 1,
                "Momentum_12M_1M":    (close.iloc[-21] / close.iloc[-252]) - 1 if len(hist) >= 252 else 0,
                "Momentum_6M_1M":     (close.iloc[-21] / close.iloc[-126]) - 1 if len(hist) >= 126 else 0,
                "Momentum_Custom":    (close.iloc[-1]  / close.iloc[-63])  - 1 if len(hist) >= 63  else 0,
                # ─ 변동성/거래량 ─
                "Volatility_30d":     close.pct_change().std() * np.sqrt(252),
                "Risk_Adj_Return":    (
                    close.pct_change().mean() / close.pct_change().std()
                    if close.pct_change().std() != 0 else 0
                ),
                "Vol_Change":         (
                    hist["Volume"].iloc[-1] / hist["Volume"].rolling(21).mean().iloc[-1]
                    if len(hist) >= 21 else 1
                ),
            }

            if is_training:
                future_prices = ticker_prices[ticker_prices.index >= ref_dt].head(22)
                if len(future_prices) >= 20:
                    data["Target_Return"] = (
                        future_prices["Close"].iloc[-1] / future_prices["Close"].iloc[0]
                    ) - 1
                else:
                    continue

            features_list.append(data)

        except Exception:
            continue

    return pd.DataFrame(features_list).replace([np.inf, -np.inf], np.nan).fillna(0)


# ─────────────────────────────────────────────
# 리밸런싱 메타 정보 수집
# ─────────────────────────────────────────────
def get_rebalance_meta(tickers, ref_date, full_hist_data, source_cache):
    ref_dt     = pd.to_datetime(ref_date)
    all_level0 = set(full_hist_data.columns.get_level_values(0))
    period_starts = []
    data_types: dict = {}

    for ticker in tickers:
        if ticker not in all_level0:
            continue
        ticker_prices = full_hist_data[ticker].dropna(how="all")
        hist = ticker_prices[ticker_prices.index < ref_dt].tail(252)
        if len(hist) < 252:
            continue
        period_starts.append(hist.index[0])

        src   = source_cache.get(ticker, {})
        q_fin = src.get("q_fin", pd.DataFrame())
        a_fin = src.get("a_fin", pd.DataFrame())
        q_valid = q_fin[q_fin.index <= (ref_dt + pd.Timedelta(days=45))] if not q_fin.empty else pd.DataFrame()
        a_valid = a_fin[a_fin.index <= (ref_dt + pd.Timedelta(days=45))] if not a_fin.empty else pd.DataFrame()

        if not q_valid.empty:
            data_types[ticker] = "분기"
        elif not a_valid.empty:
            data_types[ticker] = "연간(분기 대체)"
        else:
            data_types[ticker] = "없음"

    period_start = min(period_starts).strftime("%Y-%m-%d") if period_starts else "N/A"
    total   = len(data_types)
    q_count = sum(1 for v in data_types.values() if v == "분기")
    a_count = sum(1 for v in data_types.values() if v == "연간(분기 대체)")
    n_count = sum(1 for v in data_types.values() if v == "없음")

    return {
        "period_start": period_start,
        "period_end":   ref_dt.strftime("%Y-%m-%d"),
        "data_types":   data_types,
        "q_count": q_count, "a_count": a_count,
        "n_count": n_count, "total":   total,
    }


# ─────────────────────────────────────────────
# 동적 유니버스 선정
#  - 매 리밸런싱 시점마다 당시 시가총액 상위 N개 선정
#  - 추가 API 호출 없이 이미 받은 데이터로 계산
#  - 서바이버십 바이어스 완화 (현재 시가총액 고정 방지)
# ─────────────────────────────────────────────
def get_universe_by_mktcap(
    tickers: list,
    ref_date,
    full_hist_data: pd.DataFrame,
    source_cache: dict,
    max_n: int,
) -> tuple[list, pd.DataFrame]:
    """
    ref_date 시점 기준 시가총액 상위 max_n개 티커 반환.

    반환:
        universe   : 시가총액 순 정렬된 티커 리스트 (최대 max_n개)
        mktcap_df  : 전체 시가총액 계산 결과 DataFrame (히스토리 표시용)
    """
    ref_dt     = pd.to_datetime(ref_date)
    all_level0 = set(full_hist_data.columns.get_level_values(0))
    rows = []

    for ticker in tickers:
        try:
            if ticker not in all_level0:
                continue

            # ref_dt 시점의 종가
            prices = full_hist_data[ticker]["Close"].dropna()
            price  = prices.asof(ref_dt)
            if pd.isna(price) or price <= 0:
                continue

            # source_cache에서 발행주식수 추출
            info   = source_cache.get(ticker, {}).get("info", {})
            shares = float(info.get("sharesOutstanding") or 0)

            # shares가 없으면 시가총액 0으로 처리 (순위 뒤로 밀림)
            mktcap = price * shares if shares > 0 else 0

            rows.append({
                "Ticker":    ticker,
                "Price":     round(price, 2),
                "Shares":    shares,
                "MarketCap": mktcap,
            })
        except Exception:
            continue

    if not rows:
        # 시가총액 계산 실패 시 원래 순서 그대로 반환
        return tickers[:max_n], pd.DataFrame()

    mktcap_df = (
        pd.DataFrame(rows)
        .sort_values("MarketCap", ascending=False)
        .reset_index(drop=True)
    )
    mktcap_df["Rank"] = mktcap_df.index + 1
    mktcap_df["MarketCap_B"] = (mktcap_df["MarketCap"] / 1e9).round(1)  # 십억 달러

    universe = mktcap_df.head(max_n)["Ticker"].tolist()
    return universe, mktcap_df


# ─────────────────────────────────────────────
# ⑥ 성과 지표 계산 (확장)
# ─────────────────────────────────────────────
def get_extended_metrics(rets: pd.Series, label: str) -> dict:
    cum  = (1 + rets).prod() - 1
    yrs  = max((rets.index[-1] - rets.index[0]).days / 365.25, 0.1)
    cagr = (1 + cum) ** (1 / yrs) - 1
    mdd  = qs.stats.max_drawdown(rets)

    # 월별 승률
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly > 0).mean() if len(monthly) > 0 else 0

    # 최대 연속 손실 일수
    losing = (rets < 0).astype(int)
    max_streak = 0
    streak = 0
    for v in losing:
        streak = streak + 1 if v else 0
        max_streak = max(max_streak, streak)

    return {
        "누적 수익률":        f"{cum * 100:.2f}%",
        "연수익률(CAGR)":     f"{cagr * 100:.2f}%",
        "샤프 지수":          round(qs.stats.sharpe(rets), 2),
        "칼마 비율":          round(cagr / abs(mdd), 2) if mdd != 0 else "∞",
        "MDD":                f"{mdd * 100:.2f}%",
        "월별 승률":          f"{win_rate * 100:.1f}%",
        "최대 연속 손실(일)": max_streak,
    }


# ─────────────────────────────────────────────
# 지표 중요도 히트맵
# ─────────────────────────────────────────────
def display_importance_heatmap(imp_all_df):
    st.subheader("🌡️ 지표별 영향력 타임라인 (Heatmap)")
    if imp_all_df.empty:
        st.write("데이터가 부족합니다.")
        return
    top_15 = imp_all_df.mean().nlargest(15).index.tolist()
    heatmap_data = imp_all_df[top_15].T
    try:
        heatmap_data.columns = [pd.to_datetime(d).strftime("%Y-%m") for d in heatmap_data.columns]
    except Exception:
        heatmap_data.columns = [str(d)[:7] for d in heatmap_data.columns]
    heatmap_norm = heatmap_data.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9), axis=0
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        heatmap_norm, annot=heatmap_data.values, fmt=".2f",
        cmap="YlGnBu", linewidths=0.5,
        cbar_kws={"label": "Relative Importance"}, ax=ax,
    )
    plt.title("Feature Importance Heatmap (Top 15 Metrics)")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# ─────────────────────────────────────────────
# FMP 무료 플랜 허용 종목 (고정 유니버스)
# 출처: FMP 공식 문서 symbol 제한 목록
# ─────────────────────────────────────────────
FMP_FREE_UNIVERSE = {
    "Tech / 반도체":      ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "CSCO", "ADBE", "TSM", "SONY"],
    "빅테크 / 플랫폼":    ["GOOGL", "META", "AMZN", "NFLX", "BIDU", "BABA", "ATVI"],
    "핀테크 / 금융":      ["JPM", "BAC", "GS", "WFC", "V", "PYPL", "SQ", "C", "COIN", "HOOD", "SOFI"],
    "소비재 / 유통":      ["COST", "WMT", "TGT", "NKE", "SBUX", "KO", "PEP", "DIS", "ETSY"],
    "헬스케어 / 바이오":  ["PFE", "JNJ", "ABBV", "MRNA", "UNH", "HCA", "CPRX"],
    "에너지 / 소재":      ["XOM", "CVX", "ET", "MRO"],
    "항공 / 여행":        ["DAL", "UAL", "AAL", "CCL"],
    "모빌리티 / EV":      ["TSLA", "GM", "F", "UBER", "LCID", "RIVN", "NIO"],
    "통신 / 미디어":      ["T", "VZ", "SIRI", "SNAP", "ROKU", "PINS", "FUBO", "VIAC", "TLRY"],
    "항공우주 / 방산":    ["BA", "LMT", "GE", "FDX"],
    "기타 / ETF":         ["SPY", "VWO", "SPYG", "PLTR", "SHOP", "RBLX", "DOCU", "ZM",
                           "RKT", "BILI", "NOK", "WBA", "MGM", "CARR", "TWTR"],
}

# 전체 허용 종목 플랫 리스트
FMP_ALL_FREE = [t for tickers in FMP_FREE_UNIVERSE.values() for t in tickers]

# ─────────────────────────────────────────────
# 리밸런싱 주기 → 권장 백테스트 기간
# FMP 기준: 분기 10년, 연간 20년
# ─────────────────────────────────────────────
RECOMMENDED_YEARS = {1: 8, 3: 10, 6: 15, 12: 20}
MIN_PERIODS       = {1: 96, 3: 40,  6: 30,  12: 20}

def get_recommended_dates(reb_months: int) -> tuple[datetime, datetime]:
    years    = RECOMMENDED_YEARS.get(reb_months, 8)
    end_dt   = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(years * 365.25))
    return start_dt, end_dt


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🚀 Advanced AI Quant Lab")

# ── 사이드바: API 키 + 종목 수 설정 ───────────
with st.sidebar:
    st.header("🔑 API 설정")
    fmp_api_key = st.text_input(
        "FMP API Key",
        type="password",
        placeholder="FMP API 키 입력",
        help="https://financialmodelingprep.com 에서 무료 발급 (250 req/day)",
    )
    if fmp_api_key:
        st.success("✅ API 키 입력됨")
    else:
        st.warning("⚠️ API 키를 입력해야 실행할 수 있어요")

    st.divider()

    st.header("📊 종목 수 제한")
    FMP_CALLS_PER_TICKER = 7
    FMP_FREE_LIMIT       = 250
    max_free             = FMP_FREE_LIMIT // FMP_CALLS_PER_TICKER  # 35

    max_tickers = st.number_input(
        "최대 분석 종목 수",
        min_value=3, max_value=len(FMP_ALL_FREE),
        value=min(20, max_free),
        step=1,
        help=f"무료 플랜 허용 종목 총 {len(FMP_ALL_FREE)}개 / 하루 최대 {max_free}개 권장",
    )
    est_req = max_tickers * FMP_CALLS_PER_TICKER
    if est_req <= FMP_FREE_LIMIT:
        st.success(f"✅ 예상 요청: {est_req} / {FMP_FREE_LIMIT}")
    else:
        st.warning(f"⚠️ 예상 요청: {est_req} → 한도 초과 가능")

    st.divider()
    st.caption(
        f"FMP 무료 허용 종목: {len(FMP_ALL_FREE)}개\n"
        f"종목당 호출: {FMP_CALLS_PER_TICKER}회\n"
        f"한도: {FMP_FREE_LIMIT} req/day"
    )

with st.expander("🛠 전략 설정", expanded=True):

    # ── 행 1: 업종 필터 / 선정 종목 수 / 비중 방식 ───────────
    r1c1, r1c2, r1c3 = st.columns([3, 1, 2])
    with r1c1:
        all_industries = ["전체"] + list(FMP_FREE_UNIVERSE.keys())
        selected_industries = st.multiselect(
            "1. 업종 필터 (복수 선택 가능)",
            all_industries,
            default=["전체"],
            help="전체 선택 시 FMP 무료 허용 종목 전체를 대상으로 분석해요",
        )
    with r1c2:
        top_n = st.number_input("2. 선정 종목 수", min_value=3, max_value=20, value=5)
    with r1c3:
        weight_mode = st.radio(
            "3. 포트폴리오 비중 방식", ["동일 비중", "AI 점수 비례"], horizontal=True
        )

    st.divider()

    # ── 행 2: 리밸런싱 주기 ────────────────────────────────────
    reb_months = st.select_slider(
        "4. 리밸런싱 주기 (개월)",
        options=[1, 3, 6, 12], value=1,
        help="주기를 바꾸면 아래 백테스트 기간이 자동으로 바뀌어요."
    )
    auto_start, auto_end = get_recommended_dates(reb_months)
    min_periods_needed   = MIN_PERIODS[reb_months]
    years_needed         = RECOMMENDED_YEARS[reb_months]

    st.caption(
        f"📌 {reb_months}개월 리밸런싱 권장: **{years_needed}년** "
        f"(최소 {min_periods_needed}구간)"
    )

    # ── 행 3: 날짜 ────────────────────────────────────────────
    use_auto = st.checkbox("✅ 권장 기간 자동 적용", value=True)
    d1, d2   = st.columns(2)
    with d1:
        if use_auto:
            st.date_input("5. 백테스트 시작일 (자동)", value=auto_start, disabled=True)
            start_date = auto_start.date()
        else:
            start_date = st.date_input("5. 백테스트 시작일 (수동)", value=auto_start)
    with d2:
        if use_auto:
            st.date_input("6. 백테스트 종료일 (자동)", value=auto_end, disabled=True)
            end_date = auto_end.date()
        else:
            end_date = st.date_input("6. 백테스트 종료일 (수동)", value=auto_end)

    actual_periods = max(
        int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            / (reb_months * 30.44)), 0
    )
    if actual_periods < min_periods_needed:
        st.warning(f"⚠️ 예상 구간 수: {actual_periods}개 (권장 {min_periods_needed}개)")
    else:
        st.success(f"✅ 예상 구간 수: {actual_periods}개 ({years_needed}년)")

    st.divider()

    # ── 행 4: 거래비용 ────────────────────────────────────────
    tc_bps = st.number_input(
        "7. 거래비용 (편도, bps)",
        min_value=0, max_value=50,
        value=10 if reb_months == 1 else 8,
        help="유지 종목은 비용 없음, 교체 종목에만 편도 비용 적용"
    )

    run_analysis = st.button(
        "백테스트 실행 🚀",
        use_container_width=True,
        disabled=not fmp_api_key,
    )

TRANSACTION_COST = tc_bps / 10000 if run_analysis else TRANSACTION_COST


# ─────────────────────────────────────────────
# 백테스트 실행
# ─────────────────────────────────────────────
if run_analysis:
    if not fmp_api_key:
        st.error("사이드바에 FMP API 키를 먼저 입력해주세요.")
        st.stop()

    # ── 업종 필터 적용 후 허용 종목 추출 ─────────────────────
    if "전체" in selected_industries or not selected_industries:
        candidate_tickers = FMP_ALL_FREE.copy()
    else:
        candidate_tickers = []
        for ind in selected_industries:
            candidate_tickers.extend(FMP_FREE_UNIVERSE.get(ind, []))
        # 중복 제거 (순서 유지)
        seen = set()
        candidate_tickers = [
            t for t in candidate_tickers
            if not (t in seen or seen.add(t))
        ]

    # 최대 종목 수 제한
    tickers = candidate_tickers[:max_tickers]

    st.info(
        f"📌 분석 대상: **{len(tickers)}개** 종목 "
        f"(FMP 무료 허용 {len(FMP_ALL_FREE)}개 중 "
        f"{'전체' if '전체' in selected_industries else '+'.join(selected_industries)} 필터 적용)"
    )
    with st.expander("분석 종목 목록 보기"):
        st.write(", ".join(tickers))

    # 주가 데이터는 yfinance 유지 (주가는 무료·안정적)
    st.info(f"📡 주가 데이터 다운로드 중... ({len(tickers)}개 티커, yfinance)")
    full_hist_data = yf.download(
        tickers,
        start=pd.to_datetime(start_date) - timedelta(days=400),
        end=pd.to_datetime(end_date) + timedelta(days=40),
        group_by="ticker",
        progress=False,
        threads=True,
    )
    st.success(f"✅ 주가 데이터 완료 ({full_hist_data.shape[1]}개 컬럼)")

    st.info("📊 재무 데이터 수집 중 (FMP)...")
    source_cache = get_all_financial_source(tickers, fmp_api_key)
    fin_count    = len(source_cache)
    st.success(f"✅ 재무 데이터 완료: {fin_count}/{len(tickers)}개")
    if fin_count < len(tickers) * 0.5:
        st.warning(
            f"⚠️ 재무 데이터 수집률이 낮아요 ({fin_count}/{len(tickers)}).\n"
            "FMP 무료 플랜 일일 한도(250 req)를 초과했을 수 있어요. "
            "내일 다시 시도하거나 유료 플랜을 고려해보세요."
        )

    date_range           = pd.date_range(start=start_date, end=end_date, freq=f"{reb_months}MS")
    all_strategy_returns = pd.Series(dtype=float)
    rebalance_details    = []
    importance_history   = []
    final_model_columns  = None
    latest_model         = None
    scaler               = RobustScaler()   # ⑦ 스케일러 초기화

    # ── 초기 모델 학습 ────────────────────────────────────────
    with st.status("🏗️ 초기 모델 학습 중...", expanded=True) as status:

        # 초기 시점 유니버스도 당시 시가총액 기준으로 선정
        init_universe, init_mktcap_df = get_universe_by_mktcap(
            tickers, date_range[0], full_hist_data, source_cache, max_tickers
        )
        if len(init_universe) < top_n:
            init_universe = tickers[:max_tickers]

        if not init_mktcap_df.empty:
            st.write(
                f"초기 유니버스: 시가총액 상위 {len(init_universe)}개 "
                f"(최대 ${init_mktcap_df.head(1)['MarketCap_B'].iloc[0]:.0f}B ~ "
                f"${init_mktcap_df.iloc[len(init_universe)-1]['MarketCap_B']:.0f}B)"
            )

        train_df_init = fetch_ml_data_optimized_pit(
            init_universe, date_range[0], full_hist_data, source_cache, is_training=True
        )
        st.write(f"학습 데이터: {len(train_df_init)}개 종목")

        if not train_df_init.empty:
            fin_cols   = ["P/E", "P/S", "ROE", "ROA", "Gross_Margin"]
            zero_ratio = (train_df_init[fin_cols] == 0).all(axis=1).mean()
            st.write(f"재무 지표 정상 비율: {(1 - zero_ratio) * 100:.0f}%")

        if not train_df_init.empty and "Target_Return" in train_df_init.columns:
            X_init = train_df_init.drop(["Ticker", "Target_Return"], axis=1)
            y_init = train_df_init["Target_Return"]
            final_model_columns = X_init.columns.tolist()

            # ⑦ 초기 스케일러 fit
            X_init_scaled = pd.DataFrame(
                scaler.fit_transform(X_init),
                columns=final_model_columns
            )
            # ⑧ 교차검증으로 최적 모델 선택
            latest_model = _select_best_model(X_init_scaled, y_init)
            status.update(
                label=f"✅ 초기 학습 완료 (depth={latest_model.max_depth}, "
                      f"기준일: {date_range[0].strftime('%Y-%m-%d')})",
                state="complete"
            )
        else:
            status.update(label="❌ 학습 데이터 없음", state="error")

    if latest_model is None or final_model_columns is None:
        st.error("초기 학습 데이터를 가져오지 못했습니다.")
        st.stop()

    # ── 워크포워드 루프 ───────────────────────────────────────
    # ③ 올바른 워크포워드 구조
    #    i번째 루프:
    #      - curr_reb 시점 피처로 종목 추론  (모델: i-1 구간 학습)
    #      - curr_reb ~ next_reb 실제 수익률 기록
    #      - next_reb 이후: curr_reb 피처 + 실제 수익률(curr~next)로 재학습
    #        → 이 모델이 i+1번째 루프의 추론에 사용됨
    loop_bar    = st.progress(0, text="워크포워드 진행 중...")
    total_steps = len(date_range) - 1
    prev_tickers: set = set()   # 직전 구간 보유 종목 추적 (교체율 계산용)

    for i in range(total_steps):
        curr_reb = date_range[i]
        next_reb = date_range[i + 1]
        loop_bar.progress(
            int(i / total_steps * 100),
            text=f"리밸런싱 {i+1}/{total_steps} ({curr_reb.strftime('%Y-%m')})"
        )

        # ── 당시 시가총액 기준 유니버스 동적 선정 ────────────────
        universe, mktcap_df = get_universe_by_mktcap(
            tickers, curr_reb, full_hist_data, source_cache, max_tickers
        )
        # 유니버스가 너무 작으면 원래 tickers 사용
        if len(universe) < top_n:
            universe = tickers[:max_tickers]

        # ① curr_reb 시점 피처로 추론 (전체 tickers → universe로 교체)
        inference_df = fetch_ml_data_optimized_pit(
            universe, curr_reb, full_hist_data, source_cache, is_training=False
        )
        if inference_df.empty:
            continue

        X_infer = (
            inference_df.drop(["Ticker"], axis=1)
            .reindex(columns=final_model_columns)
            .fillna(0)
        )
        # ⑦ 스케일링 적용 (transform만, fit은 하지 않음)
        X_infer_scaled = pd.DataFrame(
            scaler.transform(X_infer),
            columns=final_model_columns
        )
        inference_df = inference_df.copy()
        inference_df["Prediction"] = latest_model.predict(X_infer_scaled)

        selected_rows = inference_df.nlargest(top_n, "Prediction").copy()
        sel_tickers   = selected_rows["Ticker"].tolist()
        predictions   = selected_rows["Prediction"].values

        importances = pd.Series(latest_model.feature_importances_, index=final_model_columns)
        importance_history.append({"Date": curr_reb.strftime("%Y-%m-%d"), **importances.to_dict()})
        meta = get_rebalance_meta(universe, curr_reb, full_hist_data, source_cache)
        rebalance_details.append({
            "date":          curr_reb.strftime("%Y-%m-%d"),
            "next_date":     next_reb.strftime("%Y-%m-%d"),
            "selected_data": selected_rows,
            "importance":    importances,
            "meta":          meta,
            "best_depth":    latest_model.max_depth,
            "universe_size": len(universe),
            "universe_top3": (
                mktcap_df.head(3)["Ticker"].tolist()
                if not mktcap_df.empty else []
            ),
        })

        # ⑤ 비중 계산 + 교체율 기반 거래비용
        valid_sel  = [t for t in sel_tickers if t in full_hist_data.columns.get_level_values(0)]
        curr_set   = set(valid_sel)

        # 교체율 계산: 이전 구간 대비 새로 매수/매도하는 종목 비율
        if prev_tickers:
            new_buys  = curr_set - prev_tickers
            new_sells = prev_tickers - curr_set
            # 교체 종목 수 / 전체 보유 종목 수 = 교체율
            turnover  = len(new_buys) / max(len(curr_set), 1)
        else:
            # 첫 리밸런싱: 전액 신규 매수
            turnover = 1.0

        # 실제 거래비용 = 편도 비용 × 교체율 × 2 (매수 + 매도)
        actual_tc = TRANSACTION_COST * turnover * 2

        # rebalance_details에 교체율 정보 추가
        rebalance_details[-1]["turnover"]    = turnover
        rebalance_details[-1]["new_buys"]    = sorted(curr_set - prev_tickers)
        rebalance_details[-1]["new_sells"]   = sorted(prev_tickers - curr_set)
        rebalance_details[-1]["kept"]        = sorted(curr_set & prev_tickers)
        rebalance_details[-1]["actual_tc"]   = actual_tc

        prev_tickers = curr_set   # 다음 루프를 위해 저장

        if valid_sel:
            subset = full_hist_data[valid_sel].loc[curr_reb:next_reb]
            if not subset.empty:
                try:
                    test_prices = subset.xs("Close", axis=1, level=1)
                    stock_rets  = test_prices.pct_change().fillna(0)

                    if weight_mode == "AI 점수 비례":
                        valid_preds = np.array([
                            predictions[sel_tickers.index(t)]
                            for t in valid_sel
                        ])
                        exp_p   = np.exp(valid_preds - valid_preds.max())
                        weights = exp_p / exp_p.sum()
                    else:
                        weights = np.ones(len(valid_sel)) / len(valid_sel)

                    weighted_rets = stock_rets.values @ weights
                    period_rets   = pd.Series(weighted_rets, index=stock_rets.index)

                    # 교체율 기반 거래비용 반영 (첫날에 일괄 차감)
                    period_rets.iloc[0] -= actual_tc

                    all_strategy_returns = pd.concat([all_strategy_returns, period_rets])
                except Exception:
                    pass

        # ③ 진짜 워크포워드 재학습 (universe 기준)
        train_df_wf = fetch_ml_data_optimized_pit(
            universe, curr_reb, full_hist_data, source_cache, is_training=True
        )
        if not train_df_wf.empty and "Target_Return" in train_df_wf.columns:
            X_wf = train_df_wf.drop(["Ticker", "Target_Return"], axis=1)
            y_wf = train_df_wf["Target_Return"]
            # ⑦ 재학습 시 스케일러도 갱신
            X_wf_scaled = pd.DataFrame(
                scaler.fit_transform(X_wf),
                columns=final_model_columns
            )
            # ⑧ 매 구간마다 최적 depth 재선택
            latest_model = _select_best_model(X_wf_scaled, y_wf)

    loop_bar.progress(100, text="워크포워드 완료!")

    # ─────────────────────────────────────────────
    # 결과 시각화
    # ─────────────────────────────────────────────
    if all_strategy_returns.empty:
        st.warning("수익률 데이터를 생성하지 못했습니다.")
        st.stop()

    all_strategy_returns = (
        all_strategy_returns[~all_strategy_returns.index.duplicated()].sort_index()
    )
    start_ts, end_ts = all_strategy_returns.index[0], all_strategy_returns.index[-1]

    bench_raw = yf.download(
        ["SPY", "QQQ", "TQQQ"],
        start=start_ts - timedelta(days=5),
        end=end_ts + timedelta(days=5),
        progress=False,
    )["Close"]
    bench_raw = bench_raw.ffill().reindex(all_strategy_returns.index).ffill()

    spy_rets  = bench_raw["SPY"].pct_change().fillna(0)
    qqq_rets  = bench_raw["QQQ"].pct_change().fillna(0)
    tqqq_rets = bench_raw["TQQQ"].pct_change().fillna(0)

    st.header(f"📊 {'전체' if '전체' in selected_industries else '+'.join(selected_industries)} 전략 성과 보고서")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 누적 수익률 비교")
        cum_returns = pd.DataFrame({
            "Strategy (AI)": (1 + all_strategy_returns).cumprod(),
            "SPY":   (1 + spy_rets).cumprod(),
            "QQQ":   (1 + qqq_rets).cumprod(),
            "TQQQ":  (1 + tqqq_rets).cumprod(),
        })
        fig_cum = px.line(
            cum_returns, x=cum_returns.index, y=cum_returns.columns,
            color_discrete_map={
                "Strategy (AI)": "firebrick",
                "SPY": "royalblue", "QQQ": "seagreen", "TQQQ": "orange",
            },
        )
        fig_cum.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    with col2:
        # ⑥ 확장된 성과 지표 테이블
        st.subheader("💡 상세 성과 지표")
        metrics_df = pd.DataFrame({
            "AI Strategy": get_extended_metrics(all_strategy_returns, "AI"),
            "SPY":         get_extended_metrics(spy_rets,  "SPY"),
            "QQQ":         get_extended_metrics(qqq_rets,  "QQQ"),
            "TQQQ":        get_extended_metrics(tqqq_rets, "TQQQ"),
        })
        st.table(metrics_df)

        # ⑥ 알파/베타 (vs SPY)
        try:
            aligned = pd.concat(
                [all_strategy_returns, spy_rets], axis=1, join="inner"
            ).dropna()
            aligned.columns = ["strat", "spy"]
            beta  = aligned.cov().iloc[0, 1] / aligned["spy"].var()
            alpha = (aligned["strat"].mean() - beta * aligned["spy"].mean()) * 252
            st.markdown(f"**vs SPY —** Alpha: `{alpha*100:.2f}%` / Beta: `{beta:.2f}`")
        except Exception:
            pass

    # ── 리밸런싱 히스토리 ─────────────────────────────────────
    st.divider()
    st.subheader("🗓️ 리밸런싱 히스토리")

    for detail in reversed(rebalance_details):
        meta       = detail.get("meta", {})
        check_cols = [c for c in ["P/E", "ROE"] if c in detail["selected_data"].columns]
        has_fin    = (
            not (detail["selected_data"][check_cols].abs() < 0.0001).all().all()
            if check_cols else False
        )
        q_count = meta.get("q_count", 0)
        a_count = meta.get("a_count", 0)
        n_count = meta.get("n_count", 0)
        total   = meta.get("total", 1) or 1

        if q_count / total >= 0.7:
            data_badge = "🟢 분기 데이터"
        elif a_count / total >= 0.5:
            data_badge = "🟡 연간(분기 대체)"
        else:
            data_badge = "🔴 데이터 부족"

        fin_badge    = "✅ 재무+가격" if has_fin else "⚠️ 모멘텀 중심"
        period_start = meta.get("period_start", "N/A")
        period_end   = meta.get("period_end", detail["date"])
        next_date    = detail.get("next_date", "N/A")
        best_depth   = detail.get("best_depth", "-")
        universe_size = detail.get("universe_size", "-")
        universe_top3 = detail.get("universe_top3", [])

        with st.expander(
            f"📅 {detail['date']}  |  {fin_badge}  |  {data_badge}  |  유니버스 {universe_size}개",
            expanded=False
        ):
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"📆 **리밸런싱 기준일**<br>{detail['date']}", unsafe_allow_html=True)
            m2.markdown(f"📈 **보유 기간**<br>{period_end} ~ {next_date}", unsafe_allow_html=True)
            m3.markdown(f"📊 **주가 학습 기간**<br>{period_start} ~ {period_end}", unsafe_allow_html=True)
            m4.markdown(
                f"🗃️ **재무 데이터**<br>분기 {q_count}개 / 연간 {a_count}개 / 없음 {n_count}개"
                f"<br>🌲 모델 depth: {best_depth}",
                unsafe_allow_html=True
            )

            # 유니버스 정보
            if universe_top3:
                st.caption(
                    f"🏆 당시 시가총액 상위 3개: {', '.join(universe_top3)} "
                    f"(전체 유니버스 {universe_size}개 중 AI가 {top_n}개 선정)"
                )
            st.divider()

            # ── 교체율 및 거래비용 정보 ──────────────────────────
            turnover  = detail.get("turnover", 0)
            new_buys  = detail.get("new_buys",  [])
            new_sells = detail.get("new_sells", [])
            kept      = detail.get("kept",      [])
            actual_tc = detail.get("actual_tc", 0)

            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.markdown(
                f"🔄 **교체율**<br>{turnover * 100:.0f}%",
                unsafe_allow_html=True
            )
            tc2.markdown(
                f"💰 **실제 거래비용**<br>{actual_tc * 100:.4f}%"
                f" ({actual_tc * 10000:.1f}bps)",
                unsafe_allow_html=True
            )
            tc3.markdown(
                f"🟢 **신규 매수** ({len(new_buys)}종목)<br>"
                + (", ".join(new_buys) if new_buys else "없음"),
                unsafe_allow_html=True
            )
            tc4.markdown(
                f"🔴 **매도** ({len(new_sells)}종목)<br>"
                + (", ".join(new_sells) if new_sells else "없음"),
                unsafe_allow_html=True
            )
            if kept:
                st.markdown(
                    f"⚪ **유지 종목** ({len(kept)}종목, 거래비용 없음): "
                    + ", ".join(f"**{t}**" for t in kept)
                )
            st.divider()

            data_types         = meta.get("data_types", {})
            sel_tickers_detail = (
                detail["selected_data"]["Ticker"].tolist()
                if "Ticker" in detail["selected_data"].columns else []
            )
            if data_types and sel_tickers_detail:
                tag_parts = []
                for t in sel_tickers_detail:
                    dtype = data_types.get(t, "없음")
                    icon  = "🟢" if dtype == "분기" else ("🟡" if dtype == "연간(분기 대체)" else "🔴")
                    tag_parts.append(f"{icon} **{t}** ({dtype})")
                st.markdown("**선정 종목 데이터 타입:**  " + "  ·  ".join(tag_parts))
                st.divider()

            d1, d2 = st.columns([3, 2])
            with d1:
                st.markdown("**선정 종목 상세**")
                st.dataframe(
                    detail["selected_data"].drop(
                        columns=["Target_Return", "Prediction"], errors="ignore"
                    ),
                    use_container_width=True, hide_index=True,
                )
            with d2:
                st.markdown("**지표 중요도 (Top 10)**")
                imp_df = detail["importance"].nlargest(10).reset_index()
                imp_df.columns = ["지표", "중요도"]
                fig_imp = px.bar(
                    imp_df, x="중요도", y="지표", orientation="h",
                    color="중요도", color_continuous_scale="Greens", height=300,
                )
                fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True, key=f"imp_{detail['date']}")

    # ── 지표 중요도 트렌드 ───────────────────────────────────
    st.subheader("🧠 지표 중요도 트렌드")
    imp_all_df = pd.DataFrame(importance_history).set_index("Date").fillna(0)
    top5       = imp_all_df.mean().nlargest(5).index.tolist()
    st.line_chart(imp_all_df[top5])

    imp_norm     = imp_all_df.div(imp_all_df.sum(axis=1), axis=0).fillna(0) * 100
    top7         = imp_norm.mean().nlargest(7).index.tolist()
    display_norm = imp_norm[top7].reset_index()
    fig_area = px.area(
        display_norm, x="Date", y=top7,
        title="주요 지표 영향력 비중 추이 (Top 7)",
        labels={"value": "중요도 비중 (%)", "Date": "리밸런싱 시점", "variable": "지표"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_area.update_layout(
        hovermode="x unified", legend_orientation="h",
        legend_y=-0.2, yaxis_range=[0, 100],
    )
    st.plotly_chart(fig_area, use_container_width=True)
    display_importance_heatmap(imp_all_df)

    # ── 실시간 추천 ──────────────────────────────────────────
    st.divider()
    st.subheader(f"🎯 실시간 AI 추천 종목 (Next {reb_months}M)")
    latest_data = fetch_ml_data_optimized_pit(
        tickers, datetime.now(), full_hist_data, source_cache, is_training=False
    )
    recommend_all = pd.DataFrame()
    display_cols  = []
    if not latest_data.empty:
        X_latest = (
            latest_data.drop(["Ticker"], axis=1, errors="ignore")
            .reindex(columns=final_model_columns).fillna(0)
        )
        X_latest_scaled = pd.DataFrame(
            scaler.transform(X_latest), columns=final_model_columns
        )
        latest_data = latest_data.copy()
        latest_data["AI_Score"] = latest_model.predict(X_latest_scaled)
        recommend_all = latest_data.sort_values("AI_Score", ascending=False)
        display_cols  = ["Ticker", "AI_Score"] + [
            c for c in final_model_columns if c in recommend_all.columns
        ]
        st.dataframe(
            recommend_all[display_cols]
            .style.background_gradient(subset=["AI_Score"], cmap="YlGn")
            .format(precision=3),
            use_container_width=True, hide_index=True,
        )
    else:
        st.warning("실시간 추천 데이터를 생성할 수 없습니다.")

    # ── 다운로드 ─────────────────────────────────────────────
    st.divider()
    st.subheader("📥 분석 결과 내보내기")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📈 누적 수익률 (CSV)",
            data=cum_returns.to_csv().encode("utf-8-sig"),
            file_name=f"returns_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True,
        )
    with dl2:
        if not recommend_all.empty:
            st.download_button(
                "🎯 AI 추천 종목 (CSV)",
                data=recommend_all[display_cols].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"recommendation_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )
    dl3, dl4 = st.columns(2)
    with dl3:
        if rebalance_details:
            history_dfs = []
            for detail in rebalance_details:
                tmp = detail["selected_data"].copy()
                tmp.insert(0, "Rebalance_Date", detail["date"])
                history_dfs.append(tmp)
            st.download_button(
                "📜 리밸런싱 히스토리 (CSV)",
                data=pd.concat(history_dfs, ignore_index=True).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )
    with dl4:
        raw_list = []
        for detail in rebalance_details:
            raw_step = fetch_ml_data_optimized_pit(
                tickers, pd.to_datetime(detail["date"]),
                full_hist_data, source_cache, is_training=False,
            )
            raw_step.insert(0, "Data_Date", detail["date"])
            raw_list.append(raw_step)
        if raw_list:
            st.download_button(
                "📊 원본 피처 데이터 (CSV)",
                data=pd.concat(raw_list, ignore_index=True).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"raw_features_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )

else:
    st.info("섹터를 선택하고 백테스트를 실행하세요.")