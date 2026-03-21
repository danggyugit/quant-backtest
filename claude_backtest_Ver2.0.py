#!/usr/bin/env python3
"""
claude_backtest_Ver2.0.py
AI 퀀트 백테스팅 & 실시간 종목 추천 플랫폼
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
import requests
from io import StringIO
import warnings
import concurrent.futures
import time
import random
import os
import calendar
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Quant Lab 2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── 전체 배경 흰색 ── */
    .stApp { background: #ffffff; }
    section[data-testid="stSidebar"] { display: none; }

    .main-title {
        font-size: 2.0rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title { font-size: 0.9rem; color: #555; margin-bottom: 0.8rem; }
    .section-hdr {
        font-size: 0.95rem; font-weight: 700; color: #1565c0;
        margin: 0.8rem 0 0.4rem 0; padding-bottom: 0.2rem;
        border-bottom: 2px solid #e3f2fd;
    }
    .metric-box {
        background: #f5f7fa; border-radius: 8px; padding: 0.7rem 1rem;
        border: 1px solid #dee2e6; text-align: center; margin-bottom: 0.3rem;
    }
    .metric-label { font-size: 0.7rem; color: #555; }
    .metric-value { font-size: 1.2rem; font-weight: 700; }
    .pos { color: #2e7d32; } .neg { color: #c62828; } .neu { color: #1565c0; }

    /* ── 상단 설정 바 ── */
    .settings-bar {
        background: #f8f9fa; border-radius: 10px; padding: 1rem 1.2rem;
        border: 1px solid #dee2e6; margin-bottom: 1rem;
    }
    .settings-title {
        font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 0.5rem;
    }

    /* ── Streamlit 기본 요소 색상 보정 ── */
    .stSlider > label { color: #222 !important; font-weight: 600; }
    .stNumberInput > label { color: #222 !important; font-weight: 600; }
    .stMultiSelect > label { color: #222 !important; font-weight: 600; }
    .stCheckbox > label { color: #222 !important; }
    .stDateInput > label { color: #222 !important; font-weight: 600; }
    div[data-testid="stExpander"] summary p { color: #222 !important; font-weight: 700; }

    /* ── 탭 스타일 ── */
    .stTabs [data-baseweb="tab-list"] { gap: 0.3rem; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem; font-weight: 600; color: #444;
    }
    .stTabs [aria-selected="true"] { color: #1565c0 !important; }

    /* ── 모바일 대응 ── */
    @media (max-width: 768px) {
        .main-title { font-size: 1.4rem; }
        .metric-value { font-size: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
IS_CLOUD = any([
    os.environ.get("STREAMLIT_SERVER_HEADLESS", "").lower() in ("1", "true"),
    os.environ.get("STREAMLIT_SHARING_MODE", "") != "",
    os.environ.get("HOME", "") == "/home/appuser",   # Streamlit Cloud 기본 홈
])
MAX_WORKERS  = 1 if IS_CLOUD else 6   # 클라우드: 동시 요청 최소화
CLOUD_DELAY  = 0.4 if IS_CLOUD else 0.0  # 클라우드: 요청 간 딜레이(초)
REPORT_LAG  = 45  # 재무보고 지연일

BENCHMARKS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "TQQQ": "3x Nasdaq"}

FEATURE_META = {
    # ── 모멘텀 ──────────────────────────────────────────────
    "Mom_1w":            {"name": "1주 수익률",        "group": "모멘텀"},
    "Mom_1m":            {"name": "1개월 수익률",      "group": "모멘텀"},
    "Mom_3m":            {"name": "3개월 수익률",      "group": "모멘텀"},
    "Mom_6m":            {"name": "6개월 수익률",      "group": "모멘텀"},
    "Mom_12m":           {"name": "12개월 수익률",     "group": "모멘텀"},
    "Mom_12_1":          {"name": "12-1개월 모멘텀",   "group": "모멘텀"},
    "Mom_6_1":           {"name": "6-1개월 모멘텀",    "group": "모멘텀"},
    "Momentum_Custom":   {"name": "3개월 커스텀",      "group": "모멘텀"},
    # ── 기술지표 ─────────────────────────────────────────────
    "RSI_14":            {"name": "RSI(14)",            "group": "기술지표"},
    "MACD_Hist":         {"name": "MACD 히스토그램",   "group": "기술지표"},
    "Price_SMA20":       {"name": "Price/SMA20",        "group": "기술지표"},
    "Price_SMA50":       {"name": "Price/SMA50",        "group": "기술지표"},
    "Price_SMA200":      {"name": "Price/SMA200",       "group": "기술지표"},
    "SMA50_SMA200":      {"name": "SMA50/SMA200 골든",  "group": "기술지표"},
    "ADX_14":            {"name": "ADX(14)",            "group": "기술지표"},
    "Stoch_K":           {"name": "Stochastic %K",      "group": "기술지표"},
    "CCI_20":            {"name": "CCI(20)",            "group": "기술지표"},
    "WilliamsR_14":      {"name": "Williams %R(14)",    "group": "기술지표"},
    "MFI_14":            {"name": "MFI(14)",            "group": "기술지표"},
    "BB_Width":          {"name": "볼린저밴드 폭",      "group": "기술지표"},
    "BB_Pos":            {"name": "볼린저밴드 위치",    "group": "기술지표"},
    "High52w_Dist":      {"name": "52주 고점 대비",     "group": "기술지표"},
    "Low52w_Dist":       {"name": "52주 저점 대비",     "group": "기술지표"},
    # ── 리스크/거래량 ─────────────────────────────────────
    "Volatility_30d":    {"name": "30일 변동성",        "group": "리스크"},
    "Volatility_90d":    {"name": "90일 변동성",        "group": "리스크"},
    "ATR_Ratio":         {"name": "ATR 비율",           "group": "리스크"},
    "Vol_Ratio":         {"name": "거래량 비율(20일)",  "group": "리스크"},
    "Risk_Adj_Return":   {"name": "위험조정수익",       "group": "리스크"},
    # ── 밸류에이션 ─────────────────────────────────────────
    "P_E":               {"name": "P/E 비율",           "group": "밸류에이션"},
    "P_B":               {"name": "P/B 비율",           "group": "밸류에이션"},
    "P_S":               {"name": "P/S 비율",           "group": "밸류에이션"},
    "EV_EBITDA":         {"name": "EV/EBITDA",          "group": "밸류에이션"},
    "P_FCF":             {"name": "P/FCF",              "group": "밸류에이션"},
    "FCF_Yield":         {"name": "FCF 수익률",         "group": "밸류에이션"},
    "Div_Yield":         {"name": "배당수익률",         "group": "밸류에이션"},
    "PEG_Ratio":         {"name": "PEG 비율",           "group": "밸류에이션"},
    # ── 수익성 ─────────────────────────────────────────────
    "ROE":               {"name": "ROE",                "group": "수익성"},
    "ROA":               {"name": "ROA",                "group": "수익성"},
    "Gross_Margin":      {"name": "매출총이익률",       "group": "수익성"},
    "Op_Margin":         {"name": "영업이익률",         "group": "수익성"},
    "EBITDA_Margin":     {"name": "EBITDA 마진",        "group": "수익성"},
    # ── 성장성 ─────────────────────────────────────────────
    "Rev_Growth":        {"name": "매출 성장률",        "group": "성장성"},
    "NI_Growth":         {"name": "순이익 성장률",      "group": "성장성"},
    "EPS_Growth":        {"name": "EPS 성장률",         "group": "성장성"},
    # ── 재무안정성 ─────────────────────────────────────────
    "Debt_Equity":       {"name": "부채비율",           "group": "재무안정성"},
    "Current_Ratio":     {"name": "유동비율",           "group": "재무안정성"},
    "Interest_Coverage": {"name": "이자보상배율",       "group": "재무안정성"},
    # ── 효율성/규모 ────────────────────────────────────────
    "Asset_Turnover":    {"name": "자산회전율",         "group": "효율성"},
    "GP_A_Quality":      {"name": "GP/자산 품질",       "group": "효율성"},
    "MktCap_Log":        {"name": "시가총액(log)",      "group": "규모"},
}

FEAT_COLS  = list(FEATURE_META.keys())
FEAT_NAMES = {k: v["name"] for k, v in FEATURE_META.items()}

PLOT_CFG = dict(
    template="plotly_white",
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(248,249,250,1)",
)

# ═══════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_info() -> tuple:
    """Wikipedia에서 S&P 500 종목 목록 조회."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(StringIO(resp.text))[0]
        df = df.rename(columns={"Symbol": "ticker", "GICS Sector": "sector", "Security": "name"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        sectors = sorted(df["sector"].dropna().unique().tolist())
        return df[["ticker", "name", "sector"]], sectors
    except Exception as e:
        st.warning(f"S&P 500 목록 조회 실패: {e}. 내장 리스트 사용.")
        return _fallback_sp500(), _fallback_sectors()


def _fallback_sp500() -> pd.DataFrame:
    data = [
        ("AAPL","Apple","Information Technology"),("MSFT","Microsoft","Information Technology"),
        ("NVDA","Nvidia","Information Technology"),("GOOGL","Alphabet","Communication Services"),
        ("META","Meta","Communication Services"),("AMZN","Amazon","Consumer Discretionary"),
        ("TSLA","Tesla","Consumer Discretionary"),("AMD","AMD","Information Technology"),
        ("INTC","Intel","Information Technology"),("QCOM","Qualcomm","Information Technology"),
        ("CRM","Salesforce","Information Technology"),("ORCL","Oracle","Information Technology"),
        ("ADBE","Adobe","Information Technology"),("TXN","TI","Information Technology"),
        ("AVGO","Broadcom","Information Technology"),("MU","Micron","Information Technology"),
        ("CSCO","Cisco","Information Technology"),("HPQ","HP","Information Technology"),
        ("JNJ","J&J","Health Care"),("PFE","Pfizer","Health Care"),
        ("UNH","UnitedHealth","Health Care"),("ABT","Abbott","Health Care"),
        ("MRK","Merck","Health Care"),("LLY","Eli Lilly","Health Care"),
        ("BMY","BMS","Health Care"),("AMGN","Amgen","Health Care"),
        ("MDT","Medtronic","Health Care"),("TMO","Thermo Fisher","Health Care"),
        ("ABBV","AbbVie","Health Care"),("CVS","CVS Health","Health Care"),
        ("GILD","Gilead","Health Care"),("REGN","Regeneron","Health Care"),
        ("JPM","JPMorgan","Financials"),("BAC","Bank of America","Financials"),
        ("WFC","Wells Fargo","Financials"),("GS","Goldman","Financials"),
        ("MS","Morgan Stanley","Financials"),("C","Citigroup","Financials"),
        ("AXP","AmEx","Financials"),("BLK","BlackRock","Financials"),
        ("SCHW","Schwab","Financials"),("COF","Capital One","Financials"),
        ("HD","Home Depot","Consumer Discretionary"),("MCD","McDonald's","Consumer Discretionary"),
        ("NKE","Nike","Consumer Discretionary"),("SBUX","Starbucks","Consumer Discretionary"),
        ("LOW","Lowe's","Consumer Discretionary"),("TJX","TJX","Consumer Discretionary"),
        ("PG","P&G","Consumer Staples"),("KO","Coca-Cola","Consumer Staples"),
        ("PEP","PepsiCo","Consumer Staples"),("WMT","Walmart","Consumer Staples"),
        ("COST","Costco","Consumer Staples"),("PM","Philip Morris","Consumer Staples"),
        ("MO","Altria","Consumer Staples"),("CL","Colgate","Consumer Staples"),
        ("XOM","ExxonMobil","Energy"),("CVX","Chevron","Energy"),
        ("COP","ConocoPhillips","Energy"),("SLB","Schlumberger","Energy"),
        ("EOG","EOG Resources","Energy"),("MPC","Marathon","Energy"),
        ("HON","Honeywell","Industrials"),("UPS","UPS","Industrials"),
        ("CAT","Caterpillar","Industrials"),("DE","Deere","Industrials"),
        ("GE","GE","Industrials"),("LMT","Lockheed","Industrials"),
        ("NEE","NextEra","Utilities"),("DUK","Duke","Utilities"),
        ("SO","Southern","Utilities"),("D","Dominion","Utilities"),
        ("AMT","American Tower","Real Estate"),("PLD","Prologis","Real Estate"),
        ("CCI","Crown Castle","Real Estate"),("EQIX","Equinix","Real Estate"),
        ("LIN","Linde","Materials"),("APD","Air Products","Materials"),
        ("SHW","Sherwin","Materials"),("FCX","Freeport","Materials"),
        ("NFLX","Netflix","Communication Services"),("DIS","Disney","Communication Services"),
        ("CMCSA","Comcast","Communication Services"),("VZ","Verizon","Communication Services"),
        ("T","AT&T","Communication Services"),
    ]
    return pd.DataFrame(data, columns=["ticker","name","sector"])


def _fallback_sectors() -> list:
    return sorted(["Information Technology","Health Care","Financials",
                   "Consumer Discretionary","Consumer Staples","Energy",
                   "Industrials","Materials","Real Estate","Utilities",
                   "Communication Services"])


@st.cache_data(ttl=3600, show_spinner=False)
def download_price_data(tickers: tuple, start: str, end: str) -> dict:
    """yfinance에서 OHLCV 데이터 일괄 다운로드."""
    result = {}
    batch = 10 if IS_CLOUD else 20   # 클라우드: 배치 크기 줄여 안정성 확보
    tlist = list(tickers)
    for i in range(0, len(tlist), batch):
        chunk = tlist[i: i + batch]
        for attempt in range(3):   # 최대 3회 재시도
            try:
                raw = yf.download(chunk, start=start, end=end,
                                  auto_adjust=True, progress=False,
                                  threads=(not IS_CLOUD))  # 클라우드: 단일 스레드
                if raw.empty:
                    break
                if isinstance(raw.columns, pd.MultiIndex):
                    for t in chunk:
                        try:
                            sub = raw.xs(t, axis=1, level=1).dropna(how="all")
                            if len(sub) >= 60:
                                result[t] = sub
                        except Exception:
                            pass
                else:
                    if len(chunk) == 1 and len(raw) >= 60:
                        result[chunk[0]] = raw
                break  # 성공 시 재시도 루프 탈출
            except Exception:
                if attempt < 2:
                    time.sleep(1.5 ** attempt)  # 0s, 1.5s 후 재시도
        if CLOUD_DELAY and i + batch < len(tlist):
            time.sleep(CLOUD_DELAY)   # 배치 간 딜레이
    return result


@st.cache_data(ttl=7200, show_spinner=False)
def get_fundamental_yf(tickers: tuple) -> dict:
    """yfinance .info에서 펀더멘털 데이터 취득."""
    out = {}
    for t in tickers:
        info = {}
        for attempt in range(3):   # 최대 3회 재시도
            try:
                info = yf.Ticker(t).info or {}
                break
            except Exception:
                if attempt < 2:
                    time.sleep(1.0 + attempt)  # 1s, 2s 후 재시도
        try:
            mkt  = info.get("marketCap", 0) or 0
            out[t] = {
                "pe":         info.get("trailingPE", np.nan),
                "fwd_pe":     info.get("forwardPE", np.nan),
                "pb":         info.get("priceToBook", np.nan),
                "ps":         info.get("priceToSalesTrailing12Months", np.nan),
                "ev_ebitda":  info.get("enterpriseToEbitda", np.nan),
                "peg":        info.get("pegRatio", np.nan),
                "div_yield":  (info.get("dividendYield") or 0),
                "roe":        info.get("returnOnEquity", np.nan),
                "roa":        info.get("returnOnAssets", np.nan),
                "gross_mg":   info.get("grossMargins", np.nan),
                "op_mg":      info.get("operatingMargins", np.nan),
                "net_mg":     info.get("profitMargins", np.nan),
                "rev_growth": info.get("revenueGrowth", np.nan),
                "ni_growth":  info.get("earningsGrowth", np.nan),
                "eps_growth": info.get("earningsQuarterlyGrowth", np.nan),
                "debt_eq":    info.get("debtToEquity", np.nan),
                "curr_ratio": info.get("currentRatio", np.nan),
                "fcf":        info.get("freeCashflow", np.nan),
                "revenue":    info.get("totalRevenue", np.nan),
                "net_income": info.get("netIncomeToCommon", np.nan),
                "total_debt": info.get("totalDebt", np.nan),
                "total_cash": info.get("totalCash", np.nan),
                "total_assets": info.get("totalAssets", np.nan),
                "ebitda":     info.get("ebitda", np.nan),
                "shares":     info.get("sharesOutstanding", np.nan),
                "mkt_cap":    mkt,
                "ev":         info.get("enterpriseValue", np.nan),
                "interest_exp": info.get("interestExpense", np.nan),
                "gross_profit": info.get("grossProfit", np.nan),
                "ebit":       info.get("ebit", np.nan),
            }
        except Exception:
            out[t] = {}
        if CLOUD_DELAY:
            time.sleep(CLOUD_DELAY)   # 종목 간 딜레이 (rate limit 방지)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_prices(start: str, end: str) -> dict:
    bm = {}
    for tk in BENCHMARKS:
        try:
            df = yf.download(tk, start=start, end=end,
                             auto_adjust=True, progress=False)
            if not df.empty:
                bm[tk] = df["Close"].squeeze()
        except Exception:
            pass
    return bm


# ═══════════════════════════════════════════════════════════
# TECHNICAL INDICATOR CALCULATORS
# ═══════════════════════════════════════════════════════════

def _rsi(close: pd.Series, n=14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _macd(close: pd.Series):
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig  = line.ewm(span=9, adjust=False).mean()
    return line - sig  # histogram


def _adx(high, low, close, n=14) -> pd.Series:
    tr  = pd.concat([high-low,
                     (high-close.shift()).abs(),
                     (low -close.shift()).abs()], axis=1).max(axis=1)
    dm_p = np.where((high-high.shift()) > (low.shift()-low),
                    np.maximum(high-high.shift(), 0), 0)
    dm_m = np.where((low.shift()-low) > (high-high.shift()),
                    np.maximum(low.shift()-low, 0), 0)
    atr   = tr.ewm(span=n, adjust=False).mean()
    di_p  = 100 * pd.Series(dm_p, index=high.index).ewm(span=n, adjust=False).mean() / atr
    di_m  = 100 * pd.Series(dm_m, index=high.index).ewm(span=n, adjust=False).mean() / atr
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    return dx.ewm(span=n, adjust=False).mean()


def _stoch_k(high, low, close, n=14) -> pd.Series:
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    return 100 * (close - ll) / (hh - ll).replace(0, np.nan)


def _cci(high, low, close, n=20) -> pd.Series:
    tp  = (high + low + close) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _williams_r(high, low, close, n=14) -> pd.Series:
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def _mfi(high, low, close, volume, n=14) -> pd.Series:
    tp   = (high + low + close) / 3
    rmf  = tp * volume
    pos  = rmf.where(tp > tp.shift(1), 0.0)
    neg  = rmf.where(tp < tp.shift(1), 0.0)
    mfr  = pos.rolling(n).sum() / neg.rolling(n).sum().replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


def calc_all_technical(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """단일 종목의 전체 기술지표 DataFrame 사전 계산 (전체 기간)."""
    c, h, l, v = (ohlcv["Close"], ohlcv["High"],
                  ohlcv["Low"],   ohlcv["Volume"])
    logr = np.log(c / c.shift(1))

    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    std20  = c.rolling(20).std()
    bb_up  = sma20 + 2 * std20
    bb_lo  = sma20 - 2 * std20

    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    macd_h = _macd(c)

    df = pd.DataFrame(index=c.index)

    # Momentum
    df["Mom_1w"]  = c.pct_change(5)
    df["Mom_1m"]  = c.pct_change(21)
    df["Mom_3m"]  = c.pct_change(63)
    df["Mom_6m"]  = c.pct_change(126)
    df["Mom_12m"] = c.pct_change(252)
    df["Mom_12_1"]       = df["Mom_12m"] - df["Mom_1m"]
    df["Mom_6_1"]        = df["Mom_6m"]  - df["Mom_1m"]
    df["Momentum_Custom"] = c.pct_change(63)

    # Trend
    df["RSI_14"]    = _rsi(c)
    df["MACD_Hist"] = macd_h / c
    df["Price_SMA20"]  = c / sma20  - 1
    df["Price_SMA50"]  = c / sma50  - 1
    df["Price_SMA200"] = c / sma200 - 1
    df["SMA50_SMA200"] = sma50 / sma200 - 1
    df["ADX_14"]       = _adx(h, l, c)
    df["Stoch_K"]      = _stoch_k(h, l, c)
    df["CCI_20"]       = _cci(h, l, c)
    df["WilliamsR_14"] = _williams_r(h, l, c)
    df["MFI_14"]       = _mfi(h, l, c, v)
    df["BB_Width"]     = (bb_up - bb_lo) / sma20
    df["BB_Pos"]       = (c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)
    df["High52w_Dist"] = c / c.rolling(252).max() - 1
    df["Low52w_Dist"]  = c / c.rolling(252).min() - 1

    # Risk / Volume
    df["Volatility_30d"]   = logr.rolling(30).std()  * np.sqrt(252)
    df["Volatility_90d"]   = logr.rolling(90).std()  * np.sqrt(252)
    df["ATR_Ratio"]        = atr / c
    vol_sma20 = v.rolling(20).mean()
    df["Vol_Ratio"]        = v / vol_sma20.replace(0, np.nan)
    df["Risk_Adj_Return"]  = logr.rolling(21).mean() / logr.rolling(21).std().replace(0, np.nan)

    return df


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def _fund_features(fund: dict, price: float) -> dict:
    """yfinance info dict → 펀더멘털 피처."""
    mkt  = fund.get("mkt_cap", np.nan) or np.nan
    fcf  = fund.get("fcf",     np.nan) or np.nan
    ev   = fund.get("ev",      np.nan) or np.nan
    rev  = fund.get("revenue", np.nan) or np.nan
    ni   = fund.get("net_income", np.nan) or np.nan
    ta   = fund.get("total_assets", 1) or 1
    eq   = max(fund.get("total_assets", 1) - (fund.get("total_debt", 0) or 0), 1)
    ebit_val = fund.get("ebit", np.nan)
    da       = 0
    ebitda   = fund.get("ebitda", np.nan)
    gp       = fund.get("gross_profit", np.nan)
    iexp     = fund.get("interest_exp", np.nan)

    p_fcf = mkt / fcf if (fcf and fcf > 0 and mkt and mkt > 0) else np.nan
    fcf_yield = fcf / mkt if (mkt and mkt > 0 and fcf) else np.nan

    ebitda_m = (ebitda / rev) if (rev and rev != 0 and ebitda is not None and not np.isnan(ebitda)) else np.nan

    interest_cov = np.nan
    if ebit_val is not None and not np.isnan(ebit_val) and iexp and iexp != 0:
        interest_cov = ebit_val / abs(iexp)

    gpa = gp / ta if (gp is not None and not np.isnan(gp)) else np.nan
    at  = rev / ta if (rev is not None and not np.isnan(rev)) else np.nan

    return {
        "P_E":               fund.get("pe", np.nan),
        "P_B":               fund.get("pb", np.nan),
        "P_S":               fund.get("ps", np.nan),
        "EV_EBITDA":         fund.get("ev_ebitda", np.nan),
        "P_FCF":             p_fcf,
        "FCF_Yield":         fcf_yield,
        "Div_Yield":         fund.get("div_yield", 0),
        "PEG_Ratio":         fund.get("peg", np.nan),
        "ROE":               fund.get("roe", np.nan),
        "ROA":               fund.get("roa", np.nan),
        "Gross_Margin":      fund.get("gross_mg", np.nan),
        "Op_Margin":         fund.get("op_mg", np.nan),
        "EBITDA_Margin":     ebitda_m,
        "Rev_Growth":        fund.get("rev_growth", np.nan),
        "NI_Growth":         fund.get("ni_growth", np.nan),
        "EPS_Growth":        fund.get("eps_growth", np.nan),
        "Debt_Equity":       fund.get("debt_eq", np.nan),
        "Current_Ratio":     fund.get("curr_ratio", np.nan),
        "Interest_Coverage": interest_cov,
        "Asset_Turnover":    at,
        "GP_A_Quality":      gpa,
        "MktCap_Log":        np.log(fund.get("mkt_cap", 0) or 1),
    }


def snapshot_at_date(
    ticker: str,
    tech_df: pd.DataFrame,
    fund: dict,
    date: pd.Timestamp,
) -> dict | None:
    """특정 날짜 기준 단일 종목 전체 피처 추출."""
    mask = tech_df.index <= date
    if mask.sum() == 0:
        return None
    row = tech_df[mask].iloc[-1]

    features = {"ticker": ticker}
    for col in FEAT_COLS:
        if col in row.index:
            features[col] = row[col]
        else:
            features[col] = np.nan

    # 펀더멘털 (yfinance는 현재 값 사용 — Point-in-Time 근사)
    fund_feat = _fund_features(fund, float(row.name) if isinstance(row.name, float) else np.nan)
    for k, v in fund_feat.items():
        features[k] = v

    return features


def build_snapshot_df(
    tickers: list,
    tech_map: dict,
    fund_map: dict,
    date: pd.Timestamp,
    min_history: int = 252,
) -> pd.DataFrame:
    """모든 종목에 대해 특정 날짜의 피처 DataFrame 구성."""
    rows = []
    for t in tickers:
        td = tech_map.get(t)
        if td is None or len(td) < min_history:
            continue
        feat = snapshot_at_date(t, td, fund_map.get(t, {}), date)
        if feat:
            rows.append(feat)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ticker")
    return df


# ═══════════════════════════════════════════════════════════
# FORWARD RETURN
# ═══════════════════════════════════════════════════════════

def forward_return(tech_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """start → end 기간의 실제 수익률."""
    try:
        close = tech_df["Close"] if "Close" in tech_df.columns else None
        # tech_df here is original OHLCV, not indicator df
    except Exception:
        return np.nan
    return np.nan  # placeholder — actual call uses price_data dict


def fwd_ret_from_price(price_data: dict, ticker: str,
                       start: pd.Timestamp, end: pd.Timestamp) -> float:
    try:
        ohlcv = price_data[ticker]
        s = ohlcv.loc[ohlcv.index >= start, "Close"].iloc[0]
        e = ohlcv.loc[ohlcv.index >= end,   "Close"].iloc[0]
        return e / s - 1
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════

def add_months(dt: datetime, n: int) -> datetime:
    month = dt.month - 1 + n
    year  = dt.year + month // 12
    month = month % 12 + 1
    day   = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def generate_rebalance_dates(start: datetime, end: datetime, months: int) -> list:
    dates = []
    cur = start
    while cur <= end:
        dates.append(pd.Timestamp(cur))
        cur = add_months(cur, months)
    return dates


def run_backtest(
    price_data:   dict,
    fund_map:     dict,
    tech_map:     dict,
    rebal_dates:  list,
    n_stocks:     int,
    tc_pct:       float,
    rolling_win:  int,
    progress,
) -> dict:
    """메인 백테스트 엔진."""
    n_dates = len(rebal_dates)
    feature_cols = FEAT_COLS

    # ── Step 1: 모든 리밸런싱 날짜의 스냅샷 + 실제 수익률 계산 ──
    progress(0.05, "📊 지표 스냅샷 계산 중...")
    snapshots: dict = {}  # date → pd.DataFrame (includes forward_return)

    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]
        tickers   = list(price_data.keys())
        snap      = build_snapshot_df(tickers, tech_map, fund_map, date)
        if snap.empty:
            snapshots[date] = snap
            continue

        # 실제 forward return 추가
        fwd = {t: fwd_ret_from_price(price_data, t, date, next_date)
               for t in snap.index}
        snap["_fwd_return"] = pd.Series(fwd)
        snapshots[date] = snap
        progress(0.05 + 0.25 * (i / max(n_dates - 2, 1)),
                 f"스냅샷 계산 중 ({i+1}/{n_dates-1})...")

    # ── Step 2: 롤링 모델 학습 + 포트폴리오 시뮬레이션 ──
    progress(0.30, "🤖 AI 모델 학습 및 백테스트 실행 중...")

    portfolio_dates  = [rebal_dates[0]]
    portfolio_values = [1.0]
    current_value    = 1.0

    ic_records     = []
    feat_imp_rows  = []
    rebal_history  = []

    imputer = SimpleImputer(strategy="median")

    for i in range(rolling_win, n_dates - 1):
        date      = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        # ── 훈련 데이터 수집 ──────────────────────────────
        X_list, y_list = [], []
        for j in range(i - rolling_win, i):
            snap = snapshots.get(rebal_dates[j])
            if snap is None or snap.empty or "_fwd_return" not in snap.columns:
                continue
            cols = [c for c in feature_cols if c in snap.columns]
            sub  = snap[cols + ["_fwd_return"]].dropna(subset=["_fwd_return"])
            if len(sub) >= 5:
                X_list.append(sub[cols])
                y_list.append(sub["_fwd_return"])

        if not X_list:
            continue

        X_train = pd.concat(X_list)
        y_train = pd.concat(y_list)
        avail_cols = X_train.columns.tolist()

        X_imp = imputer.fit_transform(X_train)

        model = RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )
        model.fit(X_imp, y_train)

        imp_dict = dict(zip(avail_cols, model.feature_importances_))
        feat_imp_rows.append({"date": date, **imp_dict})

        # ── 예측 ─────────────────────────────────────────
        cur_snap = snapshots.get(date)
        if cur_snap is None or cur_snap.empty:
            continue

        X_pred = cur_snap[[c for c in avail_cols if c in cur_snap.columns]].reindex(columns=avail_cols)
        X_pred_imp = imputer.transform(X_pred)
        pred_returns = model.predict(X_pred_imp)
        pred_series  = pd.Series(pred_returns, index=cur_snap.index)

        # ── IC 계산 ──────────────────────────────────────
        actual = cur_snap["_fwd_return"].dropna()
        common = pred_series.index.intersection(actual.index)
        ic_val = np.nan
        if len(common) >= 10:
            ic_val, _ = spearmanr(pred_series[common], actual[common])
            ic_records.append({"date": date, "IC": ic_val})

        # ── 종목 선정 ─────────────────────────────────────
        selected = pred_series.nlargest(n_stocks)

        # ── 포트폴리오 수익률 ─────────────────────────────
        port_ret = 0.0
        valid_n  = 0
        for t in selected.index:
            r = fwd_ret_from_price(price_data, t, date, next_date)
            if not np.isnan(r):
                port_ret += r
                valid_n  += 1
        if valid_n > 0:
            port_ret /= valid_n

        # 거래비용 적용
        tc = tc_pct / 100
        current_value *= (1 + port_ret - tc)
        portfolio_dates.append(next_date)
        portfolio_values.append(current_value)

        # ── 리밸런싱 히스토리 ─────────────────────────────
        ticker_rows = []
        for t in selected.index:
            row_d = {"ticker": t, "예측수익률": pred_series[t],
                     "실제수익률": actual.get(t, np.nan)}
            for fc in feature_cols:
                row_d[FEAT_NAMES.get(fc, fc)] = cur_snap.loc[t, fc] if fc in cur_snap.columns else np.nan
            ticker_rows.append(row_d)

        top10 = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        rebal_history.append({
            "rebalance_date":  date,
            "next_date":       next_date,
            "holding_period":  f"{date.strftime('%Y-%m-%d')} ~ {next_date.strftime('%Y-%m-%d')}",
            "learn_start":     rebal_dates[i - rolling_win].strftime("%Y-%m-%d"),
            "selected":        list(selected.index),
            "ticker_df":       pd.DataFrame(ticker_rows),
            "top10_features":  top10,
            "port_return":     port_ret,
            "ic":              ic_val,
        })

        pct = 0.30 + 0.65 * (i - rolling_win) / max(n_dates - rolling_win - 1, 1)
        progress(min(pct, 0.95), f"백테스트 진행 중 ({date.strftime('%Y-%m')})...")

    progress(0.98, "결과 정리 중...")

    # ── feat_imp_history DataFrame ────────────────────────
    if feat_imp_rows:
        fimp_df = pd.DataFrame(feat_imp_rows).set_index("date")
        fimp_df.index = pd.to_datetime(fimp_df.index)
    else:
        fimp_df = pd.DataFrame()

    ic_df = pd.DataFrame(ic_records) if ic_records else pd.DataFrame()

    return {
        "port_dates":   portfolio_dates,
        "port_values":  portfolio_values,
        "ic_df":        ic_df,
        "fimp_df":      fimp_df,
        "rebal_hist":   rebal_history,
    }


# ═══════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════

def build_daily_portfolio(results: dict, price_data: dict) -> pd.Series:
    """리밸런싱 히스토리 + 일별 가격으로 AI 포트폴리오 일단위 시계열 구성."""
    hist       = results.get("rebal_hist", [])
    port_dates = [pd.Timestamp(d) for d in results["port_dates"]]
    port_vals  = results["port_values"]

    # 히스토리나 가격 데이터 없으면 기간 단위 그대로 반환
    if not hist or not price_data:
        return pd.Series(port_vals, index=pd.DatetimeIndex(port_dates))

    records: dict = {}
    cur_val = 1.0

    # ① warm-up 구간 (학습 기간): 첫 운용 시작일 전까지 1.0으로 평탄
    init_date   = port_dates[0]
    first_trade = hist[0]["rebalance_date"]
    for d in pd.bdate_range(init_date, first_trade):
        records[d] = 1.0

    # ② 리밸런싱별 일단위 포트폴리오 계산
    for h in hist:
        s_dt    = h["rebalance_date"]
        e_dt    = h["next_date"]
        tickers = h["selected"]

        price_dict = {}
        for t in tickers:
            if t not in price_data:
                continue
            ohlcv = price_data[t]
            mask  = (ohlcv.index >= s_dt) & (ohlcv.index <= e_dt)
            sub   = ohlcv.loc[mask, "Close"].dropna()
            if len(sub) >= 2:
                price_dict[t] = sub / float(sub.iloc[0])  # 기간 시작 기준 정규화

        if not price_dict:
            records[e_dt] = cur_val * (1 + h["port_return"])
            cur_val = records[e_dt]
            continue

        # 동일 가중 지수 × 현재 포트폴리오 가치
        pf_df  = pd.DataFrame(price_dict).dropna(how="all")
        eq_idx = pf_df.mean(axis=1)
        for dt, v in eq_idx.items():
            if dt >= s_dt:
                records[dt] = float(v) * cur_val

        cur_val = float(eq_idx.iloc[-1]) * cur_val

    if not records:
        return pd.Series(port_vals, index=pd.DatetimeIndex(port_dates))

    return pd.Series(records).sort_index()


def calc_metrics(series: pd.Series, label: str) -> dict:
    if len(series) < 2:
        return {"지표": label}
    r  = series.pct_change().dropna()
    yr = (series.index[-1] - series.index[0]).days / 365.25
    total = series.iloc[-1] / series.iloc[0] - 1
    cagr  = (series.iloc[-1] / series.iloc[0]) ** (1 / max(yr, 0.01)) - 1
    vol   = r.std() * np.sqrt(252)
    sharpe = (cagr - 0.02) / vol if vol > 0 else 0
    neg_r  = r[r < 0]
    sortino = (cagr - 0.02) / (neg_r.std() * np.sqrt(252)) if len(neg_r) > 1 else 0
    dd = (series / series.cummax() - 1)
    mdd    = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    monthly = series.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean() if len(monthly) > 0 else 0
    return {
        "전략":     label,
        "총수익률": f"{total:.1%}",
        "CAGR":     f"{cagr:.1%}",
        "연변동성": f"{vol:.1%}",
        "Sharpe":   f"{sharpe:.2f}",
        "Sortino":  f"{sortino:.2f}",
        "Max DD":   f"{mdd:.1%}",
        "Calmar":   f"{calmar:.2f}",
        "월승률":   f"{win_rate:.1%}",
    }


def norm_series(s: pd.Series, ref: pd.Timestamp) -> pd.Series:
    s = s.dropna()
    idx = s.index[s.index >= ref]
    if not len(idx):
        return s
    return s[idx] / s[idx[0]]


# ═══════════════════════════════════════════════════════════
# TAB 1 ── 성과 비교
# ═══════════════════════════════════════════════════════════

def tab_performance(results: dict, benchmarks: dict, price_data: dict):
    pd_ = results["port_dates"]
    pv_ = results["port_values"]

    if len(pd_) < 2:
        st.warning("백테스트 기간이 짧아 성과 데이터가 부족합니다.")
        return

    # 성과 지표 계산은 기간 단위 시리즈 사용 (정확한 TC 반영)
    port_period = pd.Series(pv_, index=pd.DatetimeIndex(pd_))
    start       = port_period.index[0]
    metrics_all = [calc_metrics(port_period, "🤖 AI 전략")]

    # 차트용: 일단위 시리즈 (벤치마크와 동일 빈도)
    port_daily = build_daily_portfolio(results, price_data)

    col_chart, col_metrics = st.columns([7, 3])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_daily.index, y=port_daily.values,
        name="🤖 AI 전략",
        line=dict(color="#7c4dff", width=3),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra>AI 전략</extra>",
    ))

    colors = {"SPY": "#ff6b35", "QQQ": "#00c9a7", "TQQQ": "#ffd700"}
    for tk, label in BENCHMARKS.items():
        if tk in benchmarks:
            ns = norm_series(benchmarks[tk], start)
            if len(ns) > 1:
                fig.add_trace(go.Scatter(
                    x=ns.index, y=ns.values, name=label,
                    line=dict(color=colors.get(tk, "#888"), width=2),
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:.3f}}<extra>{label}</extra>",
                ))
                metrics_all.append(calc_metrics(ns, label))

    fig.update_layout(
        **PLOT_CFG, height=420,
        title="포트폴리오 누적 수익률 비교",
        xaxis_title="날짜", yaxis_title="누적 수익 (1.0 = 시작)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    with col_chart:
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.markdown('<div class="section-hdr">📈 성과 지표 비교</div>', unsafe_allow_html=True)
        mdf = pd.DataFrame(metrics_all).set_index("전략").T
        st.dataframe(mdf, use_container_width=True, height=300)

        ai = metrics_all[0]
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        cagr_n = float(ai["CAGR"].strip("%"))
        mdd_n  = float(ai["Max DD"].strip("%"))
        c1.markdown(f"""<div class="metric-box">
            <div class="metric-label">CAGR</div>
            <div class="metric-value {'pos' if cagr_n > 0 else 'neg'}">{ai['CAGR']}</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-box">
            <div class="metric-label">Sharpe</div>
            <div class="metric-value neu">{ai['Sharpe']}</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-box">
            <div class="metric-label">Max DD</div>
            <div class="metric-value neg">{ai['Max DD']}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 ── IC 분석
# ═══════════════════════════════════════════════════════════

def tab_ic(results: dict):
    # ── IC 분석 개념 설명 ──────────────────────────────────
    with st.expander("📖 IC 분석이란?", expanded=False):
        st.markdown("""
**IC (Information Coefficient, 정보 계수)**는 AI 모델의 예측력을 평가하는 핵심 지표입니다.

| 지표 | 정의 | 기준 |
|------|------|------|
| **IC** | AI 예측 수익률 순위와 실제 수익률 순위 간의 **Spearman 상관계수** | > 0.05 = 좋음, > 0 = 유효 |
| **IC IR** | IC 평균 ÷ IC 표준편차 (예측 일관성) | > 0.5 = 우수, > 0.3 = 양호 |
| **양(+)IC 비율** | IC > 0인 리밸런싱 기간 비율 | > 60% = 안정적 |

**해석 방법**
- IC가 꾸준히 **양수(+)**이면 AI 모델이 상승 종목을 잘 예측함을 의미합니다.
- IC = 0이면 예측력 없음, IC < 0이면 역방향 예측 (위험 신호).
- 누적 IC가 **우상향** 추세이면 모델 품질이 일관적으로 유지되고 있습니다.
- 학술적으로 IC > 0.05이면 실용적으로 유의미한 예측력으로 간주합니다.
        """)

    ic_df = results.get("ic_df", pd.DataFrame())
    if ic_df.empty:
        st.info("IC 데이터가 부족합니다. 백테스트 기간을 늘려주세요.")
        return

    ic_df = ic_df.copy()
    ic_df["date"] = pd.to_datetime(ic_df["date"])
    ic_df = ic_df.sort_values("date")

    ic_mean = ic_df["IC"].mean()
    ic_std  = ic_df["IC"].std()
    ic_ir   = ic_mean / ic_std if ic_std > 0 else 0
    pos_rate = (ic_df["IC"] > 0).mean()

    def _mcol(v): return "pos" if v > 0.03 else ("neg" if v < 0 else "neu")

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "평균 IC",    ic_mean),
        (c2, "IC IR",     ic_ir),
        (c3, "IC 표준편차", ic_std),
        (c4, "양(+)IC 비율", pos_rate),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value {_mcol(val)}">{val:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # IC 막대 + 누적 IC
    colors = ["#00c853" if x > 0 else "#ff1744" for x in ic_df["IC"]]
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=ic_df["date"], y=ic_df["IC"], marker_color=colors,
        hovertemplate="%{x|%Y-%m}<br>IC: %{y:.4f}<extra></extra>",
    ))
    fig1.add_hline(y=0, line_color="white", opacity=0.4, line_width=1)
    fig1.add_hline(y=ic_mean, line_dash="dash", line_color="#7c4dff", line_width=2,
                   annotation_text=f"평균 {ic_mean:.3f}", annotation_position="top right")
    fig1.update_layout(**PLOT_CFG, height=280, title="리밸런싱별 IC")
    fig1.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig1.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ic_df["date"], y=ic_df["IC"].cumsum(),
        fill="tozeroy", line=dict(color="#00c9a7", width=2),
        fillcolor="rgba(0,201,167,0.15)",
        hovertemplate="%{x|%Y-%m}<br>누적 IC: %{y:.3f}<extra></extra>",
    ))
    fig2.update_layout(**PLOT_CFG, height=280, title="누적 IC")
    fig2.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig2.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        st.plotly_chart(fig2, use_container_width=True)

    # IC 분포 히스토그램
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=ic_df["IC"], nbinsx=20, marker_color="#7c4dff", opacity=0.8,
    ))
    fig3.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    fig3.add_vline(x=ic_mean, line_dash="dash", line_color="#00c9a7",
                   annotation_text=f"평균: {ic_mean:.3f}")
    fig3.update_layout(**PLOT_CFG, height=240, title="IC 분포 히스토그램")
    st.plotly_chart(fig3, use_container_width=True)

    # IC 테이블
    st.markdown('<div class="section-hdr">📋 리밸런싱별 IC 상세</div>', unsafe_allow_html=True)
    disp = ic_df.copy()
    disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
    disp["IC"]   = disp["IC"].round(4)
    disp["평가"] = disp["IC"].apply(
        lambda x: "✅ 좋음(>0.05)" if x > 0.05 else ("⚠️ 보통(>0)" if x > 0 else "❌ 음수"))
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 ── 리밸런싱 히스토리
# ═══════════════════════════════════════════════════════════

def tab_history(results: dict):
    hist = results.get("rebal_hist", [])
    if not hist:
        st.info("리밸런싱 히스토리가 없습니다.")
        return

    st.markdown(f"총 **{len(hist)}**회 리밸런싱 수행됨")

    opts = [f"{h['rebalance_date'].strftime('%Y-%m-%d')} → {h['next_date'].strftime('%Y-%m-%d')}"
            for h in hist]

    # session_state로 인덱스 관리 → 탭이 0으로 리셋되는 버그 방지
    if "hist_sel_idx" not in st.session_state or st.session_state.hist_sel_idx >= len(opts):
        st.session_state.hist_sel_idx = len(opts) - 1

    sel = st.selectbox(
        "리밸런싱 기간 선택",
        opts,
        index=st.session_state.hist_sel_idx,
        key="hist_period_select",
    )
    st.session_state.hist_sel_idx = opts.index(sel)
    h = hist[opts.index(sel)]

    ret_c = "pos" if h["port_return"] > 0 else "neg"
    ic_v  = h["ic"]
    ic_c  = "pos" if (not np.isnan(ic_v) and ic_v > 0) else "neg"

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, lbl, val, clr in [
        (c1, "리밸런싱 기준일", h["rebalance_date"].strftime("%Y-%m-%d"), "neu"),
        (c2, "보유 기간",       h["holding_period"], "neu"),
        (c3, "주가 학습 기간",  f"{h['learn_start']} ~", "neu"),
        (c4, "기간 수익률",     f"{h['port_return']:.2%}", ret_c),
        (c5, "IC",             f"{ic_v:.3f}" if not np.isnan(ic_v) else "N/A", ic_c),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value {clr}" style="font-size:0.95rem">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([6, 4])

    with col_l:
        st.markdown('<div class="section-hdr">📋 선정 종목 상세 지표</div>', unsafe_allow_html=True)
        tdf = h.get("ticker_df", pd.DataFrame())
        if not tdf.empty:
            # 핵심 컬럼 우선 표시
            priority = ["ticker", "예측수익률", "실제수익률",
                        "1개월 수익률", "3개월 수익률", "6개월 수익률",
                        "RSI(14)", "P/E 비율", "P/B 비율", "ROE", "영업이익률"]
            disp_cols = [c for c in priority if c in tdf.columns]
            remaining = [c for c in tdf.columns if c not in disp_cols]
            disp = tdf[disp_cols + remaining].copy()
            # 수익률 포맷
            for c in disp.columns:
                if "수익률" in c or c in ["ROE", "ROA", "매출총이익률", "영업이익률"]:
                    disp[c] = disp[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) and isinstance(x, float) else x)
                elif c in ["P/E 비율", "P/B 비율", "EV/EBITDA", "ADX(14)", "RSI(14)"]:
                    disp[c] = disp[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) and isinstance(x, float) else x)
            st.dataframe(disp, use_container_width=True, hide_index=True, height=360,
                         key="hist_ticker_df")

    with col_r:
        st.markdown('<div class="section-hdr">🏆 지표 중요도 TOP 10</div>', unsafe_allow_html=True)
        top10 = h.get("top10_features", [])
        if top10:
            names = [FEAT_NAMES.get(f, f) for f, _ in top10]
            vals  = [v for _, v in top10]
            fig = go.Figure(go.Bar(
                x=vals[::-1], y=names[::-1], orientation="h",
                marker=dict(color=vals[::-1], colorscale="Viridis"),
                hovertemplate="%{y}<br>중요도: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                **PLOT_CFG, height=360,
                margin=dict(l=130, r=10, t=10, b=30),
                xaxis_title="중요도",
            )
            fig.update_yaxes(tickfont=dict(size=10))
            st.plotly_chart(fig, use_container_width=True, key="hist_top10_chart")


# ═══════════════════════════════════════════════════════════
# TAB 4 ── 지표 중요도
# ═══════════════════════════════════════════════════════════

def tab_importance(results: dict):
    fimp = results.get("fimp_df", pd.DataFrame())
    if fimp.empty:
        st.info("지표 중요도 데이터가 없습니다.")
        return

    avg_imp = fimp.mean().sort_values(ascending=False)
    top_n   = st.slider("표시할 지표 수", 5, min(25, len(fimp.columns)), 10, key="importance_top_n")
    top_f   = avg_imp.head(top_n).index.tolist()

    palette = px.colors.qualitative.Plotly

    # 중요도 추이 라인 그래프
    st.markdown('<div class="section-hdr">📈 주요 지표 중요도 추이 (리밸런싱별)</div>',
                unsafe_allow_html=True)
    fig1 = go.Figure()
    for i, f in enumerate(top_f):
        if f in fimp.columns:
            fig1.add_trace(go.Scatter(
                x=fimp.index, y=fimp[f],
                name=FEAT_NAMES.get(f, f),
                line=dict(color=palette[i % len(palette)], width=2),
                hovertemplate=f"{FEAT_NAMES.get(f,f)}<br>%{{x|%Y-%m}}: %{{y:.4f}}<extra></extra>",
            ))
    fig1.update_layout(
        **PLOT_CFG, height=380,
        xaxis_title="날짜", yaxis_title="중요도",
        hovermode="x unified",
        legend=dict(x=1.01, y=1),
    )
    fig1.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig1.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig1, use_container_width=True)

    # 지표 중요도 비중 누적 영역 그래프
    st.markdown('<div class="section-hdr">📊 지표 중요도 비중 추이 (누적 스택)</div>',
                unsafe_allow_html=True)
    prop = fimp[top_f].div(fimp[top_f].sum(axis=1), axis=0)
    fig2 = go.Figure()
    for i, f in enumerate(top_f):
        if f in prop.columns:
            fig2.add_trace(go.Scatter(
                x=prop.index, y=prop[f],
                name=FEAT_NAMES.get(f, f),
                stackgroup="one",
                line=dict(color=palette[i % len(palette)], width=1),
                hovertemplate=f"{FEAT_NAMES.get(f,f)}: %{{y:.1%}}<extra></extra>",
            ))
    fig2.update_layout(
        **PLOT_CFG, height=320,
        yaxis=dict(tickformat=".0%"),
        hovermode="x unified",
        legend=dict(x=1.01, y=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 최근 중요도 순위 테이블
    st.markdown('<div class="section-hdr">📋 최근 지표 중요도 순위</div>', unsafe_allow_html=True)
    latest = fimp.iloc[-1].sort_values(ascending=False)
    tbl = pd.DataFrame({
        "순위":   range(1, len(latest) + 1),
        "지표":   [FEAT_NAMES.get(k, k) for k in latest.index],
        "그룹":   [FEATURE_META.get(k, {}).get("group", "") for k in latest.index],
        "중요도": latest.values.round(4),
        "비중":   (latest / latest.sum()).apply(lambda x: f"{x:.1%}"),
    }).head(20)
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 5 ── 영향력 히트맵
# ═══════════════════════════════════════════════════════════

def tab_heatmap(results: dict):
    fimp = results.get("fimp_df", pd.DataFrame())
    if fimp.empty:
        st.info("히트맵 데이터가 없습니다.")
        return

    st.markdown('<div class="section-hdr">🗺️ 지표 영향력 타임라인 히트맵</div>',
                unsafe_allow_html=True)

    n_feats = st.slider("표시할 지표 수", 5, min(len(fimp.columns), 50),
                        min(30, len(fimp.columns)), key="heatmap_n_feats")
    avg_imp = fimp.mean().sort_values(ascending=False)
    top_f   = avg_imp.head(n_feats).index.tolist()

    heat = fimp[top_f].T
    ylbls = [FEAT_NAMES.get(f, f) for f in heat.index]
    xlbls = [d.strftime("%Y-%m") for d in heat.columns]

    fig = go.Figure(go.Heatmap(
        z=heat.values, x=xlbls, y=ylbls,
        colorscale="RdYlBu_r",
        hovertemplate="날짜: %{x}<br>지표: %{y}<br>중요도: %{z:.4f}<extra></extra>",
        colorbar=dict(title="중요도"),
    ))
    fig.update_layout(
        **PLOT_CFG,
        height=max(400, 18 * n_feats + 100),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=50, t=30, b=100),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 상관관계 히트맵
    st.markdown('<div class="section-hdr">🔗 주요 지표 중요도 상관관계</div>', unsafe_allow_html=True)
    top10 = avg_imp.head(10).index.tolist()
    corr  = fimp[top10].corr()
    clbls = [FEAT_NAMES.get(f, f) for f in corr.index]

    fig2 = go.Figure(go.Heatmap(
        z=corr.values, x=clbls, y=clbls,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        colorbar=dict(title="상관계수"),
        hovertemplate="%{y} vs %{x}<br>%{z:.3f}<extra></extra>",
    ))
    fig2.update_layout(
        **PLOT_CFG, height=420,
        margin=dict(l=160, r=50, t=30, b=160),
    )
    fig2.update_xaxes(tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 6 ── 실시간 AI 추천
# ═══════════════════════════════════════════════════════════

def tab_realtime(price_data: dict, fund_map: dict, tech_map: dict,
                 results: dict, n_stocks: int):
    st.markdown('<div class="section-hdr">🔴 실시간 AI 추천 종목 (최신 데이터 기준)</div>',
                unsafe_allow_html=True)

    fimp = results.get("fimp_df", pd.DataFrame())
    if fimp.empty:
        st.info("백테스트를 먼저 실행하세요.")
        return

    today = pd.Timestamp(datetime.today().date())

    with st.spinner("최신 지표 계산 중..."):
        cur_snap = build_snapshot_df(list(price_data.keys()), tech_map, fund_map, today)

    if cur_snap.empty:
        st.warning("현재 지표 데이터를 계산할 수 없습니다.")
        return

    # 최근 학습된 중요도를 가중치로 복합 점수 계산
    latest_imp = fimp.iloc[-1]
    avail_cols = [c for c in FEAT_COLS if c in cur_snap.columns]

    # 점수 계산: 각 피처를 백분위 순위로 변환 후 중요도 가중합
    score_df = cur_snap[avail_cols].copy()

    # 방향성 처리 (낮을수록 좋은 지표는 역방향)
    reverse_cols = {"P_E", "P_B", "P_S", "EV_EBITDA", "P_FCF",
                    "Volatility_30d", "Volatility_90d", "Debt_Equity"}

    ranked = pd.DataFrame(index=score_df.index)
    for col in avail_cols:
        asc = col not in reverse_cols
        ranked[col] = score_df[col].rank(pct=True, ascending=asc, na_option="bottom")

    weights = latest_imp.reindex(avail_cols).fillna(0)
    composite = ranked.mul(weights, axis=1).sum(axis=1)
    composite = composite / composite.max()

    top_recs = composite.nlargest(n_stocks)
    all_recs  = composite.sort_values(ascending=False)  # 전체 종목 정렬

    # ── 전체 종목 상세 테이블 ──────────────────────────────
    st.markdown(f"**{today.strftime('%Y-%m-%d')} 기준 전체 분석 종목 AI 점수 순위 ({len(all_recs)}개)**")
    rec_rows = []
    for rank, (t, score) in enumerate(all_recs.items(), 1):
        row = {"순위": rank, "티커": t, "AI 점수": round(score, 3),
               "추천": "★ 추천" if t in top_recs.index else ""}
        for col, fmt in [
            ("Mom_1m",  "pct"), ("Mom_3m",  "pct"), ("Mom_6m",  "pct"),
            ("RSI_14",  "f1"),  ("P_E",     "f1"),  ("P_B",     "f2"),
            ("ROE",     "pct"), ("Op_Margin","pct"), ("Volatility_30d","pct"),
            ("Div_Yield","pct"),("EV_EBITDA","f1"),
        ]:
            if col in cur_snap.columns and t in cur_snap.index:
                v = cur_snap.loc[t, col]
                if pd.isna(v):
                    row[FEAT_NAMES.get(col, col)] = "N/A"
                elif fmt == "pct":
                    row[FEAT_NAMES.get(col, col)] = f"{v:.2%}"
                elif fmt == "f1":
                    row[FEAT_NAMES.get(col, col)] = f"{v:.1f}"
                else:
                    row[FEAT_NAMES.get(col, col)] = f"{v:.2f}"
        rec_rows.append(row)

    rec_df = pd.DataFrame(rec_rows)
    st.dataframe(rec_df, use_container_width=True, hide_index=True, height=420)

    # 점수 막대 그래프
    fig = go.Figure(go.Bar(
        x=top_recs.index, y=top_recs.values,
        marker=dict(color=top_recs.values, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="AI 점수")),
        hovertemplate="%{x}<br>AI 점수: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_CFG, height=320,
        title=f"AI 추천 종목 점수 ({today.strftime('%Y-%m-%d')})",
        xaxis_title="티커", yaxis_title="AI 점수",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 현재 모델 지표 중요도
    st.markdown('<div class="section-hdr">🔑 현재 AI 모델 지표 중요도 (최신 리밸런싱 기준)</div>',
                unsafe_allow_html=True)
    top_imp = latest_imp.sort_values(ascending=False).head(15)
    fig2 = go.Figure(go.Bar(
        x=[FEAT_NAMES.get(f, f) for f in top_imp.index],
        y=top_imp.values,
        marker=dict(color=top_imp.values, colorscale="Plasma"),
        hovertemplate="%{x}<br>중요도: %{y:.4f}<extra></extra>",
    ))
    fig2.update_layout(
        **PLOT_CFG, height=300, xaxis_tickangle=-45,
        title="현재 지표 중요도 TOP 15",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 추천 종목 레이더 차트
    if len(top_recs) >= 3:
        st.markdown('<div class="section-hdr">🕸️ TOP 5 종목 지표 레이더</div>', unsafe_allow_html=True)
        radar_feats = ["Mom_3m", "RSI_14", "ROE", "Op_Margin", "Mom_6m",
                       "Volatility_30d", "P_B", "Rev_Growth"]
        radar_feats = [f for f in radar_feats if f in cur_snap.columns]

        fig3 = go.Figure()
        for t in top_recs.index[:5]:
            if t not in cur_snap.index:
                continue
            vals = []
            for f in radar_feats:
                v = cur_snap.loc[t, f]
                # 백분위 변환
                rk = cur_snap[f].rank(pct=True)
                vals.append(float(rk.get(t, 0.5)))
            vals.append(vals[0])
            labels = [FEAT_NAMES.get(f, f) for f in radar_feats] + [FEAT_NAMES.get(radar_feats[0], radar_feats[0])]
            fig3.add_trace(go.Scatterpolar(
                r=vals, theta=labels, fill="toself", name=t,
            ))
        fig3.update_layout(
            **PLOT_CFG, height=400,
            polar=dict(
                bgcolor="rgba(20,25,40,0.8)",
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TOP BAR (모바일 대응 상단 설정 바)
# ═══════════════════════════════════════════════════════════

def render_topbar(sp500_df: pd.DataFrame, all_sectors: list) -> dict:
    """사이드바 대신 페이지 상단 가로 배치 설정 UI."""

    with st.expander("⚙️ 백테스트 설정 펼치기 / 접기", expanded=True):

        # ── Row 1: 섹터 선택 ───────────────────────────────
        st.markdown("**📂 분석 섹터 선택** (복수 선택 가능)")
        default_sectors = ["Information Technology"] if "Information Technology" in all_sectors else all_sectors[:1]
        sel_sectors = st.multiselect(
            "GICS 섹터",
            all_sectors,
            default=default_sectors,
            label_visibility="collapsed",
            help="포함할 GICS 섹터를 선택하세요",
        )
        if not sel_sectors:
            sel_sectors = all_sectors[:2]

        universe = sp500_df[sp500_df["sector"].isin(sel_sectors)]["ticker"].tolist()
        st.caption(f"선택된 유니버스: **{len(universe)}**개 종목")

        st.markdown("---")

        # ── Row 2: 핵심 파라미터 + 실행 버튼 ──────────────
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
        with c1:
            rebal_m = st.slider(
                "리밸런싱 기간 (개월)", 1, 12, 1, 1,
                help="포트폴리오 리밸런싱 주기",
            )
        with c2:
            rolling_w = st.slider(
                "롤링 학습 윈도우 (기간 수)", 2, 12, 4, 1,
                help="모델 학습에 사용할 이전 리밸런싱 기간 수",
            )
        with c3:
            n_stocks = st.slider("투자 종목 수", 1, 20, 5, 1)
        with c4:
            tc_pct = st.number_input(
                "거래비용 (%)", min_value=0.0, max_value=5.0,
                value=0.3, step=0.05, format="%.2f",
                help="왕복 총 거래비용 (수수료 + 슬리피지)",
            )
        with c5:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 실행", type="primary", use_container_width=True)

        # ── Row 3: 날짜 설정 ────────────────────────────────
        # 유의미한 백테스트: rolling_win + min_test_periods 개 리밸런싱 날짜 필요
        # 날짜 수 = rolling_w + MIN_TEST + 1 (마지막 날짜는 forward return 종료점)
        MIN_TEST    = 5   # 최소 테스트(예측) 횟수
        auto_months = (rolling_w + MIN_TEST) * rebal_m + 12  # +12개월 지표 warm-up
        auto_end    = datetime.today()
        auto_start  = auto_end - timedelta(days=int(auto_months * 30.5))

        use_custom = st.checkbox("날짜 직접 입력", value=False)
        if use_custom:
            dc1, dc2 = st.columns(2)
            sd = dc1.date_input("시작일", value=auto_start.date())
            ed = dc2.date_input("종료일", value=auto_end.date(), min_value=sd)
        else:
            sd = auto_start.date()
            ed = auto_end.date()
            st.info(
                f"📅 자동 설정: **{sd}** ~ **{ed}** "
                f"(약 {auto_months}개월 | 최소 {MIN_TEST}회 테스트 보장 | "
                f"리밸런싱 1회 = {rebal_m}개월)"
            )

    return {
        "sectors":     sel_sectors,
        "universe":    universe,
        "rebal_m":     rebal_m,
        "rolling_w":   rolling_w,
        "start":       datetime(sd.year, sd.month, sd.day),
        "end":         datetime(ed.year, ed.month, ed.day),
        "n_stocks":    n_stocks,
        "tc_pct":      tc_pct,
        "run":         run_btn,
        "min_test":    MIN_TEST,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    st.markdown('<div class="main-title">📊 AI Quant Lab 2.0</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">AI 기반 퀀트 백테스팅 & 실시간 종목 추천 플랫폼 · '
        'Random Forest · IC 분석 · 50+ 지표</div>',
        unsafe_allow_html=True,
    )

    # S&P 500 목록 로드
    with st.spinner("S&P 500 종목 목록 로드 중..."):
        sp500_df, all_sectors = get_sp500_info()

    cfg = render_topbar(sp500_df, all_sectors)

    # ── 세션 상태 초기화 ──────────────────────────────────
    for k in ["results", "benchmarks", "price_data", "fund_map", "tech_map", "cfg"]:
        if k not in st.session_state:
            st.session_state[k] = None

    # ── 환영 화면 ─────────────────────────────────────────
    if not cfg["run"] and st.session_state.results is None:
        st.info("위 설정 바에서 원하는 조건을 선택한 후 **🚀 실행** 버튼을 누르세요.")
        c1, c2, c3, c4 = st.columns(4)
        for col, icon, title, items in [
            (c1, "🧠", "AI 모델", ["Random Forest 예측", "롤링 윈도우 학습", "피처 중요도 추출"]),
            (c2, "📐", "지표 (50+)", ["기술지표 25종", "펀더멘털 25종", "복합 지표"]),
            (c3, "📊", "분석 뷰", ["IC 분석", "리밸런싱 히스토리", "지표 히트맵"]),
            (c4, "🔴", "실시간", ["AI 추천 종목", "레이더 차트", "복합 점수 랭킹"]),
        ]:
            col.markdown(
                f"**{icon} {title}**\n" + "".join(f"\n- {i}" for i in items)
            )
        return

    # 위젯 트리 안정화: st.tabs()가 항상 동일한 위치(위젯 카운터)에 렌더링되도록
    # if cfg["run"] 안에 두면 백테스트 실행 시와 결과 조회 시 st.tabs()의
    # 자동 키가 달라져 탭 상태가 초기화되는 버그가 발생함
    _status_slot = st.empty()
    _prog_slot   = st.empty()

    # ── 백테스트 실행 ─────────────────────────────────────
    if cfg["run"]:
        universe = cfg["universe"]
        if not universe:
            _status_slot.error("섹터를 선택해주세요.")
            return

        _prog_slot.progress(0)

        def update_prog(val, msg):
            _prog_slot.progress(val, msg)
            _status_slot.info(msg)

        # 1. 가격 데이터
        update_prog(0.03, f"📡 {len(universe)}개 종목 가격 데이터 다운로드 중...")
        data_start = cfg["start"] - timedelta(days=400)  # 지표 warm-up
        price_data = download_price_data(
            tuple(universe),
            data_start.strftime("%Y-%m-%d"),
            cfg["end"].strftime("%Y-%m-%d"),
        )
        if not price_data:
            st.error("가격 데이터를 불러올 수 없습니다.")
            return

        available = list(price_data.keys())
        update_prog(0.12, f"✅ {len(available)}개 종목 가격 데이터 수신. 펀더멘털 로드 중...")

        # 2. 펀더멘털 데이터
        fund_map = get_fundamental_yf(tuple(available))
        update_prog(0.20, "🔧 기술지표 사전 계산 중...")

        # 3. 기술지표 사전 계산
        tech_map = {}
        for t, ohlcv in price_data.items():
            try:
                tech_map[t] = calc_all_technical(ohlcv)
            except Exception:
                pass
        update_prog(0.28, f"📈 기술지표 계산 완료 ({len(tech_map)}종목). 백테스트 시작...")

        # 4. 리밸런싱 날짜 생성
        # 유의미한 백테스트 조건:
        #   전체 날짜 수 >= rolling_w(학습) + MIN_TEST(예측) + 1(마지막 forward return 끝점)
        MIN_TEST    = cfg.get("min_test", 5)
        rebal_dates = generate_rebalance_dates(cfg["start"], cfg["end"], cfg["rebal_m"])
        n_needed    = cfg["rolling_w"] + MIN_TEST  # 최소 테스트 횟수 = MIN_TEST
        if len(rebal_dates) < n_needed + 1:
            needed_months = (n_needed + 1) * cfg["rebal_m"] + 12
            st.error(
                f"백테스트 기간이 부족합니다. "
                f"최소 {n_needed + 1}개 리밸런싱 날짜 필요 (현재 {len(rebal_dates)}개). "
                f"학습 {cfg['rolling_w']}회 + 테스트 {MIN_TEST}회 + 지표 warm-up 1년 = "
                f"약 {needed_months}개월 이상의 기간이 필요합니다. "
                f"날짜를 자동 설정으로 변경하거나 시작일을 앞당겨 주세요."
            )
            return

        # 5. 백테스트
        results = run_backtest(
            price_data=price_data,
            fund_map=fund_map,
            tech_map=tech_map,
            rebal_dates=rebal_dates,
            n_stocks=cfg["n_stocks"],
            tc_pct=cfg["tc_pct"],
            rolling_win=cfg["rolling_w"],
            progress=update_prog,
        )

        # 6. 벤치마크 데이터
        benchmarks = get_benchmark_prices(
            cfg["start"].strftime("%Y-%m-%d"),
            cfg["end"].strftime("%Y-%m-%d"),
        )

        # 저장
        st.session_state.results    = results
        st.session_state.benchmarks = benchmarks
        st.session_state.price_data = price_data
        st.session_state.fund_map   = fund_map
        st.session_state.tech_map   = tech_map
        st.session_state.cfg        = cfg

        _prog_slot.progress(1.0, "✅ 완료!")
        time.sleep(0.4)
        _prog_slot.empty()
        _status_slot.success(
            f"✅ 백테스트 완료! "
            f"{len(rebal_dates)}회 리밸런싱 | "
            f"{len(results['rebal_hist'])}회 학습 | "
            f"{len(available)}개 종목 분석"
        )

    # ── 결과 표시 ─────────────────────────────────────────
    if st.session_state.results is not None:
        results    = st.session_state.results
        benchmarks = st.session_state.benchmarks or {}
        price_data = st.session_state.price_data or {}
        fund_map   = st.session_state.fund_map   or {}
        tech_map   = st.session_state.tech_map   or {}
        saved_cfg  = st.session_state.cfg        or cfg

        tabs = st.tabs([
            "📈 성과 비교",
            "🎯 IC 분석",
            "📋 리밸런싱 히스토리",
            "🔍 지표 중요도",
            "🗺️ 영향력 히트맵",
            "🔴 실시간 추천",
        ])

        with tabs[0]:
            tab_performance(results, benchmarks, price_data)
        with tabs[1]:
            tab_ic(results)
        with tabs[2]:
            tab_history(results)
        with tabs[3]:
            tab_importance(results)
        with tabs[4]:
            tab_heatmap(results)
        with tabs[5]:
            tab_realtime(price_data, fund_map, tech_map, results,
                         saved_cfg.get("n_stocks", 10))


if __name__ == "__main__":
    main()
