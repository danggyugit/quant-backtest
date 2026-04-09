#!/usr/bin/env python3
"""
claude_backtest_Ver4.2.py
AI 퀀트 백테스팅 & 실시간 종목 추천 플랫폼
Ver3.0~4.0: (이전 변경 이력 생략)
Ver4.2: 백테스트 신뢰성 개선
        ① 윈저화 순서 수정 (결측대체 → 윈저화)
        ② Purged CV 엠바고 (훈련/예측 간 21일 gap으로 데이터 누수 차단)
        ③ RF 하이퍼파라미터 자동 튜닝 (GridSearchCV)
        ④ 턴오버 버퍼존 옵션 (보유 종목 가산점으로 불필요한 교체 방지)
        ⑤ 역변동성 가중 옵션 (변동성 역수 비중 → MDD 축소)
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
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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
    page_title="AI Quant Lab 4.2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── 전체 배경 흰색 ── */
    .stApp { background: #ffffff; }
    section[data-testid="stSidebar"] { display: none; }

    /* ── Streamlit 기본 UI 숨김 (Share·GitHub·Manage app 등) ── */
    #MainMenu { display: none; }
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }
    div[data-testid="manage-app-button"] { display: none; }
    .stDeployButton { display: none; }
    div[data-testid="stToolbar"] { display: none; }
    div[data-testid="stStatusWidget"] { display: none; }

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
REPORT_LAG         = 45  # 분기 재무보고 지연일
ANNUAL_REPORT_LAG  = 75  # 연간 재무보고 지연일 (60~90일 중간값)

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
    "Momentum_Custom":   {"name": "모멘텀대체(12-1→6-1)", "group": "모멘텀"},
    "RS_Market_1m":      {"name": "시장대비 상대강도(1M)", "group": "모멘텀"},
    "RS_Market_3m":      {"name": "시장대비 상대강도(3M)", "group": "모멘텀"},
    "RS_Market_6m":      {"name": "시장대비 상대강도(6M)", "group": "모멘텀"},
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
    "Beta_60d":          {"name": "베타(60일)",         "group": "리스크"},
    "Beta_120d":         {"name": "베타(120일)",        "group": "리스크"},
    "Rolling_Sharpe_60d":{"name": "롤링 Sharpe(60일)",  "group": "리스크"},
    "Daily_Win_Rate_60d":{"name": "일별 승률(60일)",    "group": "리스크"},
    "Vol_Ratio_30_90":   {"name": "변동성 비율(30/90)", "group": "리스크"},
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
    # ── Cross-Sectional 백분위 (Ver3.9) ────────────────────
    "Mom_3m_Pct":        {"name": "3개월모멘텀 백분위",  "group": "CS정규화"},
    "RS_Market_3m_Pct":  {"name": "상대강도3M 백분위",   "group": "CS정규화"},
    "P_E_Pct":           {"name": "P/E 백분위",          "group": "CS정규화"},
    "ROE_Pct":           {"name": "ROE 백분위",          "group": "CS정규화"},
    "Volatility_30d_Pct":{"name": "변동성30d 백분위",    "group": "CS정규화"},
    # ── Regime (시장 상태) ─────────────────────────────────
    "Market_Regime":     {"name": "시장 레짐(SPY/SMA200)","group": "레짐"},
    "VIX_Level":         {"name": "VIX 수준",            "group": "레짐"},
    # ── Sector Relative Strength ───────────────────────────
    "RS_Sector_1m":      {"name": "섹터대비 상대강도(1M)","group": "섹터상대"},
    "RS_Sector_3m":      {"name": "섹터대비 상대강도(3M)","group": "섹터상대"},
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
def get_sp1500_info() -> tuple:
    """Wikipedia에서 S&P 500 / 400 / 600 종목 목록을 합산하여 S&P 1500 구성."""
    headers = {"User-Agent": "Mozilla/5.0"}

    # (url, cap_tier) 순서대로 시도
    index_sources = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",  "Large Cap"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",  "Mid Cap"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",  "Small Cap"),
    ]

    frames = []
    for url, cap_tier in index_sources:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            tables = pd.read_html(StringIO(resp.text))
            # Wikipedia 표 구조: 첫 번째 표에 Symbol / GICS Sector / Security 컬럼
            raw = tables[0]
            # 컬럼명이 다를 수 있으므로 유연하게 매핑
            col_map = {}
            for c in raw.columns:
                lc = str(c).lower()
                if "symbol" in lc or "ticker" in lc:
                    col_map[c] = "ticker"
                elif "gics sector" in lc or "sector" in lc:
                    col_map[c] = "sector"
                elif "security" in lc or "company" in lc or "name" in lc:
                    col_map[c] = "name"
            raw = raw.rename(columns=col_map)
            needed = [c for c in ["ticker", "name", "sector"] if c in raw.columns]
            raw = raw[needed].copy()
            raw["ticker"] = raw["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
            raw["cap_tier"] = cap_tier
            frames.append(raw)
        except Exception as e:
            st.warning(f"{cap_tier} ({url}) 조회 실패: {e}")

    if not frames:
        st.warning("S&P 1500 목록 조회 실패. 내장 리스트 사용.")
        return _fallback_sp500(), _fallback_sectors()

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="ticker")
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    df["sector"] = df["sector"].fillna("Unknown")
    sectors = sorted(df["sector"].dropna().unique().tolist())
    return df[["ticker", "name", "sector", "cap_tier"]], sectors


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


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_changes() -> pd.DataFrame:
    """Wikipedia S&P 500 변경 이력 파싱 (편입/퇴출 날짜·티커).
    반환: DataFrame[date, added_ticker, removed_ticker]
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        tables = pd.read_html(StringIO(resp.text))
        if len(tables) < 2:
            return pd.DataFrame()
        raw = tables[1]  # 두 번째 표 = 변경 이력
    except Exception:
        return pd.DataFrame()

    # 컬럼 평탄화 (MultiIndex 가능)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(str(c) for c in col).strip() for col in raw.columns]

    # 날짜·티커 컬럼 탐색
    date_col = added_col = removed_col = None
    for c in raw.columns:
        lc = c.lower()
        if "date" in lc and date_col is None:
            date_col = c
        elif "added" in lc and "ticker" in lc and added_col is None:
            added_col = c
        elif "added" in lc and "symbol" in lc and added_col is None:
            added_col = c
        elif "removed" in lc and "ticker" in lc and removed_col is None:
            removed_col = c
        elif "removed" in lc and "symbol" in lc and removed_col is None:
            removed_col = c

    if date_col is None:
        return pd.DataFrame()

    rows = []
    for _, r in raw.iterrows():
        try:
            d = pd.to_datetime(r[date_col], format="mixed")
        except Exception:
            continue
        at = str(r.get(added_col, "")).strip() if added_col else ""
        rt = str(r.get(removed_col, "")).strip() if removed_col else ""
        at = at.replace(".", "-") if at and at != "nan" else ""
        rt = rt.replace(".", "-") if rt and rt != "nan" else ""
        rows.append({"date": d, "added_ticker": at, "removed_ticker": rt})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def reconstruct_sp500_at_date(
    current_members: list,
    changes_df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> list:
    """현재 S&P 500 구성 + 변경 이력 → target_date 시점의 실제 구성 복원.
    알고리즘: 현재 목록에서 출발, target_date 이후의 변경을 역산 적용.
    """
    if changes_df.empty:
        return current_members

    members = set(current_members)
    # target_date 이후 변경만 (최신→과거 순)
    future = changes_df[changes_df["date"] > target_date].sort_values("date", ascending=False)

    for _, row in future.iterrows():
        added   = row["added_ticker"]
        removed = row["removed_ticker"]
        # 이후에 편입된 종목 → 해당 시점엔 없었으므로 제거
        if added and added in members:
            members.discard(added)
        # 이후에 퇴출된 종목 → 해당 시점엔 있었으므로 추가
        if removed:
            members.add(removed)

    return sorted(members)


@st.cache_data(ttl=86400, show_spinner=False)
def get_riskfree_rate(start: str, end: str) -> float:
    """백테스트 기간의 평균 무위험수익률 조회 (미국 3개월 T-bill ^IRX).
    ^IRX는 연율화된 수익률을 % 단위로 제공 (예: 5.0 → 5.0%).
    조회 실패 시 기간에 따른 합리적 기본값 반환.
    """
    try:
        df = yf.download("^IRX", start=start, end=end,
                         auto_adjust=True, progress=False)
        if not df.empty:
            avg_pct = float(df["Close"].squeeze().mean())
            return avg_pct / 100  # 소수로 변환 (0.05 = 5%)
    except Exception:
        pass

    # 조회 실패 시: 기간 중간 연도 기준 합리적 기본값
    try:
        mid_year = int(start[:4])
    except Exception:
        mid_year = 2022

    # 연도별 대략적인 미국 기준금리
    fallback = {
        2018: 0.022, 2019: 0.021, 2020: 0.005,
        2021: 0.003, 2022: 0.030, 2023: 0.051,
        2024: 0.052, 2025: 0.045,
    }
    return fallback.get(mid_year, 0.03)


@st.cache_data(ttl=86400, show_spinner=False)
def get_pit_financials(tickers: tuple) -> dict:
    """분기별 재무제표(손익계산서 + 대차대조표) 수집 — PIT 근사용.
    반환: {ticker: {"income": pd.DataFrame, "balance": pd.DataFrame}}
    columns = 보고 분기 날짜, rows = 재무 항목
    주의: yfinance 분기 데이터는 최근 4~5분기만 제공하는 경우 있음.
    """
    out = {}
    workers = 1 if IS_CLOUD else 4

    def _safe(df):
        return df if (df is not None and not df.empty) else pd.DataFrame()

    def _fetch(t):
        for attempt in range(3):
            try:
                tk = yf.Ticker(t)
                return t, {
                    "income":          _safe(tk.quarterly_financials),
                    "balance":         _safe(tk.quarterly_balance_sheet),
                    "cashflow":        _safe(tk.quarterly_cashflow),
                    "annual_income":   _safe(tk.financials),
                    "annual_balance":  _safe(tk.balance_sheet),
                    "annual_cashflow": _safe(tk.cashflow),
                }
            except Exception:
                if attempt < 2:
                    time.sleep(1.0 + attempt)
        return t, {k: pd.DataFrame() for k in
                   ["income", "balance", "cashflow",
                    "annual_income", "annual_balance", "annual_cashflow"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(futs):
            t, data = fut.result()
            out[t] = data
            if CLOUD_DELAY:
                time.sleep(CLOUD_DELAY)
    return out


# ═══════════════════════════════════════════════════════════
# TECHNICAL INDICATOR CALCULATORS
# ═══════════════════════════════════════════════════════════

def _rsi(close: pd.Series, n=14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()   # Wilder's EMA
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
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


def _mfi(high, low, close, volume, n=14) -> pd.Series:
    tp   = (high + low + close) / 3
    rmf  = tp * volume
    pos  = rmf.where(tp > tp.shift(1), 0.0)
    neg  = rmf.where(tp < tp.shift(1), 0.0)
    mfr  = pos.rolling(n).sum() / neg.rolling(n).sum().replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


def calc_all_technical(ohlcv: pd.DataFrame, spy_close: pd.Series = None) -> pd.DataFrame:
    """단일 종목의 전체 기술지표 DataFrame 사전 계산 (전체 기간).
    spy_close: SPY 종가 시리즈 (상대강도·베타 계산용, 없으면 해당 피처 NaN).
    """
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
    df["Mom_12_1"]       = c.shift(21) / c.shift(252) - 1
    df["Mom_6_1"]        = c.shift(21) / c.shift(126) - 1
    df["Momentum_Custom"] = df["Mom_12_1"].fillna(df["Mom_6_1"])

    # ── 상대강도 (종목 수익률 - SPY 수익률) ───────────────
    if spy_close is not None:
        spy_a = spy_close.reindex(c.index, method="ffill")
        spy_1m = spy_a.pct_change(21)
        spy_3m = spy_a.pct_change(63)
        spy_6m = spy_a.pct_change(126)
        df["RS_Market_1m"] = df["Mom_1m"] - spy_1m
        df["RS_Market_3m"] = df["Mom_3m"] - spy_3m
        df["RS_Market_6m"] = df["Mom_6m"] - spy_6m
    else:
        df["RS_Market_1m"] = np.nan
        df["RS_Market_3m"] = np.nan
        df["RS_Market_6m"] = np.nan

    # Trend (WilliamsR_14만 제거 — Stoch_K와 수학적 동치)
    df["RSI_14"]    = _rsi(c)
    df["MACD_Hist"] = macd_h / c
    df["Price_SMA20"]  = c / sma20  - 1
    df["Price_SMA50"]  = c / sma50  - 1
    df["Price_SMA200"] = c / sma200 - 1
    df["SMA50_SMA200"] = sma50 / sma200 - 1
    df["ADX_14"]       = _adx(h, l, c)
    df["Stoch_K"]      = _stoch_k(h, l, c)
    df["CCI_20"]       = _cci(h, l, c)
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

    # ── 베타 (종목-시장 공분산/시장분산) ──────────────────
    if spy_close is not None:
        spy_logr = np.log(spy_a / spy_a.shift(1))
        for win, name in [(60, "Beta_60d"), (120, "Beta_120d")]:
            cov = logr.rolling(win).cov(spy_logr)
            var = spy_logr.rolling(win).var()
            df[name] = cov / var.replace(0, np.nan)
    else:
        df["Beta_60d"]  = np.nan
        df["Beta_120d"] = np.nan

    # ── 롤링 Sharpe (60일) ───────────────────────────────
    df["Rolling_Sharpe_60d"] = (logr.rolling(60).mean() / logr.rolling(60).std().replace(0, np.nan)) * np.sqrt(252)

    # ── 일별 승률 (최근 60일 중 양수 수익률 비율) ─────────
    df["Daily_Win_Rate_60d"] = logr.apply(lambda x: 1 if x > 0 else 0).rolling(60).mean()

    # ── 변동성 비율 (30d / 90d) — 확대/수축 추세 ─────────
    df["Vol_Ratio_30_90"] = df["Volatility_30d"] / df["Volatility_90d"].replace(0, np.nan)

    return df


# ═══════════════════════════════════════════════════════════
# PIT (POINT-IN-TIME) FINANCIALS
# ═══════════════════════════════════════════════════════════

def _to_ts(c):
    try:
        return pd.Timestamp(c)
    except Exception:
        return pd.NaT


def _safe_get(df, col, row):
    """안전한 셀 읽기."""
    try:
        v = df.loc[row, col]
        return float(v) if (v is not None and pd.notna(v)) else np.nan
    except Exception:
        return np.nan


def _safe_get_multi(df, col, row_names):
    """여러 행 이름을 순서대로 시도하여 값 추출."""
    for rn in row_names:
        v = _safe_get(df, col, rn)
        if not np.isnan(v):
            return v
    return np.nan


def _avail_cols(df, cutoff):
    """cutoff 이전에 공시된 컬럼(분기/연도) 목록 (최신순)."""
    return sorted(
        [c for c in df.columns if _to_ts(c) is not pd.NaT and _to_ts(c) <= cutoff],
        key=_to_ts, reverse=True
    )


def _ttm_sum(df, cols, row_name):
    """최근 4분기(또는 가용 분기) 합산. 부족하면 연율화."""
    use = cols[:4]
    vals = [_safe_get(df, c, row_name) for c in use]
    valid = [v for v in vals if not np.isnan(v)]
    if not valid:
        return np.nan
    return sum(valid) if len(valid) == 4 else sum(valid) * 4 / len(valid)


def _prev_ttm_sum(df, cols, row_name):
    """4~7분기 전 합산 (전년 동기 TTM, 성장률 계산용)."""
    use = cols[4:8]
    if not use:
        return np.nan
    vals = [_safe_get(df, c, row_name) for c in use]
    valid = [v for v in vals if not np.isnan(v)]
    if not valid:
        return np.nan
    return sum(valid) * 4 / len(valid)


def _get_pit_values(pit_data: dict, date: pd.Timestamp) -> dict:
    """특정 날짜 기준 사용 가능한 TTM 재무 값 추출 (Ver3.6 확장).
    우선순위: 분기 TTM → 연간 폴백.
    분기: REPORT_LAG(45일), 연간: ANNUAL_REPORT_LAG(75일).
    """
    income   = pit_data.get("income",   pd.DataFrame())
    balance  = pit_data.get("balance",  pd.DataFrame())
    cashflow = pit_data.get("cashflow", pd.DataFrame())

    cutoff = date - pd.Timedelta(days=REPORT_LAG)

    # ── 분기 손익계산서 ──────────────────────────────────
    avail_inc = _avail_cols(income, cutoff) if not income.empty else []

    ttm_rev = ttm_gross = ttm_op = ttm_net = np.nan
    ttm_ebit = ttm_ebitda = ttm_interest = ttm_eps = np.nan
    prev_rev = prev_net = prev_eps = np.nan

    if avail_inc:
        ttm_rev      = _ttm_sum(income, avail_inc, "Total Revenue")
        ttm_gross    = _ttm_sum(income, avail_inc, "Gross Profit")
        ttm_op       = _ttm_sum(income, avail_inc, "Operating Income")
        ttm_net      = _ttm_sum(income, avail_inc, "Net Income")
        ttm_ebit     = _ttm_sum(income, avail_inc, "EBIT")
        ttm_ebitda   = _ttm_sum(income, avail_inc, "EBITDA")
        ttm_interest = _ttm_sum(income, avail_inc, "Interest Expense")
        if np.isnan(ttm_interest):
            ttm_interest = _ttm_sum(income, avail_inc, "Interest Expense Non Operating")
        ttm_eps      = _ttm_sum(income, avail_inc, "Basic EPS")
        # 전년 동기
        prev_rev = _prev_ttm_sum(income, avail_inc, "Total Revenue")
        prev_net = _prev_ttm_sum(income, avail_inc, "Net Income")
        prev_eps = _prev_ttm_sum(income, avail_inc, "Basic EPS")

    # ── 분기 대차대조표 ──────────────────────────────────
    ta = eq = cur_a = cur_l = total_debt = cash = shares_out = np.nan
    if not balance.empty:
        avail_bal = _avail_cols(balance, cutoff)
        if avail_bal:
            bc = avail_bal[0]
            ta = _safe_get(balance, bc, "Total Assets")
            eq = _safe_get_multi(balance, bc, [
                "Stockholders Equity",
                "Total Equity Gross Minority Interest",
                "Common Stock Equity"])
            cur_a = _safe_get(balance, bc, "Current Assets")
            cur_l = _safe_get(balance, bc, "Current Liabilities")
            total_debt = _safe_get_multi(balance, bc, [
                "Total Debt", "Long Term Debt And Capital Lease Obligation"])
            cash = _safe_get_multi(balance, bc, [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments"])
            shares_out = _safe_get_multi(balance, bc, [
                "Ordinary Shares Number", "Share Issued"])

    # ── 분기 현금흐름표 ──────────────────────────────────
    ttm_ocf = ttm_capex = ttm_fcf = np.nan
    if not cashflow.empty:
        avail_cf = _avail_cols(cashflow, cutoff)
        if avail_cf:
            ttm_ocf   = _ttm_sum(cashflow, avail_cf, "Operating Cash Flow")
            ttm_capex = _ttm_sum(cashflow, avail_cf, "Capital Expenditure")
            ttm_fcf   = _ttm_sum(cashflow, avail_cf, "Free Cash Flow")
            if np.isnan(ttm_fcf) and not np.isnan(ttm_ocf) and not np.isnan(ttm_capex):
                ttm_fcf = ttm_ocf + ttm_capex  # CapEx는 보통 음수

    # ── 연간 폴백: NaN이 많으면 연간 데이터로 보완 ────────
    annual_inc = pit_data.get("annual_income",   pd.DataFrame())
    annual_bal = pit_data.get("annual_balance",  pd.DataFrame())
    annual_cf  = pit_data.get("annual_cashflow", pd.DataFrame())
    cutoff_a   = date - pd.Timedelta(days=ANNUAL_REPORT_LAG)

    # 손익 폴백
    if np.isnan(ttm_rev) and not annual_inc.empty:
        a_inc = _avail_cols(annual_inc, cutoff_a)
        if a_inc:
            ac = a_inc[0]  # 최신 연간
            ttm_rev      = _safe_get(annual_inc, ac, "Total Revenue")
            ttm_gross    = _safe_get(annual_inc, ac, "Gross Profit")
            ttm_op       = _safe_get(annual_inc, ac, "Operating Income")
            ttm_net      = _safe_get(annual_inc, ac, "Net Income")
            ttm_ebit     = _safe_get(annual_inc, ac, "EBIT")
            ttm_ebitda   = _safe_get(annual_inc, ac, "EBITDA")
            ttm_interest = _safe_get(annual_inc, ac, "Interest Expense")
            if np.isnan(ttm_interest):
                ttm_interest = _safe_get(annual_inc, ac, "Interest Expense Non Operating")
            ttm_eps = _safe_get(annual_inc, ac, "Basic EPS")
            # 전년 비교 (연간 2번째)
            if len(a_inc) >= 2:
                ac2 = a_inc[1]
                prev_rev = _safe_get(annual_inc, ac2, "Total Revenue")
                prev_net = _safe_get(annual_inc, ac2, "Net Income")
                prev_eps = _safe_get(annual_inc, ac2, "Basic EPS")

    # 대차대조표 폴백
    if np.isnan(ta) and not annual_bal.empty:
        a_bal = _avail_cols(annual_bal, cutoff_a)
        if a_bal:
            abc = a_bal[0]
            ta = _safe_get(annual_bal, abc, "Total Assets")
            eq = _safe_get_multi(annual_bal, abc, [
                "Stockholders Equity",
                "Total Equity Gross Minority Interest",
                "Common Stock Equity"])
            cur_a = _safe_get(annual_bal, abc, "Current Assets")
            cur_l = _safe_get(annual_bal, abc, "Current Liabilities")
            total_debt = _safe_get_multi(annual_bal, abc, [
                "Total Debt", "Long Term Debt And Capital Lease Obligation"])
            cash = _safe_get_multi(annual_bal, abc, [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments"])
            if np.isnan(shares_out):
                shares_out = _safe_get_multi(annual_bal, abc, [
                    "Ordinary Shares Number", "Share Issued"])

    # 현금흐름 폴백
    if np.isnan(ttm_ocf) and not annual_cf.empty:
        a_cf = _avail_cols(annual_cf, cutoff_a)
        if a_cf:
            acf = a_cf[0]
            ttm_ocf   = _safe_get(annual_cf, acf, "Operating Cash Flow")
            ttm_capex = _safe_get(annual_cf, acf, "Capital Expenditure")
            ttm_fcf   = _safe_get(annual_cf, acf, "Free Cash Flow")
            if np.isnan(ttm_fcf) and not np.isnan(ttm_ocf) and not np.isnan(ttm_capex):
                ttm_fcf = ttm_ocf + ttm_capex

    return {
        "ttm_rev": ttm_rev, "ttm_gross": ttm_gross, "ttm_op": ttm_op,
        "ttm_net": ttm_net, "ttm_ebit": ttm_ebit, "ttm_ebitda": ttm_ebitda,
        "ttm_interest": ttm_interest, "ttm_eps": ttm_eps,
        "ttm_ocf": ttm_ocf, "ttm_capex": ttm_capex, "ttm_fcf": ttm_fcf,
        "total_assets": ta, "equity": eq,
        "current_assets": cur_a, "current_liabilities": cur_l,
        "total_debt": total_debt, "cash": cash,
        "shares_outstanding": shares_out,
        "prev_rev": prev_rev, "prev_net": prev_net, "prev_eps": prev_eps,
    }


def _compute_pit_metrics(pit_values: dict, hist_price: float, shares: float) -> dict:
    """PIT 재무 값 + 역사적 주가 → 22개 지표 계산 (Ver3.6 확장).
    hist_price: 해당 리밸런싱 시점의 실제 주가
    shares: 발행주식수 (PIT balance sheet 우선, 없으면 .info 값)
    """
    if not pit_values or np.isnan(hist_price):
        return {}

    # PIT shares 우선 사용
    pit_shares = pit_values.get("shares_outstanding", np.nan)
    s = pit_shares if not np.isnan(pit_shares) else shares
    if not s or np.isnan(s):
        return {}

    def sdiv(a, b):
        try:
            if np.isnan(a) or np.isnan(b) or b == 0:
                return np.nan
            return float(a) / float(b)
        except Exception:
            return np.nan

    r   = pit_values.get("ttm_rev",   np.nan)
    gp  = pit_values.get("ttm_gross", np.nan)
    op  = pit_values.get("ttm_op",    np.nan)
    ni  = pit_values.get("ttm_net",   np.nan)
    ebit  = pit_values.get("ttm_ebit",    np.nan)
    ebitda = pit_values.get("ttm_ebitda",  np.nan)
    iexp  = pit_values.get("ttm_interest", np.nan)
    eps   = pit_values.get("ttm_eps",      np.nan)
    fcf   = pit_values.get("ttm_fcf",      np.nan)
    ta    = pit_values.get("total_assets",  np.nan)
    eq    = pit_values.get("equity",        np.nan)
    ca    = pit_values.get("current_assets", np.nan)
    cl    = pit_values.get("current_liabilities", np.nan)
    debt  = pit_values.get("total_debt",    np.nan)
    cash  = pit_values.get("cash",          np.nan)
    pr    = pit_values.get("prev_rev",      np.nan)
    pn    = pit_values.get("prev_net",      np.nan)
    pe_   = pit_values.get("prev_eps",      np.nan)

    mktcap = hist_price * s
    ev = mktcap + (debt if not np.isnan(debt) else 0) - (cash if not np.isnan(cash) else 0)

    rev_g = sdiv(r - pr, abs(pr)) if not np.isnan(pr) else np.nan
    ni_g  = sdiv(ni - pn, abs(pn)) if not np.isnan(pn) else np.nan
    eps_g = sdiv(eps - pe_, abs(pe_)) if not np.isnan(pe_) else np.nan

    pe_val = sdiv(mktcap, ni)
    peg = sdiv(pe_val, eps_g * 100) if (not np.isnan(eps_g) and eps_g > 0) else np.nan

    return {
        "P_E":               pe_val,
        "P_B":               sdiv(mktcap, eq),
        "P_S":               sdiv(mktcap, r),
        "EV_EBITDA":         sdiv(ev, ebitda),
        "P_FCF":             sdiv(mktcap, fcf) if (not np.isnan(fcf) and fcf > 0) else np.nan,
        "FCF_Yield":         sdiv(fcf, mktcap),
        "PEG_Ratio":         peg,
        "ROE":               sdiv(ni, eq),
        "ROA":               sdiv(ni, ta),
        "Gross_Margin":      sdiv(gp, r),
        "Op_Margin":         sdiv(op, r),
        "EBITDA_Margin":     sdiv(ebitda, r),
        "Rev_Growth":        rev_g,
        "NI_Growth":         ni_g,
        "EPS_Growth":        eps_g,
        "Debt_Equity":       sdiv(debt, eq),
        "Current_Ratio":     sdiv(ca, cl),
        "Interest_Coverage": sdiv(ebit, abs(iexp)) if not np.isnan(iexp) else np.nan,
        "Asset_Turnover":    sdiv(r, ta),
        "GP_A_Quality":      sdiv(gp, ta),
        "MktCap_Log":        np.log(max(mktcap, 1)),
    }


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def _fund_features(fund: dict, price: float, pit_overrides: dict = None,
                   backtest_mode: bool = False) -> dict:
    """yfinance info dict → 펀더멘털 피처.
    backtest_mode=True: .info 미사용, PIT 전용 (Look-Ahead 제거)
    backtest_mode=False: .info 기본값 + PIT 덮어쓰기 (실시간 추천용)
    """
    if backtest_mode:
        # ── 백테스트: .info 완전 차단, PIT만 사용 ──────────
        _ALL_FUND_KEYS = [
            "P_E", "P_B", "P_S", "EV_EBITDA", "P_FCF", "FCF_Yield",
            "Div_Yield", "PEG_Ratio", "ROE", "ROA", "Gross_Margin",
            "Op_Margin", "EBITDA_Margin", "Rev_Growth", "NI_Growth",
            "EPS_Growth", "Debt_Equity", "Current_Ratio", "Interest_Coverage",
            "Asset_Turnover", "GP_A_Quality", "MktCap_Log",
        ]
        result = {k: np.nan for k in _ALL_FUND_KEYS}
        if pit_overrides:
            for k, v in pit_overrides.items():
                if k in result and not (isinstance(v, float) and np.isnan(v)):
                    result[k] = v
        return result

    # ── 실시간: 기존 .info 기반 + PIT 덮어쓰기 ───────────
    mkt  = fund.get("mkt_cap", np.nan) or np.nan
    fcf  = fund.get("fcf",     np.nan) or np.nan
    rev  = fund.get("revenue", np.nan) or np.nan
    ta   = fund.get("total_assets", 1) or 1
    eq   = max(fund.get("total_assets", 1) - (fund.get("total_debt", 0) or 0), 1)
    ebit_val = fund.get("ebit", np.nan)
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

    result = {
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
    if pit_overrides:
        for k, v in pit_overrides.items():
            if k in result and not (isinstance(v, float) and np.isnan(v)):
                result[k] = v
    return result


def snapshot_at_date(
    ticker: str,
    tech_df: pd.DataFrame,
    fund: dict,
    date: pd.Timestamp,
    pit_data: dict = None,
    hist_price: float = np.nan,
    backtest_mode: bool = False,
) -> dict | None:
    """특정 날짜 기준 단일 종목 전체 피처 추출.
    backtest_mode=True: PIT 전용, .info 미사용
    """
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

    # PIT 재무 지표 계산 (Ver3.6: 22개 지표)
    pit_overrides = {}
    if pit_data and not (isinstance(hist_price, float) and np.isnan(hist_price)):
        pit_vals = _get_pit_values(pit_data, date)
        if pit_vals:
            # PIT shares 우선, 없으면 .info shares
            shares = fund.get("shares", np.nan)
            pit_overrides = _compute_pit_metrics(pit_vals, hist_price, shares or np.nan)

    fund_feat = _fund_features(fund, hist_price, pit_overrides=pit_overrides,
                               backtest_mode=backtest_mode)
    for k, v in fund_feat.items():
        features[k] = v

    return features


def build_snapshot_df(
    tickers: list,
    tech_map: dict,
    fund_map: dict,
    date: pd.Timestamp,
    min_history: int = 252,
    pit_map: dict = None,
    price_data: dict = None,
    backtest_mode: bool = False,
) -> pd.DataFrame:
    """모든 종목에 대해 특정 날짜의 피처 DataFrame 구성.
    pit_map: {ticker: {"income": df, "balance": df, "cashflow": df, ...}}
    price_data: {ticker: OHLCV DataFrame} — 역사적 주가 조회용
    """
    rows = []
    for t in tickers:
        td = tech_map.get(t)
        if td is None or len(td) < min_history:
            continue

        # 해당 날짜의 실제 주가 조회
        hist_price = np.nan
        if price_data is not None and t in price_data:
            ohlcv = price_data[t]
            mask_p = ohlcv.index <= date
            if mask_p.any():
                hist_price = float(ohlcv.loc[mask_p, "Close"].iloc[-1])

        pit_data = pit_map.get(t) if pit_map else None
        feat = snapshot_at_date(
            t, td, fund_map.get(t, {}), date,
            pit_data=pit_data, hist_price=hist_price,
            backtest_mode=backtest_mode,
        )
        if feat:
            rows.append(feat)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ticker")
    return df


# ═══════════════════════════════════════════════════════════
# SNAPSHOT ENRICHMENT (Ver3.9)
# ═══════════════════════════════════════════════════════════

def enrich_snapshot(
    snap: pd.DataFrame,
    date: pd.Timestamp,
    spy_close: pd.Series = None,
    vix_close: pd.Series = None,
    sector_map: dict = None,
) -> pd.DataFrame:
    """스냅샷에 Cross-Sectional 백분위 + Regime + Sector RS 추가.
    snap: build_snapshot_df 결과 (index=ticker, columns=features)
    spy_close: SPY 종가 시리즈
    vix_close: VIX 종가 시리즈
    sector_map: {ticker: sector_name}
    """
    if snap.empty:
        return snap

    # ── Cross-Sectional 백분위 (해당 시점 전체 종목 대비 순위%) ──
    cs_pairs = [
        ("Mom_3m",        "Mom_3m_Pct"),
        ("RS_Market_3m",  "RS_Market_3m_Pct"),
        ("P_E",           "P_E_Pct"),
        ("ROE",           "ROE_Pct"),
        ("Volatility_30d","Volatility_30d_Pct"),
    ]
    for src, dst in cs_pairs:
        if src in snap.columns:
            snap[dst] = snap[src].rank(pct=True)
        else:
            snap[dst] = np.nan

    # ── Regime Feature (시장 전체 — 모든 종목에 동일 값) ────────
    # Market_Regime: SPY / SPY_SMA200 - 1
    if spy_close is not None:
        spy_s = spy_close[spy_close.index <= date]
        if len(spy_s) >= 200:
            sma200 = spy_s.rolling(200).mean().iloc[-1]
            snap["Market_Regime"] = spy_s.iloc[-1] / sma200 - 1 if sma200 > 0 else 0
        else:
            snap["Market_Regime"] = np.nan
    else:
        snap["Market_Regime"] = np.nan

    # VIX_Level
    if vix_close is not None:
        vix_s = vix_close[vix_close.index <= date]
        if len(vix_s) > 0:
            snap["VIX_Level"] = float(vix_s.iloc[-1])
        else:
            snap["VIX_Level"] = np.nan
    else:
        snap["VIX_Level"] = np.nan

    # ── Sector Relative Strength (종목 수익률 - 섹터 평균 수익률) ──
    if sector_map:
        snap["_sector"] = snap.index.map(lambda t: sector_map.get(t, "Unknown"))
        for ret_col, dst_col in [("Mom_1m", "RS_Sector_1m"), ("Mom_3m", "RS_Sector_3m")]:
            if ret_col in snap.columns:
                sector_avg = snap.groupby("_sector")[ret_col].transform("mean")
                snap[dst_col] = snap[ret_col] - sector_avg
            else:
                snap[dst_col] = np.nan
        snap.drop(columns=["_sector"], inplace=True)
    else:
        snap["RS_Sector_1m"] = np.nan
        snap["RS_Sector_3m"] = np.nan

    return snap


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
                       start: pd.Timestamp, end: pd.Timestamp,
                       use_next_open: bool = False) -> float:
    """start → end 수익률. use_next_open=True이면 T+1 시가 기준."""
    try:
        ohlcv = price_data[ticker]
        if use_next_open:
            s = ohlcv.loc[ohlcv.index > start, "Open"].iloc[0]
            e = ohlcv.loc[ohlcv.index > end,   "Open"].iloc[0]
        else:
            s = ohlcv.loc[ohlcv.index >= start, "Close"].iloc[0]
            e = ohlcv.loc[ohlcv.index >= end,   "Close"].iloc[0]
        return e / s - 1
    except Exception:
        return np.nan


def _delisted_return(price_data: dict, ticker: str,
                     start: pd.Timestamp, end: pd.Timestamp,
                     use_next_open: bool = False) -> float:
    """보유 기간 중 상폐/거래중단 종목의 수익률 산출.
    마지막 거래 가격 기준으로 수익률 계산. 데이터 전무 시 -100%.
    """
    ohlcv = price_data.get(ticker)
    if ohlcv is None:
        return -1.0

    # 진입가 결정
    if use_next_open:
        after_start = ohlcv.loc[ohlcv.index > start]
        if len(after_start) == 0:
            return -1.0
        entry = float(after_start["Open"].iloc[0])
    else:
        on_or_after = ohlcv.loc[ohlcv.index >= start]
        if len(on_or_after) == 0:
            return -1.0
        entry = float(on_or_after["Close"].iloc[0])

    if entry <= 0:
        return -1.0

    # 보유 기간 내 마지막 가용 종가
    sub = ohlcv.loc[(ohlcv.index >= start) & (ohlcv.index <= end), "Close"]
    if len(sub) == 0:
        return -1.0

    last_price = float(sub.iloc[-1])
    return last_price / entry - 1


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
    pit_map:      dict = None,
    min_dollar_vol: float = 0,
    use_next_open:  bool = False,
    sp500_changes: pd.DataFrame = None,
    current_sp500: list = None,
    use_ensemble:   bool = True,
    spy_close:      pd.Series = None,
    vix_close:      pd.Series = None,
    sector_map:     dict = None,
    use_mom_filter:      bool = True,
    use_turnover_buffer: bool = False,
    turnover_buffer_pct: float = 0.05,
    use_inv_vol_weight:  bool = False,
) -> dict:
    """메인 백테스트 엔진.
    Ver3.9: enrich_snapshot으로 CS정규화 + Regime + Sector RS 추가.
    """
    n_dates = len(rebal_dates)
    feature_cols = FEAT_COLS

    # 결측 플래그 대상: 펀더멘털 그룹
    _FUND_GROUPS = {"밸류에이션", "수익성", "성장성", "재무안정성", "효율성", "규모"}

    # ── Step 1: 모든 리밸런싱 날짜의 스냅샷 + 실제 수익률 계산 ──
    progress(0.05, "📊 지표 스냅샷 계산 중...")
    snapshots: dict = {}  # date → pd.DataFrame (includes forward_return)

    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]

        # ── 생존자 편향 보정: 해당 날짜의 역사적 유니버스 ──
        if sp500_changes is not None and current_sp500 is not None:
            hist_members = reconstruct_sp500_at_date(current_sp500, sp500_changes, date)
            tickers = [t for t in hist_members if t in price_data]
        else:
            tickers = list(price_data.keys())

        # ── 유동성 필터: 20일 평균 거래대금 기준 ──────────
        if min_dollar_vol > 0:
            liquid = []
            for t in tickers:
                ohlcv = price_data.get(t)
                if ohlcv is None:
                    continue
                mask = ohlcv.index <= date
                if mask.sum() >= 20:
                    recent = ohlcv[mask].tail(20)
                    adv = (recent["Close"] * recent["Volume"]).mean()
                    if adv >= min_dollar_vol:
                        liquid.append(t)
            tickers = liquid

        snap = build_snapshot_df(tickers, tech_map, fund_map, date,
                                      pit_map=pit_map, price_data=price_data,
                                      backtest_mode=True)
        if snap.empty:
            snapshots[date] = snap
            continue

        # Ver3.9: CS정규화 + Regime + Sector RS 추가
        snap = enrich_snapshot(snap, date,
                               spy_close=spy_close, vix_close=vix_close,
                               sector_map=sector_map)

        # 실제 forward return 추가 (체결가 가정 반영)
        fwd = {t: fwd_ret_from_price(price_data, t, date, next_date, use_next_open)
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
    feat_imp_rf_rows   = []   # 모델별 중요도 (Ver3.8)
    feat_imp_xgb_rows  = []
    feat_imp_lgbm_rows = []
    rebal_history  = []
    prev_selected  = set()   # 직전 기간 보유 종목 (turnover 계산용)

    imputer = SimpleImputer(strategy="median")

    # 마지막 학습 정보 (tab_realtime 전달용)
    last_win_bounds  = {}
    last_miss_src    = []
    last_all_cols    = []
    last_avail_cols  = []

    for i in range(rolling_win, n_dates - 1):
        date      = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        # ── 훈련 데이터 수집 (Ver4.2: 엠바고 적용) ─────────
        # 예측 날짜와 21일 이내 겹치는 스냅샷 제외 → 데이터 누수 방지
        EMBARGO_DAYS = 21
        X_list, y_list = [], []
        for j in range(i - rolling_win, i):
            snap_date = rebal_dates[j]
            # 엠바고: 훈련 스냅샷의 forward return 종료일이 예측일과 겹치면 제외
            if (date - snap_date).days < EMBARGO_DAYS:
                continue
            snap = snapshots.get(snap_date)
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

        # ── 결측 플래그 (펀더멘털 피처) ───────────────────
        miss_src = [c for c in avail_cols
                    if FEATURE_META.get(c, {}).get("group", "") in _FUND_GROUPS]
        for mc in miss_src:
            X_train[f"{mc}_miss"] = X_train[mc].isna().astype(int)
        all_train_cols = X_train.columns.tolist()

        # ── Ver4.2: 결측 대체 먼저 → 윈저화 (순서 수정) ──
        # keep_empty_features=True로 all-NaN 컬럼도 유지 (컬럼 수 불일치 방지)
        imputer_step = SimpleImputer(strategy="median", keep_empty_features=True)
        X_imp_df = pd.DataFrame(
            imputer_step.fit_transform(X_train),
            columns=all_train_cols, index=X_train.index,
        )
        win_bounds = {}
        for col in avail_cols:
            lo, hi = X_imp_df[col].quantile(0.01), X_imp_df[col].quantile(0.99)
            if lo < hi:
                win_bounds[col] = (lo, hi)
                X_imp_df[col] = X_imp_df[col].clip(lo, hi)
        X_imp = X_imp_df.values

        # ── Ver4.2: RF 하이퍼파라미터 자동 튜닝 ──────────
        param_grid = {"max_depth": [3, 5, 7], "min_samples_leaf": [5, 10, 20]}
        base_rf = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=1,
        )
        gcv = GridSearchCV(base_rf, param_grid, cv=3, scoring="neg_mean_squared_error",
                           n_jobs=1, refit=True)
        gcv.fit(X_imp, y_train)
        model_rf = gcv.best_estimator_

        # XGBoost + LightGBM (앙상블 모드 시)
        model_xgb = None
        model_lgbm = None
        if use_ensemble:
            model_xgb = XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=1, verbosity=0,
            )
            model_xgb.fit(X_imp, y_train)

            model_lgbm = LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=1, verbose=-1,
            )
            model_lgbm.fit(X_imp, y_train)

        # 중요도: 모델별 개별 저장 (raw) + 앙상블 평균 (정규화 후)
        # LightGBM은 split 기반으로 RF/XGBoost 대비 스케일이 수백배 다름
        # → 각 모델을 합계 1로 정규화한 뒤 평균해야 공정한 앙상블 중요도
        imp_rf_raw = {}
        imp_xgb_raw = {}
        imp_lgbm_raw = {}
        for idx_c, col_name in enumerate(all_train_cols):
            if col_name in avail_cols:
                imp_rf_raw[col_name] = model_rf.feature_importances_[idx_c]
                if use_ensemble:
                    imp_xgb_raw[col_name]  = model_xgb.feature_importances_[idx_c]
                    imp_lgbm_raw[col_name] = model_lgbm.feature_importances_[idx_c]

        # 정규화 (합계 1)
        def _norm(d):
            s = sum(d.values())
            return {k: v / s for k, v in d.items()} if s > 0 else d

        imp_rf_n = _norm(imp_rf_raw)
        imp_dict = dict(imp_rf_n)  # RF 단독일 때 기본값

        if use_ensemble:
            imp_xgb_n  = _norm(imp_xgb_raw)
            imp_lgbm_n = _norm(imp_lgbm_raw)
            imp_dict = {k: (imp_rf_n.get(k, 0) + imp_xgb_n.get(k, 0) + imp_lgbm_n.get(k, 0)) / 3
                        for k in imp_rf_n}

        feat_imp_rows.append({"date": date, **imp_dict})
        feat_imp_rf_rows.append({"date": date, **imp_rf_raw})
        if use_ensemble:
            feat_imp_xgb_rows.append({"date": date, **imp_xgb_raw})
            feat_imp_lgbm_rows.append({"date": date, **imp_lgbm_raw})

        # ── 예측 ─────────────────────────────────────────
        cur_snap = snapshots.get(date)
        if cur_snap is None or cur_snap.empty:
            continue

        X_pred = cur_snap[[c for c in avail_cols if c in cur_snap.columns]].reindex(columns=avail_cols)

        # 결측 플래그 추가
        for mc in miss_src:
            X_pred[f"{mc}_miss"] = X_pred[mc].isna().astype(int) if mc in X_pred.columns else 0
        X_pred = X_pred.reindex(columns=all_train_cols)

        # Ver4.2: 결측 대체 먼저 → 윈저화 (학습과 동일 순서)
        X_pred_imp_df = pd.DataFrame(
            imputer_step.transform(X_pred),
            columns=all_train_cols, index=cur_snap.index,
        )
        for col in avail_cols:
            if col in win_bounds and col in X_pred_imp_df.columns:
                lo, hi = win_bounds[col]
                X_pred_imp_df[col] = X_pred_imp_df[col].clip(lo, hi)
        X_pred_imp = X_pred_imp_df.values

        # 앙상블 예측: Rank Average (각 모델 예측 → 순위 변환 → 순위 평균)
        pred_rf_s = pd.Series(model_rf.predict(X_pred_imp), index=cur_snap.index)
        if use_ensemble:
            pred_xgb_s  = pd.Series(model_xgb.predict(X_pred_imp),  index=cur_snap.index)
            pred_lgbm_s = pd.Series(model_lgbm.predict(X_pred_imp), index=cur_snap.index)
            # rank: ascending=False → 예측 수익률 높을수록 1등 (낮은 rank = 좋은 종목)
            rank_rf   = pred_rf_s.rank(ascending=False)
            rank_xgb  = pred_xgb_s.rank(ascending=False)
            rank_lgbm = pred_lgbm_s.rank(ascending=False)
            avg_rank  = (rank_rf + rank_xgb + rank_lgbm) / 3
            # rank average → 낮을수록 좋으므로 nsmallest로 선정하기 위해 부호 반전
            pred_series = -avg_rank  # 높을수록 좋은 종목
        else:
            pred_series = pred_rf_s

        # ── IC 계산 (앙상블: raw 예측 평균 기준 + 모델별) ──
        actual = cur_snap["_fwd_return"].dropna()
        common = pred_rf_s.index.intersection(actual.index)
        ic_val = np.nan
        ic_rec = {"date": date}
        if len(common) >= 10:
            ic_rf, _ = spearmanr(pred_rf_s[common], actual[common])
            ic_rec["IC_RF"] = ic_rf
            if use_ensemble:
                ic_xgb, _ = spearmanr(pred_xgb_s[common], actual[common])
                ic_lgbm, _ = spearmanr(pred_lgbm_s[common], actual[common])
                ic_rec["IC_XGB"]  = ic_xgb
                ic_rec["IC_LGBM"] = ic_lgbm
                # 앙상블 IC: 3모델 raw 예측 평균 기준
                raw_avg = (pred_rf_s[common] + pred_xgb_s[common] + pred_lgbm_s[common]) / 3
                ic_val, _ = spearmanr(raw_avg, actual[common])
            else:
                ic_val = ic_rf
            ic_rec["IC"] = ic_val
            ic_records.append(ic_rec)

        # ── 종목 선정 (모멘텀 필터 + 턴오버 버퍼) ──────────
        if use_mom_filter and "Mom_1m" in cur_snap.columns:
            mom_ok = cur_snap["Mom_1m"] > 0
            candidates = pred_series[pred_series.index.isin(cur_snap[mom_ok].index)]
        else:
            candidates = pred_series
        # Ver4.2: 턴오버 버퍼 — 보유 종목에 가산점 부여
        if use_turnover_buffer and prev_selected:
            candidates_adj = candidates.copy()
            for t in prev_selected:
                if t in candidates_adj.index:
                    candidates_adj[t] *= (1 + turnover_buffer_pct)
            if len(candidates_adj) >= n_stocks:
                selected = candidates_adj.nlargest(n_stocks)
            else:
                selected = pred_series.nlargest(n_stocks)
        elif len(candidates) >= n_stocks:
            selected = candidates.nlargest(n_stocks)
        else:
            selected = pred_series.nlargest(n_stocks)

        # ── Ver4.2: 포트폴리오 가중치 결정 ────────────────
        n_sel = len(selected)
        if n_sel == 0:
            continue
        if use_inv_vol_weight and "Volatility_30d" in cur_snap.columns:
            vols = {}
            for t in selected.index:
                v = cur_snap.loc[t, "Volatility_30d"] if t in cur_snap.index else np.nan
                vols[t] = v if (not np.isnan(v) and v > 0) else 0.30
            inv_vols = {t: 1 / v for t, v in vols.items()}
            total_inv = sum(inv_vols.values())
            weights_dict = {t: iv / total_inv for t, iv in inv_vols.items()}
        else:
            w = 1 / n_sel
            weights_dict = {t: w for t in selected.index}

        # ── 포트폴리오 수익률 (가중 적용) ─────────────────
        port_ret = 0.0
        sel_returns = {}
        for t in selected.index:
            r = fwd_ret_from_price(price_data, t, date, next_date, use_next_open)
            if np.isnan(r):
                r = _delisted_return(price_data, t, date, next_date, use_next_open)
            sel_returns[t] = r
            port_ret += weights_dict[t] * r

        # ── 유니버스 평균·Bottom N 수익률 (선정 품질 지표용) ──
        all_fwd = cur_snap["_fwd_return"].dropna()
        univ_avg_ret = float(all_fwd.mean()) if len(all_fwd) > 0 else np.nan
        # Bottom N: 예측 점수 최하위 N개
        bottom_n = pred_series.nsmallest(n_stocks)
        bottom_ret = 0.0
        n_bot = len(bottom_n)
        for t in bottom_n.index:
            r = fwd_ret_from_price(price_data, t, date, next_date, use_next_open)
            if np.isnan(r):
                r = _delisted_return(price_data, t, date, next_date, use_next_open)
            bottom_ret += r
        if n_bot > 0:
            bottom_ret /= n_bot
        # 적중률: TOP N 중 유니버스 평균 초과 종목 비율
        if not np.isnan(univ_avg_ret) and sel_returns:
            precision = sum(1 for r in sel_returns.values()
                           if not np.isnan(r) and r > univ_avg_ret) / len(sel_returns)
        else:
            precision = np.nan

        # ── Turnover 계산 + 거래비용 적용 ─────────────────
        # turnover = 신규 편입 종목 수 / 전체 보유 종목 수
        # 첫 리밸런싱은 전체 신규 매수이므로 turnover = 1.0
        new_selected = set(selected.index)
        if not prev_selected:
            turnover = 1.0
        else:
            newly_bought = len(new_selected - prev_selected)
            turnover = newly_bought / max(len(new_selected), 1)
        prev_selected = new_selected

        # 실제 거래비용 = 설정 TC × turnover (교체된 비중에만 비용 발생)
        tc_actual = (tc_pct / 100) * turnover
        current_value *= (1 + port_ret - tc_actual)
        portfolio_dates.append(next_date)
        portfolio_values.append(current_value)

        # ── 리밸런싱 히스토리 ─────────────────────────────
        ticker_rows = []
        for t in selected.index:
            if use_ensemble:
                row_d = {"ticker": t,
                         "비중": f"{weights_dict.get(t, 0):.1%}",
                         "평균순위": avg_rank.get(t, np.nan),
                         "RF순위": rank_rf.get(t, np.nan),
                         "XGB순위": rank_xgb.get(t, np.nan),
                         "LGBM순위": rank_lgbm.get(t, np.nan),
                         "실제수익률": actual.get(t, np.nan)}
            else:
                row_d = {"ticker": t,
                         "비중": f"{weights_dict.get(t, 0):.1%}",
                         "예측수익률": pred_series[t],
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
            "turnover":        turnover,
            "tc_actual":       tc_actual,
            "univ_avg_ret":    univ_avg_ret,
            "bottom_ret":      bottom_ret,
            "precision":       precision,
        })

        # 마지막 학습 정보 갱신
        last_win_bounds = win_bounds
        last_miss_src   = miss_src
        last_all_cols   = all_train_cols
        last_avail_cols = avail_cols

        pct = 0.30 + 0.65 * (i - rolling_win) / max(n_dates - rolling_win - 1, 1)
        progress(min(pct, 0.95), f"백테스트 진행 중 ({date.strftime('%Y-%m')})...")

    progress(0.98, "결과 정리 중...")

    # ── feat_imp_history DataFrame ────────────────────────
    def _to_fimp_df(rows):
        if rows:
            d = pd.DataFrame(rows).set_index("date")
            d.index = pd.to_datetime(d.index)
            return d
        return pd.DataFrame()

    fimp_df      = _to_fimp_df(feat_imp_rows)
    fimp_rf_df   = _to_fimp_df(feat_imp_rf_rows)
    fimp_xgb_df  = _to_fimp_df(feat_imp_xgb_rows)
    fimp_lgbm_df = _to_fimp_df(feat_imp_lgbm_rows)

    ic_df = pd.DataFrame(ic_records) if ic_records else pd.DataFrame()

    # 마지막 학습된 모델·imputer·피처 컬럼·전처리 정보 저장
    # → 실시간 추천 탭에서 동일 전처리 + 앙상블 예측 적용
    return {
        "port_dates":       portfolio_dates,
        "port_values":      portfolio_values,
        "ic_df":            ic_df,
        "fimp_df":          fimp_df,
        "fimp_rf_df":       fimp_rf_df,
        "fimp_xgb_df":      fimp_xgb_df,
        "fimp_lgbm_df":     fimp_lgbm_df,
        "rebal_hist":       rebal_history,
        "last_model_rf":    model_rf   if "model_rf"   in dir() else None,
        "last_model_xgb":   model_xgb  if "model_xgb"  in dir() else None,
        "last_model_lgbm":  model_lgbm if "model_lgbm" in dir() else None,
        "last_imputer":     imputer_step if "imputer_step" in dir() else None,
        "last_avail_cols":  last_avail_cols,
        "last_all_cols":    last_all_cols,
        "last_win_bounds":  last_win_bounds,
        "last_miss_src":    last_miss_src,
        "use_ensemble":     use_ensemble,
        "rebal_m":          rolling_win,   # tab_ic 연간 턴오버 계산용
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


def calc_metrics(series: pd.Series, label: str, rf: float = 0.03) -> dict:
    """성과 지표 계산. rf: 연간 무위험수익률 (소수, 예: 0.05 = 5%).
    Sharpe/Sortino: 데이터 빈도 자동 감지 후 올바른 연율화 계수 적용.
    """
    if len(series) < 2:
        return {"지표": label}
    r  = series.pct_change().dropna()
    yr = (series.index[-1] - series.index[0]).days / 365.25
    total = series.iloc[-1] / series.iloc[0] - 1
    cagr  = (series.iloc[-1] / series.iloc[0]) ** (1 / max(yr, 0.01)) - 1

    # 데이터 빈도 자동 감지: 평균 간격(일)으로 판단
    if len(r) >= 2:
        avg_gap = (series.index[-1] - series.index[0]).days / len(r)
        if avg_gap > 15:      # 월간 (간격 ~21일)
            ann_factor = 12
        elif avg_gap > 3:     # 주간 (간격 ~5일)
            ann_factor = 52
        else:                 # 일간 (간격 ~1일)
            ann_factor = 252
    else:
        ann_factor = 252

    vol = r.std() * np.sqrt(ann_factor)

    # Sharpe: 초과수익 평균 / 표준편차 → 연율화
    period_rf = rf / ann_factor
    excess    = r - period_rf
    sharpe    = excess.mean() / excess.std() * np.sqrt(ann_factor) if excess.std() > 0 else 0

    # Sortino: 초과수익 평균 / 하방 편차 → 연율화
    neg_excess = excess[excess < 0]
    sortino    = excess.mean() / neg_excess.std() * np.sqrt(ann_factor) if len(neg_excess) > 1 else 0

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
# TAB 0 ── 요약 대시보드
# ═══════════════════════════════════════════════════════════

def tab_summary(results: dict, benchmarks: dict, price_data: dict,
                fund_map: dict = None, tech_map: dict = None, n_stocks: int = 5,
                rf: float = 0.03,
                pit_map: dict = None, spy_close: pd.Series = None,
                vix_close: pd.Series = None, sector_map: dict = None):
    """전체 탭의 핵심 항목을 한 화면에 요약."""
    pd_ = results["port_dates"]
    pv_ = results["port_values"]
    rebal_hist = results.get("rebal_hist", [])
    ic_df = results.get("ic_df", pd.DataFrame())
    fimp = results.get("fimp_df", pd.DataFrame())
    is_ensemble = results.get("use_ensemble", False)

    if len(pd_) < 2:
        st.warning("백테스트 데이터가 부족합니다.")
        return

    # ── 운용 시작 시점 ────────────────────────────────────
    trade_start = rebal_hist[0]["rebalance_date"] if rebal_hist else pd.Timestamp(pd_[0])
    port_period = pd.Series(pv_, index=pd.DatetimeIndex(pd_))
    port_active = port_period[port_period.index >= trade_start]
    if len(port_active) >= 2:
        port_active = port_active / port_active.iloc[0]

    ai_metrics = calc_metrics(port_active, "AI 전략", rf=rf)
    spy_metrics = {}
    if "SPY" in benchmarks:
        ns = norm_series(benchmarks["SPY"], trade_start)
        if len(ns) > 1:
            spy_metrics = calc_metrics(ns, "S&P 500", rf=rf)

    # ════════════════════════════════════════════════════════
    # SECTION 1: 핵심 성과 지표
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">🏆 핵심 성과 요약</div>', unsafe_allow_html=True)

    def _parse(s):
        try:
            return float(str(s).strip("%").strip("x"))
        except Exception:
            return 0.0

    cagr_ai  = _parse(ai_metrics.get("CAGR", "0"))
    sharp_ai = _parse(ai_metrics.get("Sharpe", "0"))
    mdd_ai   = _parse(ai_metrics.get("Max DD", "0"))
    total_ai = _parse(ai_metrics.get("총수익률", "0"))
    sort_ai  = _parse(ai_metrics.get("Sortino", "0"))
    winr_ai  = _parse(ai_metrics.get("월승률", "0"))

    cagr_spy  = _parse(spy_metrics.get("CAGR", "0"))
    sharp_spy = _parse(spy_metrics.get("Sharpe", "0"))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, lbl, val, unit, clr in [
        (c1, "총수익률", total_ai, "%", "pos" if total_ai > 0 else "neg"),
        (c2, "CAGR",     cagr_ai,  "%", "pos" if cagr_ai > 0 else "neg"),
        (c3, "Sharpe",   sharp_ai, "",  "pos" if sharp_ai > 0.5 else ("neg" if sharp_ai < 0 else "neu")),
        (c4, "Sortino",  sort_ai,  "",  "pos" if sort_ai > 0.5 else "neu"),
        (c5, "Max DD",   mdd_ai,   "%", "neg"),
        (c6, "월승률",   winr_ai,  "%", "pos" if winr_ai > 50 else "neg"),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value {clr}">{val:.1f}{unit}</div>
        </div>""", unsafe_allow_html=True)

    if spy_metrics:
        alpha = cagr_ai - cagr_spy
        alpha_clr = "pos" if alpha > 0 else "neg"
        st.markdown(
            f'<div style="text-align:center; font-size:0.85rem; margin:0.3rem 0 0.8rem;">'
            f'vs S&P 500 — CAGR 차이: '
            f'<span class="{alpha_clr}" style="font-weight:700;">{alpha:+.1f}%p</span> · '
            f'Sharpe 차이: <span class="{alpha_clr}" style="font-weight:700;">'
            f'{sharp_ai - sharp_spy:+.2f}</span></div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════
    # SECTION 2: 누적 수익률 + 성과 지표 테이블
    # ════════════════════════════════════════════════════════
    col_chart, col_tbl = st.columns([6, 4])

    with col_chart:
        st.markdown('<div class="section-hdr">📈 누적 수익률</div>', unsafe_allow_html=True)
        port_daily = build_daily_portfolio(results, price_data)
        port_daily_active = port_daily[port_daily.index >= trade_start]
        if len(port_daily_active) >= 2:
            port_daily_active = port_daily_active / port_daily_active.iloc[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=port_daily_active.index, y=port_daily_active.values,
            name="AI 전략", line=dict(color="#7c4dff", width=2.5),
        ))
        colors_bm = {"SPY": "#ff6b35", "QQQ": "#00c9a7"}
        bm_metrics_list = []
        for tk, label in BENCHMARKS.items():
            if tk in benchmarks and tk in colors_bm:
                ns = norm_series(benchmarks[tk], trade_start)
                if len(ns) > 1:
                    fig.add_trace(go.Scatter(
                        x=ns.index, y=ns.values, name=label,
                        line=dict(color=colors_bm[tk], width=1.8),
                    ))
                    bm_metrics_list.append(calc_metrics(ns, label, rf=rf))
        fig.update_layout(
            **PLOT_CFG, height=300, margin=dict(l=40, r=10, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_tbl:
        st.markdown('<div class="section-hdr">📊 성과 지표 비교</div>', unsafe_allow_html=True)
        all_metrics = [ai_metrics] + bm_metrics_list
        mdf = pd.DataFrame(all_metrics).set_index("전략").T
        st.dataframe(mdf, use_container_width=True, height=300)

    # ════════════════════════════════════════════════════════
    # SECTION 2b: IC & 모델 예측력
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">🎯 IC & 모델 예측력</div>', unsafe_allow_html=True)
    if not ic_df.empty:
        ic_df_c = ic_df.copy()
        ic_df_c["date"] = pd.to_datetime(ic_df_c["date"])
        ic_mean = ic_df_c["IC"].mean()
        ic_std  = ic_df_c["IC"].std()
        ic_ir   = ic_mean / ic_std if ic_std > 0 else 0
        pos_r   = (ic_df_c["IC"] > 0).mean()

        ic1, ic2, ic3, ic4 = st.columns(4)
        for col, lbl, val in [
            (ic1, "평균 IC", ic_mean),
            (ic2, "IC IR", ic_ir),
            (ic3, "+IC%", pos_r),
            (ic4, "IC 표준편차", ic_std),
        ]:
            clr = "pos" if val > 0.03 else ("neg" if val < 0 else "neu")
            col.markdown(f"""<div class="metric-box">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value {clr}">{val:.3f}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("IC 데이터 없음")

    # ════════════════════════════════════════════════════════
    # SECTION 3: 종목 선정 품질 (핵심 지표)
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">🎯 종목 선정 품질</div>', unsafe_allow_html=True)

    period_returns = [h["port_return"] for h in rebal_hist]
    univ_avgs   = [h.get("univ_avg_ret", np.nan) for h in rebal_hist]
    bottom_rets = [h.get("bottom_ret", np.nan) for h in rebal_hist]
    precisions  = [h.get("precision", np.nan) for h in rebal_hist]

    # ① Top N vs 유니버스 평균 (선정 초과수익)
    excess_vs_univ = [p - u for p, u in zip(period_returns, univ_avgs)
                      if not np.isnan(u)]
    avg_excess = float(np.mean(excess_vs_univ)) if excess_vs_univ else np.nan

    # ② 롱숏 스프레드 (Top N - Bottom N)
    ls_spreads = [p - b for p, b in zip(period_returns, bottom_rets)
                  if not np.isnan(b)]
    avg_ls = float(np.mean(ls_spreads)) if ls_spreads else np.nan

    # ③ 적중률 (Top N 중 유니버스 평균 초과 비율)
    valid_prec = [p for p in precisions if not np.isnan(p)]
    avg_precision = float(np.mean(valid_prec)) if valid_prec else np.nan

    # ④ Profit Factor (총이익 / 총손실)
    gains  = sum(r for r in period_returns if r > 0)
    losses = sum(abs(r) for r in period_returns if r < 0)
    profit_factor = gains / losses if losses > 0 else float("inf")

    q1, q2, q3, q4 = st.columns(4)
    q1.markdown(f"""<div class="metric-box">
        <div class="metric-label">선정 초과수익<br>(Top N - 유니버스 평균)</div>
        <div class="metric-value {'pos' if avg_excess > 0 else 'neg'}">{avg_excess:+.2%}</div>
    </div>""", unsafe_allow_html=True)
    q2.markdown(f"""<div class="metric-box">
        <div class="metric-label">롱숏 스프레드<br>(Top N - Bottom N)</div>
        <div class="metric-value {'pos' if avg_ls > 0 else 'neg'}">{avg_ls:+.2%}</div>
    </div>""", unsafe_allow_html=True)
    q3.markdown(f"""<div class="metric-box">
        <div class="metric-label">적중률<br>(평균 초과 종목 비율)</div>
        <div class="metric-value {'pos' if avg_precision > 0.5 else 'neg'}">{avg_precision:.1%}</div>
    </div>""", unsafe_allow_html=True)
    pf_clr = "pos" if profit_factor > 1.5 else ("neu" if profit_factor > 1 else "neg")
    pf_str = f"{profit_factor:.2f}" if profit_factor < 100 else "∞"
    q4.markdown(f"""<div class="metric-box">
        <div class="metric-label">Profit Factor<br>(총이익 / 총손실)</div>
        <div class="metric-value {pf_clr}">{pf_str}</div>
    </div>""", unsafe_allow_html=True)

    # 선정 초과수익 추이 차트
    if excess_vs_univ:
        col_ex, col_ls = st.columns(2)
        with col_ex:
            ex_dates = [h["rebalance_date"] for h, u in zip(rebal_hist, univ_avgs) if not np.isnan(u)]
            clrs = ["#2e7d32" if v > 0 else "#c62828" for v in excess_vs_univ]
            fig_ex = go.Figure(go.Bar(
                x=[d.strftime("%Y-%m") for d in ex_dates], y=excess_vs_univ,
                marker_color=clrs,
                hovertemplate="%{x}<br>초과수익: %{y:.2%}<extra></extra>",
            ))
            fig_ex.add_hline(y=0, line_color="#555", line_width=1)
            fig_ex.add_hline(y=avg_excess, line_dash="dash", line_color="#7c4dff",
                             annotation_text=f"평균 {avg_excess:+.2%}", annotation_position="top right")
            fig_ex.update_layout(**PLOT_CFG, height=200, title="Top N vs 유니버스 평균 (초과수익)",
                                 yaxis_tickformat=".1%", margin=dict(l=40, r=10, t=30, b=30))
            st.plotly_chart(fig_ex, use_container_width=True)

        with col_ls:
            ls_dates = [h["rebalance_date"] for h, b in zip(rebal_hist, bottom_rets) if not np.isnan(b)]
            clrs2 = ["#2e7d32" if v > 0 else "#c62828" for v in ls_spreads]
            fig_ls = go.Figure(go.Bar(
                x=[d.strftime("%Y-%m") for d in ls_dates], y=ls_spreads,
                marker_color=clrs2,
                hovertemplate="%{x}<br>스프레드: %{y:.2%}<extra></extra>",
            ))
            fig_ls.add_hline(y=0, line_color="#555", line_width=1)
            fig_ls.add_hline(y=avg_ls, line_dash="dash", line_color="#7c4dff",
                             annotation_text=f"평균 {avg_ls:+.2%}", annotation_position="top right")
            fig_ls.update_layout(**PLOT_CFG, height=200, title="롱숏 스프레드 (Top N - Bottom N)",
                                 yaxis_tickformat=".1%", margin=dict(l=40, r=10, t=30, b=30))
            st.plotly_chart(fig_ls, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # SECTION 4: 거래 효율 & 승률
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">🔄 거래 효율 & 승률</div>', unsafe_allow_html=True)
    turnover_vals = [h["turnover"] for h in rebal_hist if "turnover" in h]
    avg_to = np.mean(turnover_vals) if turnover_vals else 0
    win_cnt = sum(1 for r in period_returns if r > 0)
    win_rate = win_cnt / len(period_returns) if period_returns else 0
    avg_ret = float(np.mean(period_returns)) if period_returns else 0

    t1, t2, t3, t4 = st.columns(4)
    t1.markdown(f"""<div class="metric-box"><div class="metric-label">리밸런싱 횟수</div>
        <div class="metric-value neu">{len(rebal_hist)}회</div></div>""", unsafe_allow_html=True)
    t2.markdown(f"""<div class="metric-box"><div class="metric-label">리밸런싱 승률</div>
        <div class="metric-value {'pos' if win_rate > 0.5 else 'neg'}">{win_rate:.1%}</div></div>""", unsafe_allow_html=True)
    t3.markdown(f"""<div class="metric-box"><div class="metric-label">기간 평균 수익</div>
        <div class="metric-value {'pos' if avg_ret > 0 else 'neg'}">{avg_ret:.2%}</div></div>""", unsafe_allow_html=True)
    t4.markdown(f"""<div class="metric-box"><div class="metric-label">평균 턴오버</div>
        <div class="metric-value {'pos' if avg_to < 0.5 else 'neg'}">{avg_to:.1%}</div></div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 5: 최근 리밸런싱 + AI 추천
    # ════════════════════════════════════════════════════════
    col_recent, col_rec = st.columns(2)

    with col_recent:
        st.markdown('<div class="section-hdr">📋 최근 리밸런싱</div>', unsafe_allow_html=True)
        if rebal_hist:
            h = rebal_hist[-1]
            ret_c = "pos" if h["port_return"] > 0 else "neg"
            st.markdown(
                f'**{h["rebalance_date"].strftime("%Y-%m-%d")}** → '
                f'{h["next_date"].strftime("%Y-%m-%d")} · '
                f'수익률: <span class="{ret_c}" style="font-weight:700;">{h["port_return"]:.2%}</span>',
                unsafe_allow_html=True,
            )
            tdf = h.get("ticker_df", pd.DataFrame())
            if not tdf.empty:
                priority = ["ticker", "비중"]
                if is_ensemble:
                    priority += ["평균순위", "실제수익률"]
                else:
                    priority += ["예측수익률", "실제수익률"]
                show_cols = [c for c in priority if c in tdf.columns]
                disp = tdf[show_cols].copy()
                if "실제수익률" in disp.columns:
                    disp["실제수익률"] = disp["실제수익률"].apply(
                        lambda x: f"{x:.2%}" if pd.notna(x) and isinstance(x, float) else x)
                st.dataframe(disp, use_container_width=True, hide_index=True, height=260)

    with col_rec:
        st.markdown('<div class="section-hdr">🔴 AI 추천 종목 (현재)</div>', unsafe_allow_html=True)
        model_rf      = results.get("last_model_rf")
        model_xgb     = results.get("last_model_xgb")
        model_lgbm    = results.get("last_model_lgbm")
        last_imputer  = results.get("last_imputer")
        last_avail_cols = results.get("last_avail_cols", [])
        last_all_cols   = results.get("last_all_cols", [])
        last_win_bounds = results.get("last_win_bounds", {})
        last_miss_src   = results.get("last_miss_src", [])
        _cfg = st.session_state.get("cfg") or {}
        _use_inv_vol = _cfg.get("use_inv_vol_weight", False)

        if (model_rf is not None and last_imputer is not None
                and last_all_cols and price_data and tech_map is not None
                and fund_map is not None):
            all_max = [ohlcv.index.max() for ohlcv in price_data.values() if len(ohlcv) > 0]
            latest_date = max(all_max) if all_max else pd.Timestamp.today()
            st.caption(f"기준일: {latest_date.strftime('%Y-%m-%d')}")

            cur_snap = build_snapshot_df(
                list(price_data.keys()), tech_map, fund_map, latest_date,
                pit_map=pit_map, price_data=price_data,
            )
            if not cur_snap.empty:
                cur_snap = enrich_snapshot(
                    cur_snap, latest_date,
                    spy_close=spy_close, vix_close=vix_close,
                    sector_map=sector_map,
                )
                avail = [c for c in last_avail_cols if c in cur_snap.columns]
                X_pred = cur_snap[avail].reindex(columns=last_avail_cols)
                for mc in last_miss_src:
                    X_pred[f"{mc}_miss"] = X_pred[mc].isna().astype(int) if mc in X_pred.columns else 0
                X_pred = X_pred.reindex(columns=last_all_cols)
                X_imp = last_imputer.transform(X_pred)

                pred_rf = pd.Series(model_rf.predict(X_imp), index=cur_snap.index)
                if is_ensemble and model_xgb is not None and model_lgbm is not None:
                    pred_xgb  = pd.Series(model_xgb.predict(X_imp), index=cur_snap.index)
                    pred_lgbm = pd.Series(model_lgbm.predict(X_imp), index=cur_snap.index)
                    avg_rank = (pred_rf.rank(ascending=False) +
                                pred_xgb.rank(ascending=False) +
                                pred_lgbm.rank(ascending=False)) / 3
                    composite = -avg_rank
                else:
                    composite = pred_rf

                _use_mom = _cfg.get("use_mom_filter", True)
                if _use_mom and "Mom_1m" in cur_snap.columns:
                    mom_ok = cur_snap["Mom_1m"] > 0
                    filtered = composite[composite.index.isin(cur_snap[mom_ok].index)]
                    top = filtered.nlargest(n_stocks) if len(filtered) >= n_stocks else composite.nlargest(n_stocks)
                else:
                    top = composite.nlargest(n_stocks)

                # 비중 계산 (역변동성 ON이면)
                _rec_weights = {}
                if _use_inv_vol and "Volatility_30d" in cur_snap.columns:
                    vols = {}
                    for t in top.index:
                        v = cur_snap.loc[t, "Volatility_30d"] if t in cur_snap.index else np.nan
                        vols[t] = v if (not np.isnan(v) and v > 0) else 0.30
                    inv_v = {t: 1 / v for t, v in vols.items()}
                    total_iv = sum(inv_v.values())
                    _rec_weights = {t: iv / total_iv for t, iv in inv_v.items()}

                rec_rows = []
                for t in top.index:
                    row = {"티커": t}
                    if _use_inv_vol and _rec_weights:
                        row["비중"] = f"{_rec_weights.get(t, 0):.1%}"
                    for col, fmt in [
                        ("Mom_1m","pct"), ("Mom_3m","pct"), ("Mom_6m","pct"), ("Mom_12m","pct"),
                    ]:
                        if col in cur_snap.columns and t in cur_snap.index:
                            v = cur_snap.loc[t, col]
                            row[FEAT_NAMES.get(col, col)] = f"{v:.1%}" if not pd.isna(v) else "N/A"
                    rec_rows.append(row)
                st.dataframe(pd.DataFrame(rec_rows), use_container_width=True,
                             hide_index=True, height=260)
            else:
                st.info("스냅샷 계산 불가")
        else:
            st.info("백테스트를 먼저 실행하세요")

    # ════════════════════════════════════════════════════════
    # SECTION 6: 지표 중요도 TOP 10
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">🔑 지표 중요도 TOP 10</div>', unsafe_allow_html=True)
    if not fimp.empty:
        latest_imp = fimp.iloc[-1].sort_values(ascending=False).head(10)
        names = [FEAT_NAMES.get(f, f) for f in latest_imp.index]
        fig_imp = go.Figure(go.Bar(
            x=latest_imp.values[::-1], y=names[::-1], orientation="h",
            marker=dict(color=latest_imp.values[::-1], colorscale="Viridis"),
            hovertemplate="%{y}<br>중요도: %{x:.4f}<extra></extra>",
        ))
        fig_imp.update_layout(
            **PLOT_CFG, height=260, margin=dict(l=120, r=10, t=5, b=5),
        )
        fig_imp.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # SECTION 7: 리밸런싱별 수익률
    # ════════════════════════════════════════════════════════
    if period_returns:
        st.markdown('<div class="section-hdr">📊 리밸런싱별 수익률</div>', unsafe_allow_html=True)
        dates_str = [h["rebalance_date"].strftime("%Y-%m") for h in rebal_hist]
        colors_bar = ["#2e7d32" if r > 0 else "#c62828" for r in period_returns]
        fig_bar = go.Figure(go.Bar(
            x=dates_str, y=period_returns, marker_color=colors_bar,
            hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
        ))
        fig_bar.add_hline(y=0, line_color="#555", line_width=1)
        fig_bar.add_hline(y=avg_ret, line_dash="dash", line_color="#7c4dff",
                          annotation_text=f"평균 {avg_ret:.2%}", annotation_position="top right")
        fig_bar.update_layout(**PLOT_CFG, height=220, yaxis_tickformat=".1%",
                              margin=dict(l=40, r=10, t=10, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 1 ── 성과 비교
# ═══════════════════════════════════════════════════════════

def tab_performance(results: dict, benchmarks: dict, price_data: dict, rf: float = 0.03):
    pd_ = results["port_dates"]
    pv_ = results["port_values"]

    if len(pd_) < 2:
        st.warning("백테스트 기간이 짧아 성과 데이터가 부족합니다.")
        return

    # 실제 운용 시작 시점 = 첫 리밸런싱 날짜 (학습 기간 이후)
    rebal_hist = results.get("rebal_hist", [])
    if rebal_hist:
        trade_start = rebal_hist[0]["rebalance_date"]
    else:
        trade_start = pd.Timestamp(pd_[0])

    # 성과 지표: 운용 시작 시점 이후만 계산 (학습 직선 구간 제외)
    port_period = pd.Series(pv_, index=pd.DatetimeIndex(pd_))
    port_active = port_period[port_period.index >= trade_start]
    if len(port_active) >= 2:
        # 운용 시작 시점 기준 1.0으로 재정규화
        port_active = port_active / port_active.iloc[0]
    metrics_all = [calc_metrics(port_active, "🤖 AI 전략", rf=rf)]

    # 차트용: 일단위 시리즈 (운용 시작 시점부터)
    port_daily = build_daily_portfolio(results, price_data)
    port_daily_active = port_daily[port_daily.index >= trade_start]
    if len(port_daily_active) >= 2:
        port_daily_active = port_daily_active / port_daily_active.iloc[0]

    col_chart, col_metrics = st.columns([7, 3])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_daily_active.index, y=port_daily_active.values,
        name="🤖 AI 전략",
        line=dict(color="#7c4dff", width=3),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra>AI 전략</extra>",
    ))

    # 벤치마크도 운용 시작 시점 기준 정규화 (공정한 비교)
    colors = {"SPY": "#ff6b35", "QQQ": "#00c9a7", "TQQQ": "#ffd700"}
    for tk, label in BENCHMARKS.items():
        if tk in benchmarks:
            ns = norm_series(benchmarks[tk], trade_start)
            if len(ns) > 1:
                fig.add_trace(go.Scatter(
                    x=ns.index, y=ns.values, name=label,
                    line=dict(color=colors.get(tk, "#888"), width=2),
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:.3f}}<extra>{label}</extra>",
                ))
                metrics_all.append(calc_metrics(ns, label, rf=rf))

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

    ic_mean  = ic_df["IC"].mean()
    ic_std   = ic_df["IC"].std()
    ic_ir    = ic_mean / ic_std if ic_std > 0 else 0
    pos_rate = (ic_df["IC"] > 0).mean()

    # 턴오버 통계 (rebal_hist에서 추출)
    rebal_hist = results.get("rebal_hist", [])
    turnover_vals = [h["turnover"] for h in rebal_hist if "turnover" in h]
    avg_turnover  = np.mean(turnover_vals) if turnover_vals else np.nan
    # 연간 턴오버율: 리밸런싱 기간당 턴오버 × 연간 리밸런싱 횟수
    n_rebal_per_year = 12 / max(results.get("rebal_m", 1), 1) if "rebal_m" in results else len(turnover_vals) / max((ic_df["date"].iloc[-1] - ic_df["date"].iloc[0]).days / 365.25, 0.1)
    annual_turnover = avg_turnover * n_rebal_per_year if not np.isnan(avg_turnover) else np.nan

    def _mcol(v): return "pos" if v > 0.03 else ("neg" if v < 0 else "neu")
    def _tcol(v): return "pos" if v < 0.5 else ("neg" if v > 0.8 else "neu")  # 낮을수록 좋음

    # ── IC 핵심 지표 ──────────────────────────────────────
    st.markdown('<div class="section-hdr">📐 IC 핵심 지표</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "평균 IC",     ic_mean),
        (c2, "IC IR",      ic_ir),
        (c3, "IC 표준편차", ic_std),
        (c4, "양(+)IC 비율", pos_rate),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value {_mcol(val)}">{val:.3f}</div>
        </div>""", unsafe_allow_html=True)

    # ── 턴오버 지표 ───────────────────────────────────────
    st.markdown('<div class="section-hdr">🔄 턴오버율 (거래 회전율)</div>', unsafe_allow_html=True)
    with st.expander("📖 턴오버율이란?", expanded=False):
        st.markdown("""
| 지표 | 정의 | 기준 |
|------|------|------|
| **평균 턴오버** | 리밸런싱 1회당 교체되는 종목 비율 | 낮을수록 거래비용 유리 |
| **연간 턴오버** | 평균 턴오버 × 연간 리밸런싱 횟수 | 1.0 = 포트폴리오 1회 완전 교체 |

- 턴오버 50% = 리밸런싱마다 절반의 종목이 교체됨
- 연간 턴오버가 높을수록 실제 거래비용 부담이 커짐
        """)

    ct1, ct2, ct3 = st.columns(3)
    ct1.markdown(f"""<div class="metric-box">
        <div class="metric-label">평균 턴오버 (1회)</div>
        <div class="metric-value {_tcol(avg_turnover) if not np.isnan(avg_turnover) else 'neu'}">{avg_turnover:.1%}</div>
    </div>""", unsafe_allow_html=True)
    ct2.markdown(f"""<div class="metric-box">
        <div class="metric-label">연간 턴오버</div>
        <div class="metric-value {_tcol(annual_turnover/2) if not np.isnan(annual_turnover) else 'neu'}">{annual_turnover:.1f}x</div>
    </div>""", unsafe_allow_html=True)
    ct3.markdown(f"""<div class="metric-box">
        <div class="metric-label">분석 기간 수</div>
        <div class="metric-value neu">{len(turnover_vals)}회</div>
    </div>""", unsafe_allow_html=True)

    # 턴오버 추이 차트
    if turnover_vals:
        to_dates = [h["rebalance_date"] for h in rebal_hist if "turnover" in h]
        fig_to = go.Figure(go.Bar(
            x=to_dates, y=turnover_vals,
            marker_color=["#ff6b35" if v > 0.6 else "#00c9a7" for v in turnover_vals],
            hovertemplate="%{x|%Y-%m}<br>턴오버: %{y:.1%}<extra></extra>",
        ))
        fig_to.add_hline(y=avg_turnover, line_dash="dash", line_color="#7c4dff",
                         annotation_text=f"평균 {avg_turnover:.1%}", annotation_position="top right")
        fig_to.update_layout(**PLOT_CFG, height=220, title="리밸런싱별 턴오버율",
                             yaxis_tickformat=".0%")
        st.plotly_chart(fig_to, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 모델별 IC 비교 (Ver3.8) ──────────────────────────
    is_ensemble = results.get("use_ensemble", False)
    if is_ensemble and "IC_RF" in ic_df.columns:
        st.markdown('<div class="section-hdr">🤖 모델별 IC 비교</div>', unsafe_allow_html=True)

        # 모델별 평균 IC 요약
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, lbl, key, clr in [
            (mc1, "앙상블 (평균)", "IC",       "#7c4dff"),
            (mc2, "RF",           "IC_RF",    "#7c4dff"),
            (mc3, "XGBoost",      "IC_XGB",   "#ff6b35"),
            (mc4, "LightGBM",     "IC_LGBM",  "#00c9a7"),
        ]:
            if key in ic_df.columns:
                v = ic_df[key].mean()
                col.markdown(f"""<div class="metric-box">
                    <div class="metric-label">{lbl} 평균 IC</div>
                    <div class="metric-value {'pos' if v > 0.03 else ('neg' if v < 0 else 'neu')}">{v:.4f}</div>
                </div>""", unsafe_allow_html=True)

        # 모델별 IC 추이 라인 차트
        fig_mic = go.Figure()
        for key, name, color in [
            ("IC",      "앙상블",    "#7c4dff"),
            ("IC_RF",   "RF",       "#9575cd"),
            ("IC_XGB",  "XGBoost",  "#ff6b35"),
            ("IC_LGBM", "LightGBM", "#00c9a7"),
        ]:
            if key in ic_df.columns:
                fig_mic.add_trace(go.Scatter(
                    x=ic_df["date"], y=ic_df[key],
                    name=name, line=dict(color=color, width=2 if key == "IC" else 1.5),
                    opacity=1.0 if key == "IC" else 0.7,
                    hovertemplate=f"{name}<br>%{{x|%Y-%m}}: %{{y:.4f}}<extra></extra>",
                ))
        fig_mic.add_hline(y=0, line_color="#888", line_width=1)
        fig_mic.update_layout(**PLOT_CFG, height=320, title="리밸런싱별 모델 IC 비교",
                              hovermode="x unified")
        st.plotly_chart(fig_mic, use_container_width=True)

        # 모델별 누적 IC 비교
        fig_cum_mic = go.Figure()
        for key, name, color in [
            ("IC",      "앙상블",    "#7c4dff"),
            ("IC_RF",   "RF",       "#9575cd"),
            ("IC_XGB",  "XGBoost",  "#ff6b35"),
            ("IC_LGBM", "LightGBM", "#00c9a7"),
        ]:
            if key in ic_df.columns:
                fig_cum_mic.add_trace(go.Scatter(
                    x=ic_df["date"], y=ic_df[key].cumsum(),
                    name=name, line=dict(color=color, width=2 if key == "IC" else 1.5),
                ))
        fig_cum_mic.update_layout(**PLOT_CFG, height=280, title="누적 IC 비교 (모델별)")
        st.plotly_chart(fig_cum_mic, use_container_width=True)

        # 모델별 양(+)IC 비율 · IC IR 테이블
        model_stats = []
        for key, name in [("IC", "앙상블"), ("IC_RF", "RF"), ("IC_XGB", "XGBoost"), ("IC_LGBM", "LightGBM")]:
            if key not in ic_df.columns:
                continue
            s = ic_df[key].dropna()
            m, sd = s.mean(), s.std()
            model_stats.append({
                "모델": name,
                "평균 IC": f"{m:.4f}",
                "IC 표준편차": f"{sd:.4f}",
                "IC IR": f"{m/sd:.3f}" if sd > 0 else "N/A",
                "양(+)IC 비율": f"{(s > 0).mean():.1%}",
            })
        if model_stats:
            st.dataframe(pd.DataFrame(model_stats), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # IC 막대 + 누적 IC (앙상블 기준)
    st.markdown('<div class="section-hdr">📈 IC 추이 (앙상블)</div>', unsafe_allow_html=True)
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

    is_ensemble = results.get("use_ensemble", False)
    fimp_rf   = results.get("fimp_rf_df",   pd.DataFrame())
    fimp_xgb  = results.get("fimp_xgb_df",  pd.DataFrame())
    fimp_lgbm = results.get("fimp_lgbm_df", pd.DataFrame())

    # ── 모델 설명 ─────────────────────────────────────────
    with st.expander("📖 사용 모델 설명", expanded=False):
        st.markdown("""
| 모델 | 특징 | 강점 | 약점 |
|------|------|------|------|
| **Random Forest (RF)** | 여러 결정 트리의 평균 예측. 각 트리는 랜덤 데이터·피처 조합으로 학습. | 과적합에 강건, 이상치에 안정적, 해석 용이 | 비선형 패턴 포착력이 부스팅 대비 약함 |
| **XGBoost (XGB)** | 이전 트리의 오차를 다음 트리가 보정하는 그래디언트 부스팅. | 높은 예측 정확도, 복잡한 패턴 학습, 정규화 내장 | 과적합 위험, 하이퍼파라미터 민감 |
| **LightGBM (LGBM)** | XGBoost와 유사하나 리프 기반 분할로 더 빠르고 메모리 효율적. | 대규모 데이터에 최적, 범주형 피처 지원, 학습 속도 | 소규모 데이터에서 과적합 가능 |

**앙상블 효과**: 3모델이 서로 다른 방식으로 데이터를 해석하므로, 한 모델이 틀려도 나머지가 보완합니다.
RF는 안정적 기반을 제공하고, XGBoost/LightGBM은 세밀한 패턴을 포착합니다.
        """)

    avg_imp = fimp.mean().sort_values(ascending=False)
    top_n   = st.slider("표시할 지표 수", 5, min(25, len(fimp.columns)), 10, key="importance_top_n")
    top_f   = avg_imp.head(top_n).index.tolist()

    palette = px.colors.qualitative.Plotly

    # ── 모델별 피처 중요도 비교 (Ver3.8) ──────────────────
    if is_ensemble and not fimp_rf.empty and not fimp_xgb.empty:
        st.markdown('<div class="section-hdr">🤖 모델별 피처 중요도 비교 (TOP 15)</div>',
                    unsafe_allow_html=True)
        top15 = avg_imp.head(15).index.tolist()

        # 각 모델 중요도를 합계 1로 정규화 (스케일 통일)
        def _norm_imp(df, cols):
            if df.empty:
                return pd.Series(dtype=float)
            avg = df[cols].mean()
            total = df.mean().sum()  # 전체 피처 합계로 정규화
            return avg / total if total > 0 else avg

        rf_norm   = _norm_imp(fimp_rf,   top15)
        xgb_norm  = _norm_imp(fimp_xgb,  top15)
        lgbm_norm = _norm_imp(fimp_lgbm, top15)

        labels = [FEAT_NAMES.get(f, f) for f in top15]
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="RF",      x=labels, y=rf_norm.values,   marker_color="#7c4dff"))
        fig_cmp.add_trace(go.Bar(name="XGBoost",  x=labels, y=xgb_norm.values,  marker_color="#ff6b35"))
        fig_cmp.add_trace(go.Bar(name="LightGBM", x=labels, y=lgbm_norm.values, marker_color="#00c9a7"))
        fig_cmp.update_layout(
            **PLOT_CFG, height=400, barmode="group",
            xaxis_tickangle=-45, yaxis_title="정규화 중요도 (비중)",
            yaxis_tickformat=".1%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # 모델별 그룹 비중 비교
        st.markdown('<div class="section-hdr">📊 모델별 그룹 비중 비교</div>', unsafe_allow_html=True)
        group_data = []
        for name, df in [("RF", fimp_rf), ("XGBoost", fimp_xgb), ("LightGBM", fimp_lgbm)]:
            if df.empty:
                continue
            avg = df.mean()
            total = avg.sum()
            grp = {}
            for f, v in avg.items():
                g = FEATURE_META.get(f, {}).get("group", "기타")
                grp[g] = grp.get(g, 0) + v
            for g, v in grp.items():
                group_data.append({"모델": name, "그룹": g, "비중": v / total if total > 0 else 0})
        if group_data:
            grp_df = pd.DataFrame(group_data)
            fig_grp = px.bar(grp_df, x="모델", y="비중", color="그룹",
                             barmode="stack", color_discrete_sequence=px.colors.qualitative.Set2)
            fig_grp.update_layout(**PLOT_CFG, height=350, yaxis_tickformat=".0%")
            st.plotly_chart(fig_grp, use_container_width=True)

    # 앙상블 중요도 추이 라인 그래프
    st.markdown('<div class="section-hdr">📈 앙상블 주요 지표 중요도 추이</div>',
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
# TAB 6 ── 성과 추적
# ═══════════════════════════════════════════════════════════

def tab_tracking(results: dict, price_data: dict):
    """과거 추천 종목의 실제 성과 추적 — 기간별 수익률·승률·종목별 통계."""
    st.markdown('<div class="section-hdr">📊 추천 종목 성과 추적</div>', unsafe_allow_html=True)

    hist = results.get("rebal_hist", [])
    if not hist:
        st.info("백테스트를 먼저 실행하세요.")
        return

    # ── 기간별 요약 테이블 빌드 ───────────────────────────
    rows = []
    ticker_stats: dict = {}  # {ticker: {wins, total, returns}}

    for h in hist:
        tdf = h.get("ticker_df", pd.DataFrame())
        avg_pred = tdf["예측수익률"].mean() if not tdf.empty and "예측수익률" in tdf.columns else np.nan
        avg_actual = tdf["실제수익률"].mean() if not tdf.empty and "실제수익률" in tdf.columns else np.nan

        # 종목 단위 집계
        if not tdf.empty and "실제수익률" in tdf.columns and "ticker" in tdf.columns:
            ticker_col = tdf["ticker"]
            actual_col = tdf["실제수익률"]
            for t, act in zip(ticker_col, actual_col):
                if pd.isna(act):
                    continue
                if t not in ticker_stats:
                    ticker_stats[t] = {"wins": 0, "total": 0, "returns": []}
                ticker_stats[t]["total"] += 1
                ticker_stats[t]["returns"].append(act)
                if act > 0:
                    ticker_stats[t]["wins"] += 1

        # 기간 내 종목 승률 (실제 수익률 양수 비율)
        period_wins = 0
        period_total = 0
        if not tdf.empty and "실제수익률" in tdf.columns:
            valid = tdf["실제수익률"].dropna()
            period_wins = int((valid > 0).sum())
            period_total = len(valid)

        rows.append({
            "날짜":            h["rebalance_date"].strftime("%Y-%m-%d"),
            "보유기간":        h["holding_period"],
            "선정종목":        " / ".join(h["selected"]),
            "예측수익(평균)":  avg_pred,
            "실제수익률":      h["port_return"],
            "종목승률":        f"{period_wins}/{period_total}" if period_total else "N/A",
            "IC":              h.get("ic", np.nan),
            "결과":            "✅ 수익" if h["port_return"] > 0 else "❌ 손실",
        })

    summary_df = pd.DataFrame(rows)

    # ── 전체 요약 지표 ─────────────────────────────────────
    period_returns = [h["port_return"] for h in hist]
    wins_total   = sum(1 for r in period_returns if r > 0)
    win_rate     = wins_total / len(period_returns) if period_returns else 0
    avg_ret      = float(np.mean(period_returns)) if period_returns else 0.0
    best_ret     = float(np.max(period_returns))  if period_returns else 0.0
    worst_ret    = float(np.min(period_returns))  if period_returns else 0.0
    cumulative   = float(np.prod([1 + r for r in period_returns]) - 1)

    st.markdown('<div class="section-hdr">📈 전체 성과 요약</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, lbl, val, fmt, color_rule in [
        (c1, "리밸런싱 승률",   win_rate,   "pct",  "pos" if win_rate > 0.5 else "neg"),
        (c2, "기간 평균 수익",  avg_ret,    "pct",  "pos" if avg_ret > 0 else "neg"),
        (c3, "누적 수익률",     cumulative, "pct",  "pos" if cumulative > 0 else "neg"),
        (c4, "최고 수익 기간",  best_ret,   "pct",  "pos"),
        (c5, "최저 수익 기간",  worst_ret,  "pct",  "neg"),
    ]:
        v_str = f"{val:.2%}" if fmt == "pct" else f"{val:.2f}"
        col.markdown(f"""<div class="metric-box">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value {color_rule}">{v_str}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 기간별 수익률 막대 차트 ────────────────────────────
    st.markdown('<div class="section-hdr">📊 리밸런싱 기간별 실제 수익률</div>', unsafe_allow_html=True)
    dates_str = [h["rebalance_date"].strftime("%Y-%m-%d") for h in hist]
    colors_bar = ["#2e7d32" if r > 0 else "#c62828" for r in period_returns]

    fig_bar = go.Figure(go.Bar(
        x=dates_str,
        y=period_returns,
        marker_color=colors_bar,
        hovertemplate="%{x}<br>수익률: %{y:.2%}<extra></extra>",
    ))
    fig_bar.add_hline(y=0, line_color="#555", line_width=1)
    fig_bar.add_hline(y=avg_ret, line_dash="dash", line_color="#7c4dff", line_width=1.5,
                      annotation_text=f"평균 {avg_ret:.2%}", annotation_position="top right")
    fig_bar.update_layout(
        **PLOT_CFG, height=320,
        xaxis_title="리밸런싱 날짜", yaxis_title="수익률",
        yaxis_tickformat=".1%",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 누적 수익률 추이 ───────────────────────────────────
    st.markdown('<div class="section-hdr">📈 추천 전략 누적 수익률 추이</div>', unsafe_allow_html=True)
    cum_vals = [1.0]
    for r in period_returns:
        cum_vals.append(cum_vals[-1] * (1 + r))
    cum_dates = [h["rebalance_date"].strftime("%Y-%m-%d") for h in hist]
    cum_dates = [hist[0]["rebalance_date"].strftime("%Y-%m-%d")] + \
                [h["next_date"].strftime("%Y-%m-%d") for h in hist]

    fig_cum = go.Figure(go.Scatter(
        x=cum_dates, y=cum_vals,
        fill="tozeroy", line=dict(color="#7c4dff", width=2),
        fillcolor="rgba(124,77,255,0.12)",
        hovertemplate="%{x}<br>누적: %{y:.3f}<extra></extra>",
    ))
    fig_cum.add_hline(y=1.0, line_dash="dash", line_color="#888", line_width=1)
    fig_cum.update_layout(
        **PLOT_CFG, height=280,
        xaxis_title="날짜", yaxis_title="누적 수익 (1.0 = 시작)",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── 종목별 성과 통계 ───────────────────────────────────
    st.markdown('<div class="section-hdr">🏆 종목별 성과 통계</div>', unsafe_allow_html=True)
    if ticker_stats:
        ticker_rows = []
        for t, stat in ticker_stats.items():
            avg_r = float(np.mean(stat["returns"])) if stat["returns"] else 0.0
            wr    = stat["wins"] / stat["total"] if stat["total"] else 0.0
            ticker_rows.append({
                "티커":     t,
                "선정 횟수": stat["total"],
                "승률":     wr,
                "평균 수익률": avg_r,
                "최고 수익": float(np.max(stat["returns"])),
                "최저 수익": float(np.min(stat["returns"])),
            })
        tstat_df = pd.DataFrame(ticker_rows).sort_values("선정 횟수", ascending=False)

        # 표시용 포맷
        disp_tstat = tstat_df.copy()
        for c in ["승률", "평균 수익률", "최고 수익", "최저 수익"]:
            disp_tstat[c] = disp_tstat[c].apply(lambda x: f"{x:.2%}")
        st.dataframe(disp_tstat, use_container_width=True, hide_index=True, height=320)

        # 종목별 평균 수익률 상위 바 차트
        top_tickers = tstat_df.nlargest(min(20, len(tstat_df)), "평균 수익률")
        clr = ["#2e7d32" if v > 0 else "#c62828" for v in top_tickers["평균 수익률"]]
        fig_tk = go.Figure(go.Bar(
            x=top_tickers["티커"],
            y=top_tickers["평균 수익률"],
            marker_color=clr,
            hovertemplate="%{x}<br>평균 수익률: %{y:.2%}<extra></extra>",
        ))
        fig_tk.add_hline(y=0, line_color="#555", line_width=1)
        fig_tk.update_layout(
            **PLOT_CFG, height=300,
            title="종목별 평균 수익률 TOP 20",
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig_tk, use_container_width=True)

    # ── 기간별 상세 테이블 ────────────────────────────────
    st.markdown('<div class="section-hdr">📋 리밸런싱 기간별 상세 기록</div>', unsafe_allow_html=True)
    disp_summary = summary_df.copy()
    for c in ["예측수익(평균)", "실제수익률"]:
        disp_summary[c] = disp_summary[c].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    disp_summary["IC"] = disp_summary["IC"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    st.dataframe(disp_summary, use_container_width=True, hide_index=True, height=400)


# ═══════════════════════════════════════════════════════════
# TAB 7 ── 실시간 AI 추천
# ═══════════════════════════════════════════════════════════

def tab_realtime(price_data: dict, fund_map: dict, tech_map: dict,
                 results: dict, n_stocks: int, pit_map: dict = None,
                 spy_close: pd.Series = None, vix_close: pd.Series = None,
                 sector_map: dict = None):
    st.markdown('<div class="section-hdr">🔴 AI 추천 종목</div>', unsafe_allow_html=True)

    fimp = results.get("fimp_df", pd.DataFrame())
    if fimp.empty:
        st.info("백테스트를 먼저 실행하세요.")
        return

    # ── 날짜 선택 UI ──────────────────────────────────────
    all_dates_max = [ohlcv.index.max().date() for ohlcv in price_data.values() if len(ohlcv) > 0]
    all_dates_min = [ohlcv.index.min().date() for ohlcv in price_data.values() if len(ohlcv) > 0]
    max_avail = max(all_dates_max) if all_dates_max else datetime.today().date()
    min_avail = min(all_dates_min) if all_dates_min else datetime.today().date() - timedelta(days=365)
    # 백테스트 종료일 다음날을 기본값으로 설정
    _saved_cfg = st.session_state.get("cfg") or {}
    _bt_end = _saved_cfg.get("end")
    if _bt_end:
        _bt_end_date = _bt_end.date() if hasattr(_bt_end, "date") else _bt_end
        default_date = min(_bt_end_date + timedelta(days=1), max_avail)
    else:
        default_date = min(datetime.today().date(), max_avail)

    col_date, col_info = st.columns([2, 5])
    with col_date:
        sel_date = st.date_input(
            "분석 기준일",
            value=default_date,
            min_value=min_avail,
            max_value=datetime.today().date(),
            key="realtime_date",
        )
    with col_info:
        is_today = (sel_date == datetime.today().date())
        if is_today:
            st.info("💡 오늘 날짜 기준 최신 지표로 예측합니다.")
        else:
            st.info(f"💡 **{sel_date}** 기준 지표로 예측합니다. 예측 모델은 백테스트의 마지막 학습 모델을 사용합니다.")

    today = pd.Timestamp(sel_date)

    with st.spinner("최신 지표 계산 중..."):
        cur_snap = build_snapshot_df(list(price_data.keys()), tech_map, fund_map, today,
                                     pit_map=pit_map, price_data=price_data)
        if not cur_snap.empty:
            cur_snap = enrich_snapshot(cur_snap, today,
                                      spy_close=spy_close, vix_close=vix_close,
                                      sector_map=sector_map)

    if cur_snap.empty:
        st.warning("현재 지표 데이터를 계산할 수 없습니다.")
        return

    # ── Ver3.7: 앙상블 예측 (백테스트와 동일 전처리+모델) ──
    model_rf      = results.get("last_model_rf")
    model_xgb     = results.get("last_model_xgb")
    model_lgbm    = results.get("last_model_lgbm")
    last_imputer  = results.get("last_imputer")
    last_avail_cols = results.get("last_avail_cols", [])
    last_all_cols   = results.get("last_all_cols", [])
    last_win_bounds = results.get("last_win_bounds", {})
    last_miss_src   = results.get("last_miss_src", [])
    is_ensemble     = results.get("use_ensemble", False)

    if model_rf is None or last_imputer is None or not last_all_cols:
        st.warning("저장된 모델이 없습니다. 백테스트를 다시 실행해 주세요.")
        return

    avail_cols = [c for c in last_avail_cols if c in cur_snap.columns]
    X_pred = cur_snap[avail_cols].reindex(columns=last_avail_cols)

    # 윈저화 (학습과 동일 기준)
    for col in last_avail_cols:
        if col in last_win_bounds and col in X_pred.columns:
            lo, hi = last_win_bounds[col]
            X_pred[col] = X_pred[col].clip(lo, hi)

    # 결측 플래그 (학습과 동일 피처)
    for mc in last_miss_src:
        X_pred[f"{mc}_miss"] = X_pred[mc].isna().astype(int) if mc in X_pred.columns else 0
    X_pred = X_pred.reindex(columns=last_all_cols)

    X_pred_imp = last_imputer.transform(X_pred)

    # 앙상블 예측: Rank Average
    pred_rf_s = pd.Series(model_rf.predict(X_pred_imp), index=cur_snap.index)
    if is_ensemble and model_xgb is not None and model_lgbm is not None:
        pred_xgb_s  = pd.Series(model_xgb.predict(X_pred_imp),  index=cur_snap.index)
        pred_lgbm_s = pd.Series(model_lgbm.predict(X_pred_imp), index=cur_snap.index)
        rank_rf   = pred_rf_s.rank(ascending=False)
        rank_xgb  = pred_xgb_s.rank(ascending=False)
        rank_lgbm = pred_lgbm_s.rank(ascending=False)
        avg_rank  = (rank_rf + rank_xgb + rank_lgbm) / 3
        composite = -avg_rank  # 높을수록 좋은 종목
        st.caption("🤖 앙상블 예측 (Rank Average: RF + XGBoost + LightGBM)")
    else:
        composite = pred_rf_s
        st.caption("🤖 단일 RF 예측")

    # ── 모델 합의도 계산 (Ver3.8) ────────────────────────
    consensus = {}
    if is_ensemble and model_xgb is not None and model_lgbm is not None:
        rf_top   = set(pred_rf_s.nlargest(n_stocks).index)
        xgb_top  = set(pred_xgb_s.nlargest(n_stocks).index)
        lgbm_top = set(pred_lgbm_s.nlargest(n_stocks).index)
        for t in cur_snap.index:
            cnt = sum([t in rf_top, t in xgb_top, t in lgbm_top])
            consensus[t] = cnt  # 0~3

    # 정규화 (0~1 표시용, 음수 포함 가능)
    c_min, c_max = composite.min(), composite.max()
    composite_norm = (composite - c_min) / (c_max - c_min + 1e-9)

    # 모멘텀 필터 (설정에 따라 ON/OFF)
    _cfg = st.session_state.get("cfg") or {}
    _use_mom = _cfg.get("use_mom_filter", True)
    if _use_mom and "Mom_1m" in cur_snap.columns:
        mom_ok = cur_snap["Mom_1m"] > 0
        filtered = composite[composite.index.isin(cur_snap[mom_ok].index)]
        top_recs = filtered.nlargest(n_stocks) if len(filtered) >= n_stocks else composite.nlargest(n_stocks)
    else:
        top_recs = composite.nlargest(n_stocks)
    all_recs  = composite.sort_values(ascending=False)  # 전체 종목 정렬

    # 표시용 점수는 정규화 버전 사용
    composite = composite_norm

    # ── 모델 설명 ─────────────────────────────────────────
    with st.expander("📖 AI 모델 설명 · 합의도 해석", expanded=False):
        st.markdown("""
| 모델 | 역할 |
|------|------|
| **Random Forest** | 안정적 기반. 여러 결정 트리의 평균으로 과적합에 강건. |
| **XGBoost** | 세밀한 패턴 포착. 이전 트리의 오차를 반복 보정. |
| **LightGBM** | 빠른 학습 + 대규모 데이터 최적화. 리프 기반 분할. |

**합의도 (★)**
- ★★★ : 3모델 모두 TOP N에 선정 → **가장 신뢰도 높음**
- ★★ : 2모델이 TOP N에 선정 → 유력 후보
- ★ : 1모델만 선정 → 특정 모델의 독자적 판단, 주의 필요
- (빈칸) : 추천 대상 아님

**활용법**: ★★★ 종목을 우선 검토하고, ★ 종목은 추가 확인 후 결정하세요.
        """)

    # ── Ver4.2: 추천 종목 비중 계산 ─────────────────────
    rec_weights = {}
    if "Volatility_30d" in cur_snap.columns:
        vols = {}
        for t in top_recs.index:
            v = cur_snap.loc[t, "Volatility_30d"] if t in cur_snap.index else np.nan
            vols[t] = v if (not np.isnan(v) and v > 0) else 0.30
        inv_vols = {t: 1 / v for t, v in vols.items()}
        total_inv = sum(inv_vols.values())
        rec_weights = {t: iv / total_inv for t, iv in inv_vols.items()}
    else:
        w = 1 / max(len(top_recs), 1)
        rec_weights = {t: w for t in top_recs.index}

    # ── 전체 종목 상세 테이블 ──────────────────────────────
    st.markdown(f"**{today.strftime('%Y-%m-%d')} 기준 전체 분석 종목 AI 점수 순위 ({len(all_recs)}개)**")
    rec_rows = []
    for rank, (t, score) in enumerate(all_recs.items(), 1):
        stars = consensus.get(t, 0)
        star_str = "★" * stars if stars > 0 else ""
        row = {"순위": rank, "티커": t, "AI 점수": round(score, 3),
               "추천": "O" if t in top_recs.index else "",
               "비중": f"{rec_weights[t]:.1%}" if t in rec_weights else "",
               "합의도": star_str if consensus else ""}
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
    top_imp = fimp.iloc[-1].sort_values(ascending=False).head(15)
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

def render_topbar(sp1500_df: pd.DataFrame, all_sectors: list) -> dict:
    """사이드바 대신 페이지 상단 가로 배치 설정 UI."""

    ALL_CAP_TIERS = ["Large Cap", "Mid Cap", "Small Cap"]

    with st.expander("⚙️ 백테스트 설정 펼치기 / 접기", expanded=True):

        # ── Row 0: Cap Tier 선택 ────────────────────────────
        st.markdown("**📊 시가총액 구간 선택**")
        sel_cap = st.multiselect(
            "Cap Tier",
            ALL_CAP_TIERS,
            default=["Large Cap"],
            label_visibility="collapsed",
            help="Large Cap = S&P 500 · Mid Cap = S&P 400 · Small Cap = S&P 600",
        )
        if not sel_cap:
            sel_cap = ["Large Cap"]

        cap_filtered = sp1500_df[sp1500_df["cap_tier"].isin(sel_cap)]

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

        universe = cap_filtered[cap_filtered["sector"].isin(sel_sectors)]["ticker"].tolist()

        tier_counts = cap_filtered[cap_filtered["sector"].isin(sel_sectors)]["cap_tier"].value_counts()
        tier_str = " · ".join([f"{t}: {c}개" for t, c in tier_counts.items()])
        st.caption(f"선택된 유니버스: **{len(universe)}**개 종목  ({tier_str})")

        st.markdown("---")

        # ── Row 2: 핵심 파라미터 ─────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rebal_m = st.slider(
                "리밸런싱 기간 (개월)", 1, 12, 1, 1,
                help="포트폴리오 리밸런싱 주기",
            )
        with c2:
            rolling_w = st.slider(
                "롤링 학습 윈도우 (기간 수)", 2, 24, 12, 1,
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

        # ── 고급 설정 ─────────────────────────────────────
        st.markdown("---")
        st.markdown("**🔬 고급 설정**")
        ac1, ac2 = st.columns(2)
        with ac1:
            min_dollar_vol = st.number_input(
                "최소 일평균 거래대금 ($)",
                min_value=0, max_value=100_000_000,
                value=10_000_000, step=500_000,
                format="%d",
                help="20일 평균 거래대금 기준, 미달 종목 제외 (0 = 필터 없음)",
            )
        with ac2:
            exec_price = st.selectbox(
                "체결가 가정",
                ["당일 종가", "T+1 시가"],
                index=1,
                help="당일 종가: 리밸런싱일 종가에 매매 / T+1 시가: 다음 영업일 시가에 매매 (더 현실적)",
            )
        use_next_open = (exec_price == "T+1 시가")

        ck1, ck2, ck3 = st.columns(3)
        with ck1:
            use_surv_fix = st.checkbox(
                "생존자 편향 보정",
                value=("Large Cap" in sel_cap),
                help="S&P 500 변경 이력으로 리밸런싱 시점의 실제 구성 종목 복원 (Large Cap 전용)",
                disabled=("Large Cap" not in sel_cap),
            )
        with ck2:
            use_turnover_buffer = st.checkbox(
                "턴오버 버퍼 (보유 종목 우선)",
                value=True,
                help="이미 보유 중인 종목에 5% 가산점을 부여합니다. "
                     "순위가 약간 밀려도 기존 보유 종목을 유지하여 불필요한 매매를 줄입니다. "
                     "효과: 턴오버 30~50% 감소 → 거래비용 절약, 노이즈성 교체 방지.",
            )
        with ck3:
            use_ensemble = st.checkbox(
                "앙상블 모델 (RF + XGBoost + LightGBM)",
                value=False,
                help="3개 모델의 예측을 평균하여 안정적인 종목 선정. 해제 시 RF 단일 모델 사용.",
            )

        ck4, ck5 = st.columns(2)
        with ck4:
            use_mom_filter = st.checkbox(
                "모멘텀 필터 (하락 종목 제외)",
                value=False,
                help="최근 1개월 수익률이 마이너스인 종목을 추천 후보에서 제외합니다. "
                     "AI 점수가 높아도 현재 하락 추세인 종목은 선정하지 않습니다. "
                     "효과: 하락 종목 회피 → 선정 초과수익·Profit Factor 개선.",
            )
        with ck5:
            use_inv_vol_weight = st.checkbox(
                "역변동성 가중 (Risk-Parity)",
                value=False,
                help="변동성이 낮은 종목에 더 많은 비중을 배분합니다. "
                     "예: 변동성 15% 종목에 28%, 변동성 50% 종목에 11%. "
                     "효과: MDD 축소, Sharpe 개선. 단, 고변동 종목의 급등 수익은 줄어들 수 있습니다. "
                     "추천 탭에서 종목별 투자 비중(%)을 함께 표시합니다.",
            )

        # ── 날짜 설정 ────────────────────────────────────
        st.markdown("---")
        MIN_TEST    = 5
        auto_months = (rolling_w + MIN_TEST) * rebal_m + 12
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

        # ── 실행 버튼 ────────────────────────────────────
        run_btn = st.button("🚀 백테스트 실행", type="primary", use_container_width=True)

    return {
        "cap_tiers":      sel_cap,
        "sectors":        sel_sectors,
        "universe":       universe,
        "rebal_m":        rebal_m,
        "rolling_w":      rolling_w,
        "start":          datetime(sd.year, sd.month, sd.day),
        "end":            datetime(ed.year, ed.month, ed.day),
        "n_stocks":       n_stocks,
        "tc_pct":         tc_pct,
        "run":            run_btn,
        "min_test":       MIN_TEST,
        "min_dollar_vol": min_dollar_vol,
        "use_next_open":  use_next_open,
        "use_surv_fix":   use_surv_fix and ("Large Cap" in sel_cap),
        "use_ensemble":          use_ensemble,
        "use_mom_filter":        use_mom_filter,
        "use_turnover_buffer":   use_turnover_buffer,
        "use_inv_vol_weight":    use_inv_vol_weight,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    st.markdown('<div class="main-title">📊 AI Quant Lab 4.2</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">AI 퀀트 백테스팅 & 실시간 추천 · '
        '66개 피처 (CS정규화·시장레짐·VIX·섹터RS 포함) · RF+XGB+LGBM Rank Average · '
        '생존자 편향 보정 · PIT 22개 지표</div>',
        unsafe_allow_html=True,
    )

    # S&P 1500 목록 로드
    with st.spinner("S&P 1500 종목 목록 로드 중... (Large/Mid/Small Cap 합산)"):
        sp1500_df, all_sectors = get_sp1500_info()

    cfg = render_topbar(sp1500_df, all_sectors)

    # ── 세션 상태 초기화 ──────────────────────────────────
    for k in ["results", "benchmarks", "price_data", "fund_map", "tech_map",
              "cfg", "rf_rate", "pit_map", "sp500_changes",
              "spy_close", "vix_close", "sector_map"]:
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
            (c4, "🔴", "실시간", ["특정일 추천 종목", "레이더 차트", "성과 추적 대시보드"]),
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

        # 0. 생존자 편향 보정: S&P 500 변경 이력 로드 + 과거 퇴출 종목 추가
        sp500_changes = None
        current_sp500 = None
        extra_hist_tickers = []
        if cfg.get("use_surv_fix", False):
            update_prog(0.01, "📜 S&P 500 변경 이력 로드 중 (생존자 편향 보정)...")
            sp500_changes = get_sp500_changes()
            current_sp500 = sp1500_df[sp1500_df["cap_tier"] == "Large Cap"]["ticker"].tolist()
            if not sp500_changes.empty:
                # 백테스트 기간 중 퇴출된 종목 → 가격 데이터도 다운로드 필요
                # 단, 사용자가 선택한 섹터에 해당하는 종목만 추가
                bt_start = pd.Timestamp(cfg["start"])
                removed_in_period = sp500_changes[
                    (sp500_changes["date"] >= bt_start) &
                    (sp500_changes["removed_ticker"] != "")
                ]["removed_ticker"].unique().tolist()
                # 섹터 필터: sp1500_df에 있으면 섹터 확인, 없으면 포함 (과거 종목이라 목록에 없을 수 있음)
                sel_sectors = set(cfg.get("sectors", []))
                filtered_removed = []
                for t in removed_in_period:
                    match = sp1500_df[sp1500_df["ticker"] == t]
                    if not match.empty:
                        if match.iloc[0].get("sector", "") in sel_sectors:
                            filtered_removed.append(t)
                    # sp1500_df에 없는 퇴출 종목은 섹터 확인 불가 → 제외 (안전 우선)
                extra_hist_tickers = [t for t in filtered_removed if t not in universe]
                st.session_state.sp500_changes = sp500_changes

        # 1. 가격 데이터
        all_tickers = list(set(universe + extra_hist_tickers))
        update_prog(0.03, f"📡 {len(all_tickers)}개 종목 가격 데이터 다운로드 중...")
        data_start = cfg["start"] - timedelta(days=400)  # 지표 warm-up
        price_data = download_price_data(
            tuple(all_tickers),
            data_start.strftime("%Y-%m-%d"),
            cfg["end"].strftime("%Y-%m-%d"),
        )
        if not price_data:
            st.error("가격 데이터를 불러올 수 없습니다.")
            return

        available = list(price_data.keys())
        n_extra = len([t for t in extra_hist_tickers if t in price_data])
        update_prog(0.12, f"✅ {len(available)}개 종목 ({n_extra}개 역사적 퇴출 종목 포함). 펀더멘털 로드 중...")

        # 2. 펀더멘털 데이터 (.info — EV/EBITDA, FCF, 배당 등 유지)
        fund_map = get_fundamental_yf(tuple(available))
        fund_ok  = sum(1 for v in fund_map.values() if v)
        update_prog(0.18, f"✅ 펀더멘털 완료: {fund_ok}/{len(available)}개. PIT 분기 재무제표 수집 중...")

        # 2-b. PIT 분기 재무제표 수집 (P/E·P/B·P/S·ROE·ROA·마진·성장률 개선)
        # 종목 수에 따라 시간이 걸릴 수 있음 (종목당 ~0.5초)
        update_prog(0.20, f"📋 분기 재무제표 수집 중... ({len(available)}개 · 시간이 걸릴 수 있습니다)")
        pit_map = get_pit_financials(tuple(available))
        pit_ok  = sum(1 for v in pit_map.values() if not v["income"].empty)
        update_prog(0.28, f"✅ PIT 재무제표 완료: {pit_ok}/{len(available)}개. 기술지표 계산 중...")

        # 3. SPY + VIX 다운로드 + 기술지표 사전 계산
        update_prog(0.28, "📡 SPY·VIX 다운로드 + 기술지표 계산 중...")
        spy_close = None
        vix_close = None
        try:
            spy_df = yf.download("SPY", start=data_start.strftime("%Y-%m-%d"),
                                 end=cfg["end"].strftime("%Y-%m-%d"),
                                 auto_adjust=True, progress=False)
            if not spy_df.empty:
                spy_close = spy_df["Close"].squeeze()
        except Exception:
            pass
        try:
            vix_df = yf.download("^VIX", start=data_start.strftime("%Y-%m-%d"),
                                 end=cfg["end"].strftime("%Y-%m-%d"),
                                 auto_adjust=True, progress=False)
            if not vix_df.empty:
                vix_close = vix_df["Close"].squeeze()
        except Exception:
            pass

        # 섹터 맵 구성 (Sector RS 계산용)
        sector_map = {}
        for _, row in sp1500_df.iterrows():
            sector_map[row["ticker"]] = row.get("sector", "Unknown")

        tech_map = {}
        for t, ohlcv in price_data.items():
            try:
                tech_map[t] = calc_all_technical(ohlcv, spy_close=spy_close)
            except Exception:
                pass
        update_prog(0.30, f"📈 기술지표 계산 완료 ({len(tech_map)}종목). 백테스트 시작...")

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
            pit_map=pit_map,
            min_dollar_vol=cfg.get("min_dollar_vol", 0),
            use_next_open=cfg.get("use_next_open", False),
            sp500_changes=sp500_changes,
            current_sp500=current_sp500,
            use_ensemble=cfg.get("use_ensemble", True),
            spy_close=spy_close,
            vix_close=vix_close,
            sector_map=sector_map,
            use_mom_filter=cfg.get("use_mom_filter", True),
            use_turnover_buffer=cfg.get("use_turnover_buffer", False),
            use_inv_vol_weight=cfg.get("use_inv_vol_weight", False),
        )

        # 6. 벤치마크 데이터
        benchmarks = get_benchmark_prices(
            cfg["start"].strftime("%Y-%m-%d"),
            cfg["end"].strftime("%Y-%m-%d"),
        )

        # 7. 무위험수익률 조회 (기간 평균 T-bill)
        rf_rate = get_riskfree_rate(
            cfg["start"].strftime("%Y-%m-%d"),
            cfg["end"].strftime("%Y-%m-%d"),
        )

        # 저장
        st.session_state.results    = results
        st.session_state.rf_rate    = rf_rate
        st.session_state.pit_map    = pit_map
        st.session_state.benchmarks = benchmarks
        st.session_state.price_data = price_data
        st.session_state.fund_map   = fund_map
        st.session_state.tech_map   = tech_map
        st.session_state.spy_close   = spy_close
        st.session_state.vix_close   = vix_close
        st.session_state.sector_map  = sector_map
        st.session_state.cfg         = cfg

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
        rf_rate    = st.session_state.get("rf_rate", 0.03)

        # 무위험수익률 표시
        st.caption(f"무위험수익률 (T-bill 3M 기간평균): **{rf_rate:.2%}**")

        tabs = st.tabs([
            "🏠 요약",
            "📈 성과 비교",
            "🎯 IC 분석 & 턴오버",
            "📋 리밸런싱 히스토리",
            "🔍 지표 중요도",
            "🗺️ 영향력 히트맵",
            "📊 성과 추적",
            "🔴 AI 추천",
        ])

        with tabs[0]:
            tab_summary(results, benchmarks, price_data,
                        fund_map=fund_map, tech_map=tech_map,
                        n_stocks=saved_cfg.get("n_stocks", 5),
                        rf=rf_rate,
                        pit_map=st.session_state.get("pit_map"),
                        spy_close=st.session_state.get("spy_close"),
                        vix_close=st.session_state.get("vix_close"),
                        sector_map=st.session_state.get("sector_map"))
        with tabs[1]:
            tab_performance(results, benchmarks, price_data, rf=rf_rate)
        with tabs[2]:
            tab_ic(results)
        with tabs[3]:
            tab_history(results)
        with tabs[4]:
            tab_importance(results)
        with tabs[5]:
            tab_heatmap(results)
        with tabs[6]:
            tab_tracking(results, price_data)
        with tabs[7]:
            tab_realtime(price_data, fund_map, tech_map, results,
                         saved_cfg.get("n_stocks", 10),
                         pit_map=st.session_state.get("pit_map"),
                         spy_close=st.session_state.get("spy_close"),
                         vix_close=st.session_state.get("vix_close"),
                         sector_map=st.session_state.get("sector_map"))


if __name__ == "__main__":
    main()
