import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS", "0") == "1"

MAX_WORKERS  = 2        if IS_CLOUD else 8
SLEEP_JITTER = (2.0, 4.0) if IS_CLOUD else (0.5, 1.5)   # 재시도 백오프 지터
RETRY_SLEEP  = (4.0, 7.0) if IS_CLOUD else (2.0, 3.5)   # 2차 순차 재시도 대기


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
# 단일 티커 재무 fetch
#  - requests_cache 의존성 제거 (설치 문제 방지)
#  - info 중복 호출 제거 (한 번만 호출, 저장해서 재사용)
#  - 지수 백오프 재시도
# ─────────────────────────────────────────────
def _fetch_one_ticker(ticker: str) -> tuple[str, dict | None]:
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        try:
            tk = yf.Ticker(ticker)

            # info를 한 번만 호출해서 저장 (기존 코드는 두 번 호출)
            info = tk.info or {}
            if not info or len(info) < 5:
                raise ValueError("info 응답 비어 있음")

            def _safe_T(df) -> pd.DataFrame:
                if df is None or (hasattr(df, "empty") and df.empty):
                    return pd.DataFrame()
                return df.T

            # 분기 재무제표 우선, 없으면 연간으로 폴백
            q_fin = tk.quarterly_financials
            a_fin = tk.financials
            if (q_fin is None or q_fin.empty) and (a_fin is None or a_fin.empty):
                return ticker, None  # 재무 데이터 자체가 없으면 스킵

            return ticker, {
                "q_fin": _safe_T(q_fin),
                "q_bal": _safe_T(tk.quarterly_balance_sheet),
                "q_cf":  _safe_T(tk.quarterly_cashflow),
                "a_fin": _safe_T(a_fin),
                "a_bal": _safe_T(tk.balance_sheet),
                "a_cf":  _safe_T(tk.cashflow),
                "info":  info,  # 위에서 받아둔 info 재사용
            }

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep((2 ** (attempt + 1)) + random.uniform(*SLEEP_JITTER))

    return ticker, None


# ─────────────────────────────────────────────
# 재무 데이터 수집 (전체 티커)
#  - submit + as_completed: 한 개 느려도 나머지 계속 처리
#  - 개별/전체 타임아웃으로 무한 대기 방지
#  - 1차 실패 티커 순차 재시도
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_all_financial_source(tickers: list) -> dict:
    data_cache: dict = {}
    failed: list = []

    progress_bar = st.progress(0, text="재무 데이터 수집 중...")
    total = len(tickers)

    # 1차: 병렬 수집
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(_fetch_one_ticker, t): t for t in tickers}

        done_count = 0
        try:
            for future in concurrent.futures.as_completed(future_map, timeout=300):
                t = future_map[future]
                try:
                    ticker, result = future.result(timeout=45)
                    if result:
                        data_cache[ticker] = result
                    else:
                        failed.append(ticker)
                except Exception:
                    failed.append(t)

                done_count += 1
                progress_bar.progress(
                    int(done_count / total * 80),
                    text=f"재무 데이터 수집 중... ({done_count}/{total})"
                )
        except concurrent.futures.TimeoutError:
            # 전체 타임아웃 시 미완료 티커를 failed에 추가
            for f, t in future_map.items():
                if not f.done():
                    failed.append(t)
                    f.cancel()

    # 2차: 실패 티커 순차 재시도
    if failed:
        for idx, ticker in enumerate(failed):
            progress_bar.progress(
                int(80 + idx / max(len(failed), 1) * 18),
                text=f"재시도 중... ({idx+1}/{len(failed)})"
            )
            time.sleep(random.uniform(*RETRY_SLEEP))
            _, result = _fetch_one_ticker(ticker)
            if result:
                data_cache[ticker] = result

    progress_bar.progress(100, text=f"완료: {len(data_cache)}/{total}개 수집")
    time.sleep(0.5)
    progress_bar.empty()

    return data_cache


# ─────────────────────────────────────────────
# 안전한 Series 값 추출 헬퍼
# ─────────────────────────────────────────────
def _safe_get(series: pd.Series, key: str, default=0):
    val = series.get(key, default)
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


# ─────────────────────────────────────────────
# 재무 데이터 Point-in-Time 추출 헬퍼
#  - 분기/연간 데이터 통합 후 기준일 기준 가장 가까운 행 선택
#  - ffill/bfill로 NaN 보완
# ─────────────────────────────────────────────
def _get_pit_row(src: dict, q_key: str, a_key: str, ref_dt: pd.Timestamp) -> pd.Series:
    """기준일(ref_dt) 기준 가장 가까운 재무 데이터 행을 반환."""
    q_df = src.get(q_key, pd.DataFrame())
    a_df = src.get(a_key, pd.DataFrame())

    if q_df.empty and a_df.empty:
        return pd.Series(dtype=float)

    combined = pd.concat([q_df, a_df]).sort_index(ascending=False)
    combined = combined[~combined.index.duplicated(keep="first")]

    # 기준일 + 45일 이내 데이터만 (미래 참조 방지 + 발표 지연 허용)
    valid = combined[combined.index <= (ref_dt + pd.Timedelta(days=45))]
    if valid.empty:
        return pd.Series(dtype=float)

    # 가장 가까운 시점 행 선택
    idx_min = int(np.abs((valid.index - ref_dt).days).argmin())
    row = valid.iloc[idx_min].copy()

    # NaN이 많으면 ffill/bfill로 보완
    if row.isna().sum() > len(row) * 0.5:
        filled = combined.ffill().bfill()
        if not filled.empty and idx_min < len(filled):
            row = filled.iloc[idx_min]

    return row


# ─────────────────────────────────────────────
# ML 피처 추출 (Point-in-Time)
# ─────────────────────────────────────────────
def fetch_ml_data_optimized_pit(
    tickers, ref_date, full_hist_data, source_cache, is_training=True
):
    features_list = []
    ref_dt = pd.to_datetime(ref_date)
    all_level0 = set(full_hist_data.columns.get_level_values(0))

    for ticker in tickers:
        try:
            if ticker not in all_level0:
                continue

            ticker_prices = full_hist_data[ticker].dropna(how="all")
            hist = ticker_prices[ticker_prices.index < ref_dt].tail(252)
            if len(hist) < 252:
                continue

            close_now = float(hist["Close"].iloc[-1])
            src  = source_cache.get(ticker, {})
            info = src.get("info", {})

            # ── Point-in-Time 재무 데이터 ──────────────────────────
            cur  = _get_pit_row(src, "q_fin", "a_fin", ref_dt)
            bal  = _get_pit_row(src, "q_bal", "a_bal", ref_dt)
            cf   = _get_pit_row(src, "q_cf",  "a_cf",  ref_dt)

            # 직전 분기 데이터 (성장률 계산용)
            q_df = src.get("q_fin", pd.DataFrame())
            a_df = src.get("a_fin", pd.DataFrame())
            combined_fin = pd.concat([q_df, a_df]).sort_index(ascending=False)
            combined_fin = combined_fin[~combined_fin.index.duplicated(keep="first")]
            valid_fin    = combined_fin[combined_fin.index <= (ref_dt + pd.Timedelta(days=45))]
            prev = valid_fin.iloc[1] if len(valid_fin) > 1 else cur
            # ─────────────────────────────────────────────────────────

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
            annual_ebitda = ebitda * 4
            total_debt    = _safe_get(bal, "Total Debt")
            cash_eq       = _safe_get(bal, "Cash And Cash Equivalents")
            equity        = _safe_get(bal, "Stockholders Equity", 1) or 1
            ev            = mkt_cap + total_debt - cash_eq

            prev_revenue     = _safe_get(prev, "Total Revenue", 1) or 1
            prev_net_income  = _safe_get(prev, "Net Income",    1) or 1

            close = hist["Close"]

            data = {
                "Ticker":             ticker,
                "P/E":                mkt_cap / (net_income * 4)    if net_income > 0    else 0,
                "P/S":                mkt_cap / (revenue * 4)       if revenue > 0       else 0,
                "P/B":                mkt_cap / equity,
                "P/FCF":              mkt_cap / (fcf * 4)           if fcf > 0           else 0,
                "EV/EBITDA":          ev / annual_ebitda             if annual_ebitda > 0 else 0,
                "FCF_Yield":          (fcf * 4) / mkt_cap           if mkt_cap > 0       else 0,
                "ROE":                net_income / equity,
                "ROA":                net_income / total_assets,
                "Gross_Margin":       gross_profit / revenue         if revenue > 0       else 0,
                "Operating_Margin":   ebit / revenue                 if revenue > 0       else 0,
                "EBITDA_Margin":      ebitda / revenue               if revenue > 0       else 0,
                "GP_A_Quality":       gross_profit / total_assets,
                "Asset_Turnover":     revenue / total_assets,
                "Inventory_Turnover": (
                    _safe_get(cur, "Cost Of Revenue") / (_safe_get(bal, "Inventory", 1) or 1)
                    if "Inventory" in bal.index else 0
                ),
                "Revenue_Growth":     (revenue / prev_revenue) - 1,
                "NetIncome_Growth":   (net_income / prev_net_income) - 1,
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
                "Mom_1w":             (close_now / close.iloc[-6])   - 1 if len(hist) >= 6   else 0,
                "Mom_1m":             (close_now / close.iloc[-21])  - 1 if len(hist) >= 21  else 0,
                "Mom_6m":             (close_now / close.iloc[-127]) - 1 if len(hist) >= 127 else 0,
                "Mom_12m":            (close_now / close.iloc[-252]) - 1 if len(hist) >= 252 else 0,
                "MA_Convergence": (
                    (close.rolling(20).mean().iloc[-1] / close.rolling(200).mean().iloc[-1]) - 1
                    if len(hist) >= 200 else 0
                ),
                "MA50_Dist":      close_now / close.rolling(50).mean().iloc[-1]  if len(hist) >= 50  else 1,
                "MA200_Dist":     close_now / close.rolling(200).mean().iloc[-1] if len(hist) >= 200 else 1,
                "Momentum_12M_1M":(close.iloc[-21] / close.iloc[-252]) - 1 if len(hist) >= 252 else 0,
                "Momentum_6M_1M": (close.iloc[-21] / close.iloc[-126]) - 1 if len(hist) >= 126 else 0,
                "Momentum_Custom":(close.iloc[-1]  / close.iloc[-63])  - 1 if len(hist) >= 63  else 0,
                "Volatility_30d": close.pct_change().std() * np.sqrt(252),
                "Risk_Adj_Return":(
                    close.pct_change().mean() / close.pct_change().std()
                    if close.pct_change().std() != 0 else 0
                ),
                "Vol_Change": (
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
# UI
# ─────────────────────────────────────────────
st.title("🚀 Advanced AI Quant Lab")
df_sp500, all_sectors = get_sp500_info()

with st.expander("🛠 전략 설정 및 필터링", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: selected_sectors = st.multiselect("1. 분석 섹터 선택", all_sectors, default=["Information Technology"])
    with c2: start_date = st.date_input("2. 백테스트 시작일", datetime(2024, 1, 1))
    with c3: end_date   = st.date_input("3. 백테스트 종료일", datetime.now())
    with c4: reb_months = st.select_slider("4. 리밸런싱 주기 (개월)", options=[1, 3, 6, 12], value=3)
    with c5: top_n      = st.number_input("5. 선정 종목 수", min_value=3, max_value=20, value=5)
    run_analysis = st.button("백테스트 실행 🚀", use_container_width=True)


# ─────────────────────────────────────────────
# 백테스트 실행
# ─────────────────────────────────────────────
if run_analysis:
    tickers = (
        df_sp500[df_sp500["GICS Sector"].isin(selected_sectors)]["Symbol"]
        .str.replace(".", "-", regex=False)
        .tolist()
    )

    # 주가 데이터
    st.info(f"📡 주가 데이터 다운로드 중... ({len(tickers)}개 티커)")
    full_hist_data = yf.download(
        tickers,
        start=pd.to_datetime(start_date) - timedelta(days=400),
        end=pd.to_datetime(end_date) + timedelta(days=40),
        group_by="ticker",
        progress=False,
        threads=True,
    )
    st.success(f"✅ 주가 데이터 완료 ({full_hist_data.shape[1]}개 컬럼)")

    # 재무 데이터
    st.info("📊 재무 데이터 수집 중... (시간이 걸릴 수 있어요)")
    source_cache = get_all_financial_source(tickers)
    fin_count = len(source_cache)
    st.success(f"✅ 재무 데이터 완료: {fin_count}/{len(tickers)}개")

    # 재무 데이터가 너무 적으면 경고
    if fin_count < len(tickers) * 0.5:
        st.warning(
            f"⚠️ 재무 데이터 수집률이 낮아요 ({fin_count}/{len(tickers)}). "
            "Yahoo Finance 접속이 불안정할 수 있습니다. 잠시 후 다시 시도해보세요."
        )

    date_range           = pd.date_range(start=start_date, end=end_date, freq=f"{reb_months}MS")
    all_strategy_returns = pd.Series(dtype=float)
    rebalance_details    = []
    importance_history   = []
    final_model_columns  = None
    latest_model         = None

    # ── 초기 모델 학습 ─────────────────────────────────────────
    with st.status("🏗️ 초기 모델 학습 중...", expanded=True) as status:
        train_df_init = fetch_ml_data_optimized_pit(
            tickers, date_range[0], full_hist_data, source_cache, is_training=True
        )
        st.write(f"학습 데이터: {len(train_df_init)}개 종목")

        # 재무 지표가 0인 비율 체크 (디버깅용)
        if not train_df_init.empty:
            fin_cols   = ["P/E", "P/S", "ROE", "ROA", "Gross_Margin"]
            zero_ratio = (train_df_init[fin_cols] == 0).all(axis=1).mean()
            if zero_ratio > 0.7:
                st.warning(f"⚠️ {zero_ratio*100:.0f}%의 종목에서 재무 지표가 0입니다. 재무 데이터 수집을 확인하세요.")
            else:
                st.write(f"재무 지표 정상 비율: {(1-zero_ratio)*100:.0f}%")

        if not train_df_init.empty and "Target_Return" in train_df_init.columns:
            X_init = train_df_init.drop(["Ticker", "Target_Return"], axis=1)
            y_init = train_df_init["Target_Return"]
            final_model_columns = X_init.columns.tolist()
            latest_model = RandomForestRegressor(
                n_estimators=100, random_state=42,
                max_depth=10, min_samples_leaf=5, n_jobs=-1
            )
            latest_model.fit(X_init, y_init)
            status.update(label=f"✅ 초기 학습 완료 ({date_range[0].strftime('%Y-%m-%d')})", state="complete")
        else:
            status.update(label="❌ 학습 데이터 없음", state="error")

    if latest_model is None or final_model_columns is None:
        st.error("초기 학습 데이터를 가져오지 못했습니다. 섹터나 기간을 조정해 주세요.")
        st.stop()

    # ── 워크포워드 루프 ─────────────────────────────────────────
    loop_bar = st.progress(0, text="워크포워드 진행 중...")
    total_steps = len(date_range) - 1

    for i in range(total_steps):
        curr_reb = date_range[i]
        next_reb = date_range[i + 1]
        loop_bar.progress(int(i / total_steps * 100), text=f"리밸런싱 {i+1}/{total_steps} ({curr_reb.strftime('%Y-%m')})")

        inference_df = fetch_ml_data_optimized_pit(
            tickers, curr_reb, full_hist_data, source_cache, is_training=False
        )
        if inference_df.empty:
            continue

        X_test = (
            inference_df.drop(["Ticker"], axis=1)
            .reindex(columns=final_model_columns)
            .fillna(0)
        )
        inference_df = inference_df.copy()
        inference_df["Prediction"] = latest_model.predict(X_test)

        selected_rows = inference_df.nlargest(top_n, "Prediction").copy()
        sel_tickers   = selected_rows["Ticker"].tolist()

        importances = pd.Series(latest_model.feature_importances_, index=final_model_columns)
        importance_history.append({"Date": curr_reb.strftime("%Y-%m-%d"), **importances.to_dict()})
        rebalance_details.append({
            "date": curr_reb.strftime("%Y-%m-%d"),
            "selected_data": selected_rows,
            "importance": importances,
        })

        valid_sel = [t for t in sel_tickers if t in full_hist_data.columns.get_level_values(0)]
        if valid_sel:
            subset = full_hist_data[valid_sel].loc[curr_reb:next_reb]
            if not subset.empty:
                try:
                    test_prices = subset.xs("Close", axis=1, level=1)
                    period_rets = test_prices.pct_change().mean(axis=1).fillna(0)
                    all_strategy_returns = pd.concat([all_strategy_returns, period_rets])
                except Exception:
                    pass

        train_df_next = fetch_ml_data_optimized_pit(
            tickers, next_reb, full_hist_data, source_cache, is_training=True
        )
        if not train_df_next.empty and "Target_Return" in train_df_next.columns:
            X_update = train_df_next.drop(["Ticker", "Target_Return"], axis=1)
            y_update = train_df_next["Target_Return"]
            latest_model.fit(X_update, y_update)

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

    st.header(f"📊 {', '.join(selected_sectors)} 전략 성과 보고서")

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
        st.subheader("💡 상세 성과 지표")

        def get_metrics(rets):
            cum = (1 + rets).prod() - 1
            yrs = max((rets.index[-1] - rets.index[0]).days / 365.25, 0.1)
            return {
                "누적 수익률":    f"{cum * 100:.2f}%",
                "연수익률(CAGR)": f"{((1 + cum) ** (1 / yrs) - 1) * 100:.2f}%",
                "샤프 지수":      round(qs.stats.sharpe(rets), 2),
                "MDD":            f"{qs.stats.max_drawdown(rets) * 100:.2f}%",
            }

        st.table(pd.DataFrame({
            "AI Strategy": get_metrics(all_strategy_returns),
            "SPY":  get_metrics(spy_rets),
            "QQQ":  get_metrics(qqq_rets),
            "TQQQ": get_metrics(tqqq_rets),
        }))

    st.divider()
    st.subheader("🗓️ 리밸런싱 히스토리")

    for detail in reversed(rebalance_details):
        curr_date_obj = pd.to_datetime(detail["date"])
        check_cols    = [c for c in ["P/E", "ROE"] if c in detail["selected_data"].columns]
        has_fin       = (
            not (detail["selected_data"][check_cols].abs() < 0.0001).all().all()
            if check_cols else False
        )
        status_suffix = " ✅ (재무+가격)" if has_fin else " ⚠️ (모멘텀 중심)"

        with st.expander(f"📅 {detail['date']}{status_suffix}", expanded=False):
            d1, d2 = st.columns([3, 2])
            with d1:
                st.markdown("**선정 종목**")
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

    st.subheader("🧠 지표 중요도 트렌드")
    imp_all_df = pd.DataFrame(importance_history).set_index("Date").fillna(0)
    top5 = imp_all_df.mean().nlargest(5).index.tolist()
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
        latest_data = latest_data.copy()
        latest_data["AI_Score"] = latest_model.predict(X_latest)
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