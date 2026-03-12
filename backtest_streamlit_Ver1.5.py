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
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime, timedelta
import concurrent.futures
import time

# 페이지 설정
st.set_page_config(page_title="Advanced AI Quant Lab", layout="wide")

# 1. S&P 500 정보 가져오기
@st.cache_data
def get_sp500_info():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_html(StringIO(response.text))[0]
    sectors = sorted(df['GICS Sector'].unique().tolist())
    return df, sectors

# 재무 데이터 수집 (서버 환경 최적화 버전)
@st.cache_data(show_spinner=False)
def get_all_financial_source(tickers):
    data_cache = {}
    
    def fetch(t):
        # 1. 서버 차단을 피하기 위해 실행 시점을 랜덤하게 분산 (중요!)
        import random
        # 2, 3, 4, 5 중에서 하나를 무작위 선택
        time.sleep(random.choice([2.0, 5.0]))
        
        try:
            # 2. session을 넣지 말고, 직접 Ticker 객체 생성
            tk = yf.Ticker(t)
            
            # 3. 데이터가 즉시 안 올 수 있으므로 'info'를 먼저 호출해서 깨워줌
            _ = tk.info 
            
            # 4. 재무제표 시도 (분기 -> 연간 순서)
            q_fin = tk.quarterly_financials
            if q_fin is None or q_fin.empty:
                q_fin = tk.financials
            
            if q_fin is None or q_fin.empty:
                return t, None

            # 5. 모든 데이터가 정상일 때만 수집
            return t, {
                'q_fin': q_fin.T, 
                'q_bal': tk.quarterly_balance_sheet.T if tk.quarterly_balance_sheet is not None else pd.DataFrame(), 
                'q_cf': tk.quarterly_cashflow.T if tk.quarterly_cashflow is not None else pd.DataFrame(),
                'a_fin': tk.financials.T, 
                'a_bal': tk.balance_sheet.T, 
                'a_cf': tk.cashflow.T,
                'info': tk.info
            }
        except Exception as e:
            return t, None

    # 6. 클라우드 서버에서는 병렬 처리를 '포기'해야 합니다. 
    # 여러 명이 동시에 요청하는 것처럼 보이면 바로 차단당합니다.
    # 속도는 느리지만 '확실하게' 가져오기 위해 max_workers=1 또는 2로 낮춥니다.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(fetch, tickers))
    
    for t, val in results:
        if val: data_cache[t] = val
    return data_cache


# 데이터 추출 함수
def fetch_ml_data_optimized_pit(tickers, ref_date, full_hist_data, source_cache, is_training=True):
    features_list = []
    ref_dt = pd.to_datetime(ref_date)
    
    for ticker in tickers:
        try:
            if ticker not in full_hist_data.columns.get_level_values(0): continue
            
            ticker_all_prices = full_hist_data[ticker].dropna()
            # [수정] 기준일(ref_dt) 이전의 데이터만 사용하여 지표 계산
            hist = ticker_all_prices[ticker_all_prices.index < ref_dt].tail(252)
            if len(hist) < 252: continue
            
            close_now = hist['Close'].iloc[-1]
            src = source_cache.get(ticker, {})
            info = src.get('info', {})
            
            def get_fallback_data(q_key, a_key, ref_dt):
                df = src.get(q_key, pd.DataFrame())
                if df.empty:
                    df = src.get(a_key, pd.DataFrame())
                
                if df.empty: return pd.DataFrame()

                # [로직 변경] 기준일(ref_dt)과 각 데이터 날짜의 차이를 계산
                # 기준일보다 미래여도, 그 차이가 31일 이내라면 '당일 시점 데이터'로 인정 (CASE 1 대응)
                # 기준일보다 과거라면 당연히 인정 (CASE 2 대응)
                
                # 1. 날짜 차이 계산 (기준일 - 데이터일)
                # diff가 0보다 크면 과거, -31보다 크면 한 달 이내의 미래
                time_diffs = (ref_dt - df.index).days
                
                # 2. 조건에 맞는 데이터 필터링: 기준일 이전이거나, 기준일 후 31일 이내인 것
                valid_mask = (time_diffs >= -31) 
                valid_df = df[valid_mask]

                if not valid_df.empty:
                    # 기준일과 가장 가까운(절대값 차이가 작은) 데이터를 반환
                    closest_idx = np.abs(ref_dt - valid_df.index).argmin()
                    return valid_df.iloc[[closest_idx]]
                
                return pd.DataFrame()

            past_fin = get_fallback_data('q_fin', 'a_fin', ref_dt)
            past_bal = get_fallback_data('q_bal', 'a_bal', ref_dt)
            past_cf = get_fallback_data('q_cf', 'a_cf', ref_dt)
            
            cur = past_fin.iloc[0] if not past_fin.empty else pd.Series(dtype=float)
            prev = past_fin.iloc[1] if len(past_fin) > 1 else cur
            bal = past_bal.iloc[0] if not past_bal.empty else pd.Series(dtype=float)
            cf = past_cf.iloc[0] if not past_cf.empty else pd.Series(dtype=float)
            
            shares = info.get('sharesOutstanding', 1)
            mkt_cap = close_now * shares
            
            net_income = cur.get('Net Income', 0); revenue = cur.get('Total Revenue', 0)
            gross_profit = cur.get('Gross Profit', 0); total_assets = bal.get('Total Assets', 1)
            fcf = cf.get('Free Cash Flow', 0)
            ebit = cur.get('EBIT', 0)
            da = cf.get('Depreciation And Amortization', 0)
            ebitda = ebit + da
            annual_ebitda = ebitda * 4
            total_debt = bal.get('Total Debt', 0)
            cash_and_equiv = bal.get('Cash And Cash Equivalents', 0)
            ev = mkt_cap + total_debt - cash_and_equiv

            # 데이터 딕셔너리 (지표 변경 없음)
            data = {
                'Ticker': ticker,
                'P/E': mkt_cap / (net_income * 4) if net_income > 0 else 0,
                'P/S': mkt_cap / (revenue * 4) if revenue > 0 else 0,
                'P/B': mkt_cap / bal.get('Stockholders Equity', 1) if 'Stockholders Equity' in bal else 0,
                'P/FCF': mkt_cap / (fcf * 4) if fcf > 0 else 0,
                'EV/EBITDA': ev / annual_ebitda if annual_ebitda > 0 else 0,
                'FCF_Yield': (fcf * 4) / mkt_cap if mkt_cap != 0 else 0,
                #'ROI': (cur.get('EBIT', 0) * 4) / (bal.get('Total Assets', 1) - bal.get('Total Current Liabilities', 0)) if 'Total Current Liabilities' in bal else 0,
                'ROE': net_income / bal.get('Stockholders Equity', 1) if 'Stockholders Equity' in bal else 0,
                'ROA': net_income / total_assets if total_assets != 0 else 0,
                'Gross_Margin': gross_profit / revenue if revenue > 0 else 0,
                'Operating_Margin': ebit / revenue if revenue > 0 else 0,
                'EBITDA_Margin': ebitda / revenue if revenue > 0 else 0,
                'GP_A_Quality': gross_profit / total_assets if total_assets != 0 else 0,
                'Asset_Turnover': revenue / total_assets if total_assets != 0 else 0,
                'Inventory_Turnover': (cur.get('Cost Of Revenue', 0)) / bal.get('Inventory', 1) if 'Inventory' in bal else 0,
                'Revenue_Growth': (revenue / prev.get('Total Revenue', 1)) - 1 if not prev.empty else 0,
                'NetIncome_Growth': (net_income / prev.get('Net Income', 1)) - 1 if not prev.empty else 0,
                'Debt_Equity': total_debt / bal.get('Stockholders Equity', 1) if 'Stockholders Equity' in bal else 0,
                'Current_Ratio': bal.get('Total Current Assets', 0) / bal.get('Total Current Liabilities', 1) if 'Total Current Liabilities' in bal else 0,
                'Interest_Coverage': ebit / cur.get('Interest Expense', 1) if 'Interest Expense' in cur else 0,
                'Mom_1w': (close_now / hist['Close'].iloc[-6]) - 1 if len(hist) >= 6 else 0,
                'Mom_1m': (close_now / hist['Close'].iloc[-21]) - 1 if len(hist) >= 21 else 0,
                'Mom_6m': (close_now / hist['Close'].iloc[-127]) - 1 if len(hist) >= 127 else 0,
                'Mom_12m': (close_now / hist['Close'].iloc[-252]) - 1 if len(hist) >= 252 else 0,
                'MA_Convergence': (hist['Close'].rolling(20).mean().iloc[-1] / hist['Close'].rolling(200).mean().iloc[-1]) - 1 if len(hist) >= 200 else 0,
                'MA50_Dist': close_now / hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else 1,
                'MA200_Dist': close_now / hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else 1,
                'Momentum_12M_1M': (hist['Close'].iloc[-21] / hist['Close'].iloc[-252]) - 1 if len(hist) >= 252 else 0,
                'Momentum_6M_1M': (hist['Close'].iloc[-21] / hist['Close'].iloc[-126]) - 1 if len(hist) >= 126 else 0,
                'Momentum_Custom': (hist['Close'].iloc[-1] / hist['Close'].iloc[-63]) - 1 if len(hist) >= 63 else 0,
                'Volatility_30d': hist['Close'].pct_change().std() * np.sqrt(252),
                'Risk_Adj_Return': (hist['Close'].pct_change().mean() / hist['Close'].pct_change().std()) if hist['Close'].pct_change().std() != 0 else 0,
                'Vol_Change': hist['Volume'].iloc[-1] / hist['Volume'].rolling(21).mean().iloc[-1] if len(hist) >= 21 else 1,
            }

            # [수정] 학습용 데이터일 때만 Target_Return(미래 수익률) 계산
            if is_training:
                # ref_dt 이후의 22거래일 수익률을 타겟으로 잡음
                future_prices = ticker_all_prices[ticker_all_prices.index >= ref_dt].head(22)
                if len(future_prices) >= 20:
                    data['Target_Return'] = future_prices['Close'].pct_change().sum()
                else:
                    continue # 미래 데이터 부족 시 학습에서 제외

            features_list.append(data)
        except: continue
    return pd.DataFrame(features_list).replace([np.inf, -np.inf], np.nan).fillna(0)

# 시각화 함수 (절대 변경 안함)
def display_importance_heatmap(imp_all_df):
    st.subheader("🌡️ 지표별 영향력 타임라인 (Heatmap)")
    if imp_all_df.empty:
        st.write("데이터가 부족합니다.")
        return
    top_15_features = imp_all_df.mean().nlargest(15).index.tolist()
    heatmap_data = imp_all_df[top_15_features].T
    try:
        heatmap_data.columns = [pd.to_datetime(d).strftime('%Y-%m') for d in heatmap_data.columns]
    except:
        heatmap_data.columns = [str(d)[:7] for d in heatmap_data.columns]
    heatmap_data_norm = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9), axis=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(heatmap_data_norm, annot=heatmap_data.values, fmt=".2f", cmap="YlGnBu", linewidths=.5, ax=ax)
    plt.title("Feature Importance Heatmap (Top 15 Metrics)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 실행 로직
st.title("🚀 Advanced AI Quant Lab")
df_sp500, all_sectors = get_sp500_info()

with st.expander("🛠 전략 설정 및 필터링", expanded=True):
    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns(5)
    with top_col1: selected_sectors = st.multiselect("1. 분석 섹터 선택", all_sectors, default=["Information Technology"])
    with top_col2: start_date = st.date_input("2. 백테스트 시작일", datetime(2025, 1, 1))
    with top_col3: end_date = st.date_input("3. 백테스트 종료일", datetime.now())
    with top_col4: reb_months = st.select_slider("4. 리밸런싱 주기 (개월)", options=[1, 3, 6, 12], value=3)
    with top_col5: top_n = st.number_input("5. 선정 종목 수", min_value=3, max_value=20, value=5)
    run_analysis = st.button("백테스트 실행 🚀", use_container_width=True)

if run_analysis:
    tickers = df_sp500[df_sp500['GICS Sector'].isin(selected_sectors)]['Symbol'].str.replace('.', '-', regex=False).tolist()
    with st.spinner("📦 데이터 수집 및 미래 참조 차단 검증 중..."):
        full_hist_data = yf.download(tickers, start=pd.to_datetime(start_date) - timedelta(days=400), end=pd.to_datetime(end_date) + timedelta(days=40), group_by='ticker', progress=False)
        source_cache = get_all_financial_source(tickers)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=f'{reb_months}MS')
        all_strategy_returns = pd.Series(dtype=float)
        rebalance_details = []; importance_history = []

        final_model_columns = None 
        latest_model = None

        with st.status("🏗️ 초기 모델 사전 학습 중...", expanded=False):
            initial_train_date = date_range[0]
            train_df_init = fetch_ml_data_optimized_pit(tickers, initial_train_date, full_hist_data, source_cache, is_training=True)
            
            if not train_df_init.empty:
                X_train_init = train_df_init.drop(['Ticker', 'Target_Return'], axis=1)
                y_train_init = train_df_init['Target_Return']
                
                final_model_columns = X_train_init.columns.tolist() 
                latest_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=5, n_jobs=-1)
                latest_model.fit(X_train_init, y_train_init)
                st.write(f"✅ 초기 학습 완료 (기준일: {initial_train_date.strftime('%Y-%m-%d')})")
        
        # [수정] 워크포워드(Walk-forward) 방식 리밸런싱 루프
        for i in range(len(date_range) - 1):
            curr_reb = date_range[i]
            next_reb = date_range[i+1]
            
            # [핵심] 현재 시점(curr_reb)의 데이터로 종목 선정 (이미 latest_model이 있으므로 바로 추론 가능)
            inference_df = fetch_ml_data_optimized_pit(tickers, curr_reb, full_hist_data, source_cache, is_training=False)
            if inference_df.empty: continue

            # 컬럼 정렬 및 예측
            X_test = inference_df.drop(['Ticker'], axis=1).reindex(columns=final_model_columns).fillna(0)
            inference_df['Prediction'] = latest_model.predict(X_test)
            
            # 종목 선정 및 수익률 기록 (기존 로직 동일)
            selected_rows = inference_df.nlargest(top_n, 'Prediction').copy()
            sel_tickers = selected_rows['Ticker'].tolist()
            
            importances = pd.Series(latest_model.feature_importances_, index=final_model_columns)
            importance_history.append({'Date': curr_reb.strftime('%Y-%m-%d'), **importances.to_dict()})
            rebalance_details.append({'date': curr_reb.strftime('%Y-%m-%d'), 'selected_data': selected_rows, 'importance': importances})

            subset = full_hist_data[sel_tickers].loc[curr_reb:next_reb]
            if not subset.empty:
                test_prices = subset.xs('Close', axis=1, level=1)
                period_returns = test_prices.pct_change().mean(axis=1).fillna(0)
                all_strategy_returns = pd.concat([all_strategy_returns, period_returns])

            # [수정] 다음 리밸런싱을 위해 모델 업데이트 (Online Learning)
            # 현재 구간의 데이터를 학습 세트에 포함시켜 다음 구간 종목 선정에 반영
            train_df_next = fetch_ml_data_optimized_pit(tickers, curr_reb, full_hist_data, source_cache, is_training=True)
            if not train_df_next.empty:
                X_update = train_df_next.drop(['Ticker', 'Target_Return'], axis=1)
                y_update = train_df_next['Target_Return']
                latest_model.fit(X_update, y_update) # 모델 갱신

            # --- 결과 처리 및 시각화 (기존 로직 유지) ---
        if not all_strategy_returns.empty:
            all_strategy_returns = all_strategy_returns[~all_strategy_returns.index.duplicated()].sort_index()
            start_ts, end_ts = all_strategy_returns.index[0], all_strategy_returns.index[-1]
            bench_raw = yf.download(['SPY', 'QQQ', 'TQQQ'], start=start_ts - timedelta(days=5), end=end_ts + timedelta(days=5), progress=False)['Close']
            bench_raw = bench_raw.ffill().reindex(all_strategy_returns.index).ffill()
            
            spy_rets = bench_raw['SPY'].pct_change().fillna(0); qqq_rets = bench_raw['QQQ'].pct_change().fillna(0); tqqq_rets = bench_raw['TQQQ'].pct_change().fillna(0)

            st.header(f"📊 {', '.join(selected_sectors)} 통합 전략 성과 보고서")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📈 누적 수익률 비교 (Interactive)")
                cum_returns = pd.DataFrame({
                    "Strategy (AI)": (1 + all_strategy_returns).cumprod(),
                    "SPY": (1 + spy_rets).cumprod(),
                    "QQQ": (1 + qqq_rets).cumprod(),
                    "TQQQ": (1 + tqqq_rets).cumprod()
                })
                fig_cum = px.line(cum_returns, x=cum_returns.index, y=cum_returns.columns, color_discrete_map={"Strategy (AI)": "firebrick", "SPY": "royalblue", "QQQ": "seagreen", "TQQQ": "orange"})
                fig_cum.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500)
                st.plotly_chart(fig_cum, use_container_width=True)
            
            with col2:
                st.subheader("💡 상세 성과 지표")
                def get_metrics(rets):
                    cum = (1 + rets).prod() - 1
                    yrs = max((rets.index[-1] - rets.index[0]).days / 365.25, 0.1)
                    return {"누적 수익률": f"{cum * 100:.2f}%", "연수익률(CAGR)": f"{( (1+cum)**(1/yrs)-1 )*100:.2f}%", "샤프 지수": round(qs.stats.sharpe(rets), 2), "MDD": f"{qs.stats.max_drawdown(rets) * 100:.2f}%"}
                st.table(pd.DataFrame({"AI Strategy": get_metrics(all_strategy_returns), "SPY": get_metrics(spy_rets), "QQQ": get_metrics(qqq_rets), "TQQQ": get_metrics(tqqq_rets)}))

            st.divider()
            st.subheader("🗓️ 리밸런싱 히스토리 분석")
            for detail in reversed(rebalance_details):
                curr_date_obj = pd.to_datetime(detail['date'])
                train_start = (curr_date_obj - timedelta(days=365)).strftime('%Y-%m-%d')
                train_end = (curr_date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
                
                # 재무 데이터 체크 (P/E, ROE 등이 모두 0이면 재무 누락으로 판단)
                check_cols = [c for c in ['P/E', 'ROE'] if c in detail['selected_data'].columns]
                has_fin = not (detail['selected_data'][check_cols].abs() < 0.0001).all().all() if check_cols else False
                status_suffix = " ✅ (재무+가격 통합)" if has_fin else " ⚠️ (가격/모멘텀 중심)"
                
                with st.expander(f"📅 {detail['date']} 분석 결과{status_suffix}", expanded=True):
                    st.write(f"학습 기간: {train_start} ~ {train_end}")
                    d_col1, d_col2 = st.columns([3, 2])
                    with d_col1: 
                        st.markdown("**선정 종목 상세 데이터**")
                        st.dataframe( detail['selected_data'].drop(columns=['Target_Return', 'Prediction'], errors='ignore'), use_container_width=True,  hide_index=True)
                    with d_col2:
                        st.markdown("**AI 모델 지표 중요도 (Top 10)**")
                        imp_df = detail['importance'].head(10).reset_index()
                        imp_df.columns = ['지표', '중요도']
                        fig_imp = px.bar( imp_df, x='중요도', y='지표',  orientation='h', color='중요도', color_continuous_scale='Greens',height=300)
                        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
                        st.plotly_chart(fig_imp, use_container_width=True, key=f"imp_chart_{detail['date']}")
                        
            st.subheader("🧠 지표 중요도 트렌드")
            imp_all_df = pd.DataFrame(importance_history).set_index('Date').fillna(0)
            top_features = imp_all_df.mean().nlargest(5).index.tolist()
            st.line_chart(imp_all_df[top_features])

            import plotly.express as px
            imp_norm = imp_all_df.div(imp_all_df.sum(axis=1), axis=0).fillna(0) * 100

            top_features = imp_norm.mean().nlargest(7).index.tolist()
            display_df = imp_norm[top_features].reset_index()

            fig = px.area(display_df, x='Date', y=top_features, title="🧠 주요 지표별 영향력 비중 추이 (Top 7)", labels={'value': '중요도 비중 (%)', 'Date': '리밸런싱 시점', 'variable': '지표'}, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(
                hovermode="x unified", legend_orientation="h", legend_y=-0.2, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

            import seaborn as sns
            import matplotlib.pyplot as plt
            import pandas as pd

            def display_importance_heatmap(imp_all_df):
                st.subheader("🌡️ 지표별 영향력 타임라인 (Heatmap)")
                
                if imp_all_df.empty:
                    st.write("데이터가 부족하여 히트맵을 표시할 수 없습니다.")
                    return

                top_15_features = imp_all_df.mean().nlargest(15).index.tolist()
                heatmap_data = imp_all_df[top_15_features].T

                try:
                    heatmap_data.columns = [pd.to_datetime(d).strftime('%Y-%m') for d in heatmap_data.columns]
                except:
                    heatmap_data.columns = [str(d)[:7] for d in heatmap_data.columns]

                heatmap_data_norm = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9), axis=0)

                # 4. 시각화
                fig, ax = plt.subplots(figsize=(12, 10)) # 지표가 많으니 세로로 좀 더 길게
                sns.heatmap(heatmap_data_norm, annot=heatmap_data.values, fmt=".2f",cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Relative Importance'}, ax=ax)
                plt.title("Feature Importance Heatmap (Top 15 Metrics)")
                plt.xticks(rotation=45) # 날짜가 겹치지 않게 회전
                
                st.pyplot(fig)
            display_importance_heatmap(imp_all_df)

            st.divider()
            st.subheader(f"🎯 실시간 AI 추천 종목 (Next {reb_months}M)")
            latest_data = fetch_ml_data_optimized_pit(tickers, datetime.now(), full_hist_data, source_cache, is_training=False)
            if not latest_data.empty and latest_model is not None and final_model_columns is not None:
                X_latest = latest_data.drop(['Ticker'], axis=1, errors='ignore').reindex(columns=final_model_columns).fillna(0)
                latest_data['AI_Score'] = latest_model.predict(X_latest)
                recommend_all = latest_data.sort_values(by='AI_Score', ascending=False)
                display_cols = ['Ticker', 'AI_Score'] + [c for c in final_model_columns if c in recommend_all.columns]
                st.dataframe(recommend_all[display_cols].style.background_gradient(subset=['AI_Score'], cmap='YlGn').format(precision=3), use_container_width=True, hide_index=True)
            else:
                st.warning("실시간 추천 데이터를 생성할 수 없습니다. (데이터 부족 또는 모델 미학습)")

    # --- [추가] 데이터 다운로드 섹션 ---
            st.divider()
            st.subheader("📥 분석 결과 내보내기")
            down_col1, down_col2 = st.columns(2)

            with down_col1:
                # 1. 누적 수익률 데이터 다운로드
                if not all_strategy_returns.empty:
                    csv_returns = cum_returns.to_csv().encode('utf-8-sig') # 한글 깨짐 방지
                    st.download_button(
                        label="📈 누적 수익률 시계열 다운로드 (CSV)",
                        data=csv_returns,
                        file_name=f"ai_quant_returns_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            with down_col2:
                # 2. 최신 추천 종목 데이터 다운로드
                if not latest_data.empty:
                    # 추천 리스트를 점수순으로 정렬한 데이터프레임 준비
                    csv_recommend = recommend_all[display_cols].to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="🎯 AI 추천 종목 리스트 다운로드 (CSV)",
                        data=csv_recommend,
                        file_name=f"ai_recommendation_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            log_col1, log_col2 = st.columns(2)

            with log_col1:
                # 3. 모든 리밸런싱 히스토리 합치기 (과거에 어떤 종목을 샀었는지)
                if rebalance_details:
                    history_dfs = []
                    for detail in rebalance_details:
                        temp_df = detail['selected_data'].copy()
                        temp_df.insert(0, 'Rebalance_Date', detail['date']) # 날짜를 맨 앞에 삽입
                        history_dfs.append(temp_df)
                    
                    full_history_df = pd.concat(history_dfs, ignore_index=True)
                    csv_history = full_history_df.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        label="📜 전체 리밸런싱 히스토리 (CSV)",
                        data=csv_history,
                        file_name=f"full_rebalance_history_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            with log_col2:
                # 4. 분석 원본(Raw) 데이터 (모델이 학습/추론에 쓴 모든 지표 데이터)
                raw_data_list = []
                for i in range(len(date_range) - 1):
                    curr_reb = date_range[i]
                    # 해당 시점의 모든 종목 데이터를 다시 추출
                    raw_step = fetch_ml_data_optimized_pit(tickers, curr_reb, full_hist_data, source_cache, is_training=False)
                    raw_step.insert(0, 'Data_Date', curr_reb.strftime('%Y-%m-%d'))
                    raw_data_list.append(raw_step)
                
                if raw_data_list:
                    full_raw_df = pd.concat(raw_data_list, ignore_index=True)
                    csv_raw = full_raw_df.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        label="📊 분석 원본(Raw) 데이터 다운로드 (CSV)",
                        data=csv_raw,
                        file_name=f"quant_raw_features_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
else:
    st.info("섹터를 선택하고 백테스트를 실행하세요.")

