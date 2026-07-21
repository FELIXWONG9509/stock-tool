import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from datetime import datetime, timedelta

# ---------- 页面设置 ----------
st.set_page_config(page_title="历史相似买点概率", layout="wide")
st.title("📈 相似历史匹配 · 获利概率评估")
st.caption("比较当前技术指标与历史每一天的相似度，统计历史上相似情况下的上涨概率。")

# ---------- 用户输入 ----------
code = st.text_input("请输入股票代码（如 000001 或 600519）", "000001")
days_hold = st.selectbox("选择持仓周期", [5, 10, 20], index=1)

# ---------- 数据获取函数 ----------
@st.cache_data
def load_data(stock_code):
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty:
            return None
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

# ---------- 指标计算 ----------
def compute_features(df):
    close = df["close"]
    volume = df["volume"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # 布林带位置 (20,2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2*bb_std
    bb_lower = bb_mid - 2*bb_std
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # 量比
    vol_ratio = volume / volume.rolling(20).mean()

    # 收盘价距年内高点比例
    year_high = close.rolling(250).max()
    close_to_high = (close - year_high) / (year_high + 1e-10)

    # MACD柱 标准化
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line
    macd_norm = macd_hist / (close + 1e-10)

    # 趋势强度 (5日与20日线差)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    trend_strength = (ma5 - ma20) / (close + 1e-10)

    features = pd.DataFrame({
        "rsi": rsi,
        "bb_position": bb_position,
        "vol_ratio": vol_ratio,
        "close_to_high": close_to_high,
        "macd_norm": macd_norm,
        "trend_strength": trend_strength,
    })
    return features

# ---------- 主分析逻辑 ----------
if st.button("开始分析"):
    if not code:
        st.warning("请输入股票代码")
    else:
        with st.spinner("正在获取数据并计算指标..."):
            data = load_data(code)
        if data is None:
            st.error("未获取到数据，请检查代码是否正确（如 000001 或 600519）")
        else:
            features = compute_features(data)
            combined = pd.concat([data[["date","close"]], features], axis=1).dropna()
            if len(combined) < 252:
                st.error("可用历史数据不足，至少需要1年以上数据。")
            else:
                current = features.iloc[-1:].values
                hist = features.iloc[:-20].dropna().values
                hist_dates = combined["date"].iloc[:-20].reset_index(drop=True)

                if len(hist) < 50:
                    st.warning("历史相似样本数不足，结果仅供参考。")

                sim = cosine_similarity(current, hist)[0]
                top_k = min(50, len(sim))
                top_idx = np.argsort(sim)[-top_k:][::-1]
                sim_scores = sim[top_idx]

                close_series = combined["close"].reset_index(drop=True)
                matched_indices = top_idx
                future_returns = []
                for idx in matched_indices:
                    if idx + days_hold < len(close_series):
                        ret = (close_series.iloc[idx + days_hold] / close_series.iloc[idx]) - 1
                        future_returns.append(ret)
                if len(future_returns) < 10:
                    st.error("有效相似样本太少，无法计算。")
                else:
                    ret_arr = np.array(future_returns)
                    win_rate = (ret_arr > 0).mean()
                    avg_ret = ret_arr.mean()
                    positive = ret_arr[ret_arr > 0]
                    negative = ret_arr[ret_arr < 0]
                    if len(positive) > 0 and len(negative) > 0:
                        pl_ratio = positive.mean() / abs(negative.mean())
                    else:
                        pl_ratio = np.inf if len(negative)==0 else 0

                    col1, col2, col3 = st.columns(3)
                    col1.metric("上涨概率", f"{win_rate:.1%}")
                    col2.metric("平均收益", f"{avg_ret:.2%}")
                    col3.metric("盈亏比", f"{pl_ratio:.2f}")

                    if win_rate > 0.55 and avg_ret > 0:
                        st.success("✅ 概率买点信号：历史相似情境下上涨概率较高且平均收益为正")
                    else:
                        st.info("ℹ️ 当前相似历史样本未达到高概率买点标准")

                    fig = px.histogram(ret_arr, nbins=20, title=f"相似历史持有{days_hold}天收益分布",
                                       labels={"value": "收益率"}, opacity=0.7)
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("查看匹配的历史日期及相似度"):
                        match_dates = combined["date"].iloc[matched_indices].reset_index(drop=True)
                        sim_df = pd.DataFrame({
                            "历史日期": match_dates.values[:len(sim_scores)],
                            "相似度": sim_scores
                        })
                        st.dataframe(sim_df.head(20))

                    st.warning("⚠️ 风险提示：历史表现不代表未来，本工具仅作统计参考，不构成投资建议。")