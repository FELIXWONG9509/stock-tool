import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="自定义指标历史相似概率", layout="wide")
st.title("📈 自选技术指标 · 历史相似匹配获利概率")
st.caption("选择技术指标及参数，寻找历史上最相似的时刻（含趋势形态），计算后续上涨概率。")

code = st.text_input("股票代码（如 600887）", "600887")
days_hold = st.selectbox("持仓周期（天）", [5, 10, 20, 50, 100, 150, 200, 300, 400], index=2)

# ----- 指标选择区（含说明）-----
st.sidebar.header("🔧 选择技术指标组合")
st.sidebar.markdown("勾选你想使用的指标，并可调整参数。鼠标悬停或看下方说明。")

# SKDJ
use_skdj = st.sidebar.checkbox("SKDJ (慢速随机指标)", value=True)
st.sidebar.caption("衡量超买超卖及K/D线交叉，适合判断波段拐点。")
if use_skdj:
    skdj_n = st.sidebar.slider("SKDJ 参数 N (快线周期)", 5, 30, 9)
    skdj_m = st.sidebar.slider("SKDJ 参数 M (慢线周期)", 2, 10, 3)

# RSI
use_rsi = st.sidebar.checkbox("RSI (相对强弱指标)", value=True)
st.sidebar.caption("0~100摆动，高于70超买，低于30超卖，反映近期涨跌力度。")
if use_rsi:
    rsi_period = st.sidebar.slider("RSI 周期", 5, 30, 14)

# MACD
use_macd = st.sidebar.checkbox("MACD 柱", value=True)
st.sidebar.caption("快慢均线差与信号线的关系，红绿柱反映多空动能变化。")
if use_macd:
    macd_fast = st.sidebar.slider("MACD 快线", 5, 30, 12)
    macd_slow = st.sidebar.slider("MACD 慢线", 10, 40, 26)
    macd_signal = st.sidebar.slider("MACD 信号线", 5, 15, 9)

# 布林带
use_bb = st.sidebar.checkbox("布林带位置", value=True)
st.sidebar.caption("股价在上下轨间的相对位置，触及下轨可能超卖，上轨可能超买。")
if use_bb:
    bb_period = st.sidebar.slider("布林带周期", 10, 50, 20)
    bb_std = st.sidebar.slider("布林带标准差倍数", 1, 4, 2)

# 量比
use_vol = st.sidebar.checkbox("量比", value=True)
st.sidebar.caption("当日成交量与近期均量的比值，大于1放量，小于1缩量。")
if use_vol:
    vol_period = st.sidebar.slider("均量周期", 5, 30, 20)

# 趋势强度
use_trend = st.sidebar.checkbox("短期趋势强度 (5日/20日线)", value=True)
st.sidebar.caption("短均线与长均线的距离，正值多头排列，负值空头排列。")

if not any([use_skdj, use_rsi, use_macd, use_bb, use_vol, use_trend]):
    st.error("请在左侧至少选择一个技术指标！")
    st.stop()

# ----- 数据获取 -----
@st.cache_data
def load_data(stock_code):
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty:
            return None
        df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

# ----- 指标计算引擎（含趋势特征）-----
def compute_all_features(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    features = pd.DataFrame(index=df.index)

    if use_skdj:
        low_n = low.rolling(window=skdj_n).min()
        high_n = high.rolling(window=skdj_n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-10) * 100
        k = rsv.ewm(alpha=1/skdj_m, adjust=False).mean()
        d = k.ewm(alpha=1/skdj_m, adjust=False).mean()
        skdj_k = d
        skdj_d = d.ewm(alpha=1/skdj_m, adjust=False).mean()
        features["skdj_k"] = skdj_k / 100.0
        features["skdj_d"] = skdj_d / 100.0
        features["skdj_k_5d_change"] = skdj_k.diff(5) / 100.0
        features["skdj_d_5d_change"] = skdj_d.diff(5) / 100.0
        features["skdj_kd_diff"] = (skdj_k - skdj_d) / 100.0

    if use_rsi:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        features["rsi"] = rsi / 100.0
        features["rsi_5d_change"] = rsi.diff(5) / 100.0

    if use_macd:
        ema_fast = close.ewm(span=macd_fast).mean()
        ema_slow = close.ewm(span=macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal = macd_line.ewm(span=macd_signal).mean()
        macd_hist = macd_line - signal
        features["macd_hist_norm"] = macd_hist / (close + 1e-10)
        features["macd_hist_5d_change"] = macd_hist.diff(5) / (close + 1e-10)

    if use_bb:
        bb_mid = close.rolling(bb_period).mean()
        bb_std_val = close.rolling(bb_period).std()
        bb_upper = bb_mid + bb_std*bb_std_val
        bb_lower = bb_mid - bb_std*bb_std_val
        features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    if use_vol:
        features["vol_ratio"] = volume / volume.rolling(vol_period).mean()

    if use_trend:
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        features["trend_strength"] = (ma5 - ma20) / (close + 1e-10)

    return features

# ----- 主逻辑 -----
if st.button("🔍 开始分析"):
    if not code:
        st.warning("请输入股票代码")
    else:
        with st.spinner("下载数据并计算指标..."):
            data = load_data(code)
        if data is None:
            st.error("无法获取数据，请检查代码是否正确")
        else:
            features = compute_all_features(data)
            combined = pd.concat([data[["date","close"]], features], axis=1).dropna()
            if len(combined) < 252:
                st.error("有效历史数据不足，至少需1年以上")
            else:
                feature_cols = [col for col in combined.columns if col not in ["date", "close"]]
                current_feat = combined[feature_cols].iloc[-1:].values
                hist_feat = combined[feature_cols].iloc[:-20].values

                if len(hist_feat) < 50:
                    st.warning("历史相似样本数较少，结果可能有偏差")

                sim = cosine_similarity(current_feat, hist_feat)[0]
                top_k = min(50, len(sim))
                top_idx = np.argsort(sim)[-top_k:][::-1]
                sim_scores = sim[top_idx]

                close_series = combined["close"].reset_index(drop=True)
                rets = []
                for idx in top_idx:
                    if idx + days_hold < len(close_series):
                        ret = (close_series.iloc[idx + days_hold] / close_series.iloc[idx]) - 1
                        rets.append(ret)

                if len(rets) < 10:
                    st.error("有效相似样本太少，无法统计")
                else:
                    ret_arr = np.array(rets)
                    win_rate = (ret_arr > 0).mean()
                    avg_ret = ret_arr.mean()
                    pos = ret_arr[ret_arr > 0]
                    neg = ret_arr[ret_arr < 0]
                    if len(pos) > 0 and len(neg) > 0:
                        pl_ratio = pos.mean() / abs(neg.mean())
                    else:
                        pl_ratio = np.inf if len(neg) == 0 else 0

                    col1, col2, col3 = st.columns(3)
                    col1.metric("上涨概率", f"{win_rate:.1%}")
                    col2.metric("平均收益", f"{avg_ret:.2%}")
                    col3.metric("盈亏比", f"{pl_ratio:.2f}")

                    if win_rate > 0.55 and avg_ret > 0:
                        st.success("✅ 概率买点信号")
                    else:
                        st.info("ℹ️ 未达到高概率买点标准")

                    fig = px.histogram(ret_arr, nbins=20,
                                       title=f"相似历史持有{days_hold}天收益分布",
                                       labels={"value": "收益率"}, opacity=0.7)
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("查看相似历史日期及相似度"):
                        match_dates = combined["date"].iloc[top_idx].reset_index(drop=True)
                        sim_df = pd.DataFrame({
                            "历史日期": match_dates.values[:len(sim_scores)],
                            "相似度": sim_scores
                        })
                        st.dataframe(sim_df.head(20))

                    st.warning("⚠️ 风险提示：历史表现不代表未来，本工具仅供参考，不构成投资建议。")
