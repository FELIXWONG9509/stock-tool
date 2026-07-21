import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="多指标历史相似概率", layout="wide")
st.title("📈 多技术指标 · 历史相似匹配获利概率")
st.caption("短线/长线指标自由组合，寻找历史上最相似的时刻，计算后续上涨概率。")

code = st.text_input("股票代码（如 600887）", "600887")
days_hold = st.selectbox("持仓周期（天）", [5, 10, 20, 50, 100, 150, 200, 300, 400], index=2)

# ========== 预设组合管理 ==========
if 'preset' not in st.session_state:
    st.session_state.preset = '自定义'

# 定义所有指标 session key
short_keys = ['use_kdj', 'use_skdj', 'use_rsi', 'use_wr', 'use_bias', 'use_cci', 'use_roc']
long_keys = ['use_ma', 'use_macd', 'use_expma', 'use_boll', 'use_sar', 'use_dmi', 'use_obv', 'use_vol', 'use_trend']
all_keys = short_keys + long_keys

# 初始化 session_state
for k in all_keys:
    if k not in st.session_state:
        st.session_state[k] = False

# 预设选择
preset = st.sidebar.selectbox("🎯 预设组合（快速勾选）", ["自定义", "短线波段组合", "长线趋势组合", "全能组合"], key='preset_select')
if preset != st.session_state.preset:
    st.session_state.preset = preset
    if preset == "短线波段组合":
        for k in short_keys:
            st.session_state[k] = True
        for k in long_keys:
            st.session_state[k] = False
    elif preset == "长线趋势组合":
        for k in long_keys:
            st.session_state[k] = True
        for k in short_keys:
            st.session_state[k] = False
    elif preset == "全能组合":
        for k in all_keys:
            st.session_state[k] = True
    # 自定义：保持当前勾选不变
    st.rerun()

# ========== 短线指标区域 ==========
with st.sidebar.expander("⚡ 短线指标区域", expanded=True):
    use_kdj = st.checkbox("KDJ (随机指标)", key='use_kdj')
    st.caption("K/D/J三线，反映超买超卖与交叉信号。")
    use_skdj = st.checkbox("SKDJ (慢速随机指标)", key='use_skdj')
    st.caption("慢速平滑KDJ，适合波段拐点判断。")
    if use_skdj:
        skdj_n = st.slider("SKDJ N", 5, 30, 9)
        skdj_m = st.slider("SKDJ M", 2, 10, 3)
    use_rsi = st.checkbox("RSI (相对强弱)", key='use_rsi')
    st.caption("0~100摆动，>70超买，<30超卖。")
    if use_rsi:
        rsi_period = st.slider("RSI 周期", 5, 30, 14)
    use_wr = st.checkbox("WR (威廉指标)", key='use_wr')
    st.caption("与KDJ类似，-80以下超卖，-20以上超买。")
    if use_wr:
        wr_period = st.slider("WR 周期", 5, 30, 14)
    use_bias = st.checkbox("BIAS (乖离率)", key='use_bias')
    st.caption("收盘价与均线的偏离程度，捕捉回归机会。")
    if use_bias:
        bias_period = st.slider("BIAS 均线周期", 5, 60, 20)
    use_cci = st.checkbox("CCI (商品通道指数)", key='use_cci')
    st.caption("突破+100/-100为强/弱势信号。")
    if use_cci:
        cci_period = st.slider("CCI 周期", 5, 30, 20)
    use_roc = st.checkbox("ROC (变动速率)", key='use_roc')
    st.caption("价格N日涨跌幅，衡量趋势速度。")
    if use_roc:
        roc_period = st.slider("ROC 周期", 5, 30, 12)

# ========== 长线指标区域 ==========
with st.sidebar.expander("📊 长线指标区域", expanded=True):
    use_ma = st.checkbox("MA (均线排列)", key='use_ma')
    st.caption("多周期均线位置与多头排列强度。")
    use_macd = st.checkbox("MACD", key='use_macd')
    st.caption("快慢线差与柱体，反映趋势动能。")
    if use_macd:
        macd_fast = st.slider("MACD 快线", 5, 30, 12)
        macd_slow = st.slider("MACD 慢线", 10, 40, 26)
        macd_signal = st.slider("MACD 信号线", 5, 15, 9)
    use_expma = st.checkbox("EXPMA (指数均线)", key='use_expma')
    st.caption("近期价格偏重，反应更快。")
    if use_expma:
        expma_short = st.slider("EXPMA 短期", 5, 30, 12)
        expma_long = st.slider("EXPMA 长期", 20, 60, 50)
    use_boll = st.checkbox("BOLL (布林带)", key='use_boll')
    st.caption("上下轨与中轨的相对位置。")
    if use_boll:
        bb_period = st.slider("BOLL 周期", 10, 50, 20)
        bb_std = st.slider("标准差倍数", 1, 4, 2)
    use_sar = st.checkbox("SAR (抛物线转向)", key='use_sar')
    st.caption("停损点，价格与SAR的距离反映趋势强度。")
    use_dmi = st.checkbox("DMI (趋向指标)", key='use_dmi')
    st.caption("PDI/MDI/ADX，判断趋势有无及方向。")
    if use_dmi:
        dmi_period = st.slider("DMI 周期", 5, 30, 14)
    use_obv = st.checkbox("OBV (能量潮)", key='use_obv')
    st.caption("成交量累计，验证价格趋势。")
    use_vol = st.checkbox("量比", key='use_vol')
    st.caption("当日量与近期均量之比。")
    if use_vol:
        vol_period = st.slider("均量周期", 5, 30, 20)
    use_trend = st.checkbox("短期趋势强度", key='use_trend')
    st.caption("5日与20日线的距离，正为多头。")

# 至少勾选一个指标
selected_any = any([st.session_state[k] for k in all_keys])
if not selected_any:
    st.error("请在左侧至少选择一个技术指标！")
    st.stop()

# ========== 数据获取 ==========
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

# ========== 指标计算引擎（利用 pandas_ta） ==========
def compute_all_features(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    features = pd.DataFrame(index=df.index)

    # ---- 短线指标 ----
    if use_kdj:
        kdj = ta.kdj(high=high, low=low, close=close)
        features["kdj_k"] = kdj.iloc[:,0] / 100.0
        features["kdj_d"] = kdj.iloc[:,1] / 100.0
        features["kdj_j"] = kdj.iloc[:,2] / 100.0
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
        features["skdj_kd_diff"] = (skdj_k - skdj_d) / 100.0
    if use_rsi:
        rsi = ta.rsi(close=close, length=rsi_period)
        features["rsi"] = rsi / 100.0
    if use_wr:
        wr = ta.willr(high=high, low=low, close=close, length=wr_period)
        features["wr"] = (wr / -100.0)  # 0~1
    if use_bias:
        ma = close.rolling(bias_period).mean()
        bias = (close - ma) / ma
        features["bias"] = bias
    if use_cci:
        cci = ta.cci(high=high, low=low, close=close, length=cci_period)
        # 归一化到约-1..1，除以200裁剪
        cci_clip = cci.clip(-200, 200)
        features["cci"] = cci_clip / 200.0
    if use_roc:
        roc = ta.roc(close=close, length=roc_period)
        features["roc"] = roc / 100.0

    # ---- 长线指标 ----
    if use_ma:
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        features["ma5_dist"] = (close - ma5) / close
        features["ma20_dist"] = (close - ma20) / close
        features["ma60_dist"] = (close - ma60) / close
        # 多头排列强度：均线斜率比较（短均大于长均）
        features["ma_align"] = ((ma5 > ma10).astype(int) + (ma10 > ma20).astype(int) + (ma20 > ma60).astype(int)) / 3.0
    if use_macd:
        macd = ta.macd(close=close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        macd_hist = macd.iloc[:,1] - macd.iloc[:,2]  # 柱体
        features["macd_hist_norm"] = macd_hist / (close + 1e-10)
    if use_expma:
        ema_short = ta.ema(close=close, length=expma_short)
        ema_long = ta.ema(close=close, length=expma_long)
        features["expma_short_dist"] = (close - ema_short) / close
        features["expma_long_dist"] = (close - ema_long) / close
        features["expma_diff"] = (ema_short - ema_long) / close
    if use_boll:
        boll = ta.bbands(close=close, length=bb_period, std=bb_std)
        bb_lower = boll.iloc[:,0]
        bb_mid = boll.iloc[:,1]
        bb_upper = boll.iloc[:,2]
        features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    if use_sar:
        sar = ta.psar(high=high, low=low, close=close)
        features["sar_dist"] = (close - sar) / close
    if use_dmi:
        dmi = ta.adx(high=high, low=low, close=close, length=dmi_period)
        plus_di = dmi.iloc[:,0] / 100.0
        minus_di = dmi.iloc[:,1] / 100.0
        adx = dmi.iloc[:,2] / 100.0
        features["dmi_plus"] = plus_di
        features["dmi_minus"] = minus_di
        features["dmi_adx"] = adx
        features["dmi_diff"] = plus_di - minus_di
    if use_obv:
        obv = ta.obv(close=close, volume=volume)
        features["obv_change"] = obv.pct_change(5)  # 5日变化率
    if use_vol:
        vol_ma = volume.rolling(vol_period).mean()
        features["vol_ratio"] = volume / vol_ma
    if use_trend:
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        features["trend_strength"] = (ma5 - ma20) / (close + 1e-10)

    return features

# ========== 主分析 ==========
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
                # 当前最新特征（最后一行）
                current_feat = combined[feature_cols].iloc[-1:].values
                # 历史特征（排除最近20天，防止未来信息）
                hist_feat = combined[feature_cols].iloc[:-20].values

                if len(hist_feat) < 50:
                    st.warning("历史相似样本数较少，结果可能有偏差")

                # 标准化（让不同量纲的指标公平比较）
                scaler = StandardScaler()
                scaler.fit(hist_feat)
                hist_feat_scaled = scaler.transform(hist_feat)
                current_feat_scaled = scaler.transform(current_feat)

                sim = cosine_similarity(current_feat_scaled, hist_feat_scaled)[0]
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
