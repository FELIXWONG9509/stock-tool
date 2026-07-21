import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="多指标历史相似概率", layout="wide")
st.title("📈 多技术指标 · 历史相似匹配获利概率")
st.caption("选择固定搭配或自由组合，并可调整各指标参数，寻找历史上最相似的时刻，计算后续上涨概率。")

code = st.text_input("股票代码（如 600887）", "600887")
days_hold = st.selectbox("持仓周期（天）", [5, 10, 20, 50, 100, 150, 200, 300, 400], index=2)

# ========== 固定搭配定义 ==========
FIXED_COMBOS = {
    "自定义（手动选择）": {
        "说明": "在下方短线/长线区域自由勾选指标，完全自定义。",
        "适合周期": "不限",
        "keys": []
    },
    "MA 双均线": {
        "说明": "利用MA均线排列及价格与多条均线的关系，判断趋势方向。适合有明显趋势的单边行情。",
        "适合周期": "20天 ~ 100天",
        "keys": ["use_ma"]
    },
    "MACD + MA": {
        "说明": "用MACD判断动能变化，均线确认趋势方向，经典中长线趋势策略。",
        "适合周期": "20天 ~ 100天",
        "keys": ["use_macd", "use_ma"]
    },
    "KDJ + RSI": {
        "说明": "KDJ捕捉短期拐点，RSI过滤极值区域，适合短线搏反弹/回调。",
        "适合周期": "5天 ~ 20天",
        "keys": ["use_kdj", "use_rsi"]
    },
    "BOLL + MACD": {
        "说明": "MACD确认突破方向，布林带判断波动区间，常用于波段交易。",
        "适合周期": "10天 ~ 50天",
        "keys": ["use_boll", "use_macd"]
    },
    "SKDJ + VOL": {
        "说明": "慢速KDJ过滤杂讯，配合成交量验证，提高波段拐点可靠性。",
        "适合周期": "10天 ~ 30天",
        "keys": ["use_skdj", "use_vol"]
    },
    "DMI + MA": {
        "说明": "DMI判断趋势有无及方向，均线确认排列，适合中大波段持有。",
        "适合周期": "50天 ~ 150天",
        "keys": ["use_dmi", "use_ma"]
    },
    "EXPMA + SAR": {
        "说明": "EXPMA跟踪趋势，SAR提供动态止损位，适合单边行情。",
        "适合周期": "20天 ~ 100天",
        "keys": ["use_expma", "use_sar"]
    },
    "短线全能": {
        "说明": "KDJ + SKDJ + RSI + WR + BIAS + CCI + ROC，全面监控短线状态。",
        "适合周期": "5天 ~ 20天",
        "keys": ["use_kdj", "use_skdj", "use_rsi", "use_wr", "use_bias", "use_cci", "use_roc"]
    },
    "长线全能": {
        "说明": "MA + MACD + EXPMA + BOLL + SAR + DMI + OBV + VOL，全面监控趋势。",
        "适合周期": "50天 ~ 200天",
        "keys": ["use_ma", "use_macd", "use_expma", "use_boll", "use_sar", "use_dmi", "use_obv", "use_vol"]
    },
    "全能组合": {
        "说明": "所有指标全部启用，综合评估市场状态。",
        "适合周期": "不限",
        "keys": ["use_kdj", "use_skdj", "use_rsi", "use_wr", "use_bias", "use_cci", "use_roc",
                 "use_ma", "use_macd", "use_expma", "use_boll", "use_sar", "use_dmi", "use_obv", "use_vol", "use_trend"]
    }
}

# 所有可能的指标 key
short_keys = ['use_kdj', 'use_skdj', 'use_rsi', 'use_wr', 'use_bias', 'use_cci', 'use_roc']
long_keys = ['use_ma', 'use_macd', 'use_expma', 'use_boll', 'use_sar', 'use_dmi', 'use_obv', 'use_vol', 'use_trend']
all_keys = short_keys + long_keys

# 初始化 session_state
for k in all_keys:
    if k not in st.session_state:
        st.session_state[k] = False
if 'combo' not in st.session_state:
    st.session_state.combo = "自定义（手动选择）"

# ========== 侧边栏：固定搭配选项卡 ==========
with st.sidebar.expander("📦 固定搭配", expanded=True):
    combo_choice = st.radio(
        "选择一组经典指标组合",
        list(FIXED_COMBOS.keys()),
        index=list(FIXED_COMBOS.keys()).index(st.session_state.combo)
    )
    if combo_choice != st.session_state.combo:
        st.session_state.combo = combo_choice
        # 先全部取消
        for k in all_keys:
            st.session_state[k] = False
        # 再勾选该组合对应的指标
        for k in FIXED_COMBOS[combo_choice]["keys"]:
            st.session_state[k] = True
        st.rerun()

    info = FIXED_COMBOS[combo_choice]
    st.caption(f"**{combo_choice}**")
    st.caption(f"📖 {info['说明']}")
    st.caption(f"⏱️ 建议持仓周期：{info['适合周期']}")

# ========== 参数统一调整区（仅显示已勾选指标的参数） ==========
with st.sidebar.expander("🔧 参数调整", expanded=True):
    # 根据 session_state 决定显示哪些参数
    if st.session_state.use_skdj:
        skdj_n = st.slider("SKDJ N", 5, 30, 9, key='skdj_n')
        skdj_m = st.slider("SKDJ M", 2, 10, 3, key='skdj_m')
    if st.session_state.use_rsi:
        rsi_period = st.slider("RSI 周期", 5, 30, 14, key='rsi_period')
    if st.session_state.use_wr:
        wr_period = st.slider("WR 周期", 5, 30, 14, key='wr_period')
    if st.session_state.use_bias:
        bias_period = st.slider("BIAS 均线周期", 5, 60, 20, key='bias_period')
    if st.session_state.use_cci:
        cci_period = st.slider("CCI 周期", 5, 30, 20, key='cci_period')
    if st.session_state.use_roc:
        roc_period = st.slider("ROC 周期", 5, 30, 12, key='roc_period')
    if st.session_state.use_macd:
        macd_fast = st.slider("MACD 快线", 5, 30, 12, key='macd_fast')
        macd_slow = st.slider("MACD 慢线", 10, 40, 26, key='macd_slow')
        macd_signal = st.slider("MACD 信号线", 5, 15, 9, key='macd_signal')
    if st.session_state.use_expma:
        expma_short = st.slider("EXPMA 短期", 5, 30, 12, key='expma_short')
        expma_long = st.slider("EXPMA 长期", 20, 60, 50, key='expma_long')
    if st.session_state.use_boll:
        bb_period = st.slider("BOLL 周期", 10, 50, 20, key='bb_period')
        bb_std = st.slider("标准差倍数", 1, 4, 2, key='bb_std')
    if st.session_state.use_dmi:
        dmi_period = st.slider("DMI 周期", 5, 30, 14, key='dmi_period')
    if st.session_state.use_vol:
        vol_period = st.slider("均量周期", 5, 30, 20, key='vol_period')
    # 无参数指标不显示滑块

# ========== 短线指标区域（只负责勾选，不显示参数） ==========
with st.sidebar.expander("⚡ 短线指标（可增减）", expanded=True):
    use_kdj = st.checkbox("KDJ (随机指标)", key='use_kdj')
    st.caption("K/D/J三线，反映超买超卖与交叉信号。")
    use_skdj = st.checkbox("SKDJ (慢速随机指标)", key='use_skdj')
    st.caption("慢速平滑KDJ，适合波段拐点判断。")
    use_rsi = st.checkbox("RSI (相对强弱)", key='use_rsi')
    st.caption("0~100摆动，>70超买，<30超卖。")
    use_wr = st.checkbox("WR (威廉指标)", key='use_wr')
    st.caption("与KDJ类似，-80以下超卖，-20以上超买。")
    use_bias = st.checkbox("BIAS (乖离率)", key='use_bias')
    st.caption("收盘价与均线的偏离程度，捕捉回归机会。")
    use_cci = st.checkbox("CCI (商品通道指数)", key='use_cci')
    st.caption("突破+100/-100为强/弱势信号。")
    use_roc = st.checkbox("ROC (变动速率)", key='use_roc')
    st.caption("价格N日涨跌幅，衡量趋势速度。")

# ========== 长线指标区域（只负责勾选，不显示参数） ==========
with st.sidebar.expander("📊 长线指标（可增减）", expanded=True):
    use_ma = st.checkbox("MA (均线排列)", key='use_ma')
    st.caption("多周期均线位置与多头排列强度。")
    use_macd = st.checkbox("MACD", key='use_macd')
    st.caption("快慢线差与柱体，反映趋势动能。")
    use_expma = st.checkbox("EXPMA (指数均线)", key='use_expma')
    st.caption("近期价格偏重，反应更快。")
    use_boll = st.checkbox("BOLL (布林带)", key='use_boll')
    st.caption("上下轨与中轨的相对位置。")
    use_sar = st.checkbox("SAR (抛物线转向)", key='use_sar')
    st.caption("停损点，价格与SAR的距离反映趋势强度。")
    use_dmi = st.checkbox("DMI (趋向指标)", key='use_dmi')
    st.caption("PDI/MDI/ADX，判断趋势有无及方向。")
    use_obv = st.checkbox("OBV (能量潮)", key='use_obv')
    st.caption("成交量累计，验证价格趋势。")
    use_vol = st.checkbox("量比", key='use_vol')
    st.caption("当日量与近期均量之比。")
    use_trend = st.checkbox("短期趋势强度", key='use_trend')
    st.caption("5日与20日线的距离，正为多头。")

# 确保至少选了一个指标
selected_any = any([st.session_state[k] for k in all_keys])
if not selected_any:
    st.error("请在左侧至少选择一个技术指标！")
    st.stop()

# ========== 数据获取 ==========
@st.cache_data
def load_data(stock_code):
    max_retries = 3
    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                st.error(f"数据获取失败，已重试{max_retries}次。错误: {e}")
                return None

# ========== 指标计算引擎 ==========
def compute_all_features(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    features = pd.DataFrame(index=df.index)

    if use_kdj:
        low_min = low.rolling(9).min()
        high_max = high.rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
        k_val = rsv.copy()
        d_val = rsv.copy()
        for i in range(1, len(k_val)):
            k_val.iloc[i] = 2/3 * k_val.iloc[i-1] + 1/3 * rsv.iloc[i]
            d_val.iloc[i] = 2/3 * d_val.iloc[i-1] + 1/3 * k_val.iloc[i]
        features["kdj_k"] = k_val / 100.0
        features["kdj_d"] = d_val / 100.0
        features["kdj_j"] = (3 * k_val - 2 * d_val) / 100.0

    if use_skdj:
        low_n = low.rolling(skdj_n).min()
        high_n = high.rolling(skdj_n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-10) * 100
        k = rsv.ewm(alpha=1/skdj_m, adjust=False).mean()
        d = k.ewm(alpha=1/skdj_m, adjust=False).mean()
        skdj_k = d
        skdj_d = d.ewm(alpha=1/skdj_m, adjust=False).mean()
        features["skdj_k"] = skdj_k / 100.0
        features["skdj_d"] = skdj_d / 100.0
        features["skdj_kd_diff"] = (skdj_k - skdj_d) / 100.0

    if use_rsi:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        features["rsi"] = rsi / 100.0

    if use_wr:
        high_n = high.rolling(wr_period).max()
        low_n = low.rolling(wr_period).min()
        wr = (high_n - close) / (high_n - low_n + 1e-10) * -100
        features["wr"] = wr / -100.0

    if use_bias:
        ma = close.rolling(bias_period).mean()
        features["bias"] = (close - ma) / ma

    if use_cci:
        tp = (high + low + close) / 3
        ma_tp = tp.rolling(cci_period).mean()
        mad = tp.rolling(cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - ma_tp) / (0.015 * mad + 1e-10)
        cci_clipped = cci.clip(-200, 200)
        features["cci"] = cci_clipped / 200.0

    if use_roc:
        roc = close.pct_change(roc_period) * 100
        features["roc"] = roc / 100.0

    if use_ma:
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        features["ma5_dist"] = (close - ma5) / close
        features["ma20_dist"] = (close - ma20) / close
        features["ma60_dist"] = (close - ma60) / close
        features["ma_align"] = ((ma5 > ma10).astype(int) + (ma10 > ma20).astype(int) + (ma20 > ma60).astype(int)) / 3.0

    if use_macd:
        ema_fast = close.ewm(span=macd_fast).mean()
        ema_slow = close.ewm(span=macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal = macd_line.ewm(span=macd_signal).mean()
        macd_hist = macd_line - signal
        features["macd_hist_norm"] = macd_hist / (close + 1e-10)

    if use_expma:
        ema_short = close.ewm(span=expma_short).mean()
        ema_long = close.ewm(span=expma_long).mean()
        features["expma_short_dist"] = (close - ema_short) / close
        features["expma_long_dist"] = (close - ema_long) / close
        features["expma_diff"] = (ema_short - ema_long) / close

    if use_boll:
        bb_mid = close.rolling(bb_period).mean()
        bb_std_val = close.rolling(bb_period).std()
        bb_upper = bb_mid + bb_std * bb_std_val
        bb_lower = bb_mid - bb_std * bb_std_val
        features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    if use_sar:
        af = 0.02
        max_af = 0.2
        sar = pd.Series(np.nan, index=close.index)
        ep = low.copy()
        trend = pd.Series(1, index=close.index)
        for i in range(1, len(close)):
            if trend.iloc[i-1] == 1:
                sar.iloc[i] = sar.iloc[i-1] + af * (ep.iloc[i-1] - sar.iloc[i-1])
                if low.iloc[i] < sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = max(high.iloc[i], high.iloc[i-1]) if i>0 else high.iloc[i]
                    ep.iloc[i] = low.iloc[i]
                    af = 0.02
                else:
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af = min(af + 0.02, max_af)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
            else:
                sar.iloc[i] = sar.iloc[i-1] + af * (ep.iloc[i-1] - sar.iloc[i-1])
                if high.iloc[i] > sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = min(low.iloc[i], low.iloc[i-1]) if i>0 else low.iloc[i]
                    ep.iloc[i] = high.iloc[i]
                    af = 0.02
                else:
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af = min(af + 0.02, max_af)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
        sar.iloc[0] = close.iloc[0]
        features["sar_dist"] = (close - sar) / close

    if use_dmi:
        up_move = high.diff()
        down_move = -low.diff()
        pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        mdm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = pd.Series(tr).ewm(alpha=1/dmi_period, adjust=False).mean()
        pdm_smooth = pd.Series(pdm).ewm(alpha=1/dmi_period, adjust=False).mean()
        mdm_smooth = pd.Series(mdm).ewm(alpha=1/dmi_period, adjust=False).mean()
        pdi = 100 * pdm_smooth / atr
        mdi = 100 * mdm_smooth / atr
        dx = (abs(pdi - mdi) / (pdi + mdi + 1e-10)) * 100
        adx = dx.ewm(alpha=1/dmi_period, adjust=False).mean()
        features["dmi_plus"] = pdi / 100.0
        features["dmi_minus"] = mdi / 100.0
        features["dmi_adx"] = adx / 100.0
        features["dmi_diff"] = (pdi - mdi) / 100.0

    if use_obv:
        sign = np.sign(close.diff())
        obv = (sign * volume).cumsum()
        features["obv_change"] = obv.pct_change(5)

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
                current_feat = combined[feature_cols].iloc[-1:].values
                hist_feat = combined[feature_cols].iloc[:-20].values

                if len(hist_feat) < 50:
                    st.warning("历史相似样本数较少，结果可能有偏差")

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
