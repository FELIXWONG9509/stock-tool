import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from datetime import datetime, timedelta, date
import io
import json
import re

st.set_page_config(page_title="多指标历史相似概率", layout="wide")
st.caption("选择经典组合或自由搭配，指定分析日期，寻找历史上最相似的时刻，计算后续上涨概率。")

def extract_code_from_filename(filename):
    match = re.search(r'(sh|sz)\d{6}', filename, re.IGNORECASE)
    if match:
        return match.group(0)
    match = re.search(r'(\d{6})', filename)
    if match:
        code = match.group(1)
        return 'sh' + code if code.startswith('6') else 'sz' + code
    return None

uploaded_file = st.file_uploader("📤 上传东方财富下载的 JSON 或 CSV 文件", type=["json", "csv"])

if uploaded_file is not None:
    try:
        file_name = uploaded_file.name.lower()
        auto_code = extract_code_from_filename(uploaded_file.name)
        if auto_code and 'auto_code_set' not in st.session_state:
            st.session_state['auto_code'] = auto_code
            st.session_state['auto_code_set'] = True

        if file_name.endswith(".json"):
            content = uploaded_file.getvalue().decode("utf-8-sig")
            data_json = json.loads(content)
            if "data" in data_json and "klines" in data_json["data"]:
                klines = data_json["data"]["klines"]
                if not klines:
                    st.error("JSON文件中没有历史数据。")
                    st.stop()
                csv_text = "\n".join(klines)
                col_names = ["date","open","close","high","low","volume",
                             "amount","amplitude","pct_change","change","turnover"]
                df_upload = pd.read_csv(io.StringIO(csv_text), header=None, names=col_names)
                df_upload = df_upload[["date","open","close","high","low","volume"]].copy()
            else:
                st.error("JSON格式不正确，缺少 data.klines 字段。")
                st.stop()
        else:
            content = uploaded_file.getvalue().decode("utf-8-sig")
            lines = content.strip().split("\n")
            if not lines:
                st.error("文件为空。")
                st.stop()
            first_line = lines[0].strip()
            if first_line and first_line[0].isdigit():
                columns = ["date","open","close","high","low","volume"]
                df_upload = pd.read_csv(io.StringIO(content), header=None)
                df_upload = df_upload.iloc[:, :6]
                df_upload.columns = columns
            else:
                df_upload = pd.read_csv(io.StringIO(content))
                rename_map = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume",
                              "date":"date","open":"open","close":"close","high":"high","low":"low","volume":"volume"}
                df_upload.columns = [col.strip().lower() for col in df_upload.columns]
                col_map = {col:rename_map[col] for col in df_upload.columns if col in rename_map}
                df_upload = df_upload.rename(columns=col_map)
                required = ["date","open","close","high","low","volume"]
                if not all(c in df_upload.columns for c in required):
                    st.error("CSV缺少必要列。")
                    st.stop()

        df_upload["date"] = pd.to_datetime(df_upload["date"], errors="coerce")
        for col in ["open","close","high","low","volume"]:
            df_upload[col] = df_upload[col].astype(str).str.replace(",","").str.strip()
            df_upload[col] = pd.to_numeric(df_upload[col], errors="coerce")
        df_upload[["open","close","high","low"]] = df_upload[["open","close","high","low"]].abs()
        df_upload["volume"] = df_upload["volume"].abs()
        df_upload = df_upload.dropna(subset=["date","open","close","high","low","volume"]).sort_values("date").reset_index(drop=True)

        if len(df_upload) < 60:
            st.error("有效数据不足（少于60个交易日），请检查文件。")
            st.stop()

        st.session_state["data"] = df_upload
        st.success(f"✅ 上传成功，共 {len(df_upload)} 条有效数据。")
        st.write("📋 前3行收盘价：", df_upload["close"].head(3).tolist())
    except Exception as e:
        st.error(f"文件解析失败：{e}")

default_code = st.session_state.get("auto_code", "600887")
code = st.text_input("股票代码", value=default_code)

analysis_date = st.date_input("📅 分析日期（默认今天）", date.today())
days_hold = st.selectbox("持仓周期（天）", [5, 10, 20, 50, 100, 150, 200, 300, 400], index=2)

if "data" not in st.session_state:
    st.info("👆 请先上传历史数据文件（JSON 或 CSV）。")
    st.stop()
data = st.session_state["data"]

FIXED_COMBOS = {
    "自定义（手动选择）": {"说明":"自由勾选指标。","适合周期":"不限","类别":"","keys":[]},
    "BOLL + KDJ 经典组合": {"说明":"布林带+KDJ，趋势与短线结合。","适合周期":"10~60天","类别":"经典组合","keys":["use_boll","use_kdj"]},
    "MACD + OBV 量价组合": {"说明":"MACD趋势+OBV能量验证。","适合周期":"20~100天","类别":"经典组合","keys":["use_macd","use_obv"]},
    "EXPMA + CCI 短线组合": {"说明":"EXPMA支撑压力+CCI超买超卖。","适合周期":"5~30天","类别":"经典组合","keys":["use_expma","use_cci"]},
    "DMI + SAR 多空组合": {"说明":"DMI趋势强度+SAR止损。","适合周期":"20~150天","类别":"经典组合","keys":["use_dmi","use_sar"]}
}

display_to_combo = {}
combo_options = ["自定义（手动选择）"]
display_to_combo["自定义（手动选择）"] = "自定义（手动选择）"
combo_options.append("── 经典组合 ──")
for name, info in FIXED_COMBOS.items():
    if info["类别"] == "经典组合":
        display_name = f"   {name}"
        combo_options.append(display_name)
        display_to_combo[display_name] = name

short_keys = ['use_kdj','use_skdj','use_rsi','use_wr','use_bias','use_cci','use_roc']
long_keys = ['use_ma','use_macd','use_expma','use_boll','use_sar','use_dmi','use_obv','use_vol','use_trend']
all_keys = short_keys + long_keys

for k in all_keys:
    if k not in st.session_state:
        st.session_state[k] = False
if 'combo' not in st.session_state:
    st.session_state.combo = "自定义（手动选择）"

if st.sidebar.button("🔄 重置所有指标"):
    for k in all_keys:
        st.session_state[k] = False
    st.session_state.combo = "自定义（手动选择）"
    st.rerun()

with st.sidebar.expander("📦 固定搭配", expanded=True):
    current_display = None
    for disp, combo in display_to_combo.items():
        if combo == st.session_state.combo:
            current_display = disp
            break
    if current_display is None:
        current_display = "自定义（手动选择）"
    selected_display = st.selectbox("选择一组经典指标组合", combo_options,
                                    index=combo_options.index(current_display) if current_display in combo_options else 0)
    if selected_display.startswith("──"):
        selected_display = "自定义（手动选择）"
        st.warning("请选择一个具体组合。")
    chosen_combo = display_to_combo.get(selected_display, "自定义（手动选择）")
    if chosen_combo != st.session_state.combo:
        st.session_state.combo = chosen_combo
        for k in all_keys:
            st.session_state[k] = False
        for k in FIXED_COMBOS[chosen_combo]["keys"]:
            st.session_state[k] = True
        st.rerun()
    info = FIXED_COMBOS[chosen_combo]
    st.caption(f"**{chosen_combo}**")
    st.caption(f"📖 {info['说明']}")
    st.caption(f"⏱️ 建议持仓周期：{info['适合周期']}")

params = {}
with st.sidebar.expander("🔧 参数调整", expanded=True):
    params['use_kdj'] = st.session_state.use_kdj
    params['kdj_n'] = st.slider("KDJ 周期", 5, 30, 9, key='kdj_n') if params['use_kdj'] else 9

    params['use_skdj'] = st.session_state.use_skdj
    params['skdj_n'] = st.slider("SKDJ N", 5, 30, 9, key='skdj_n') if params['use_skdj'] else 9
    params['skdj_m'] = st.slider("SKDJ M", 2, 10, 3, key='skdj_m') if params['use_skdj'] else 3

    params['use_rsi'] = st.session_state.use_rsi
    params['rsi_period'] = st.slider("RSI 周期", 5, 30, 14, key='rsi_period') if params['use_rsi'] else 14

    params['use_wr'] = st.session_state.use_wr
    params['wr_period'] = st.slider("WR 周期", 5, 30, 14, key='wr_period') if params['use_wr'] else 14

    params['use_bias'] = st.session_state.use_bias
    params['bias_period'] = st.slider("BIAS 均线周期", 5, 60, 20, key='bias_period') if params['use_bias'] else 20

    params['use_cci'] = st.session_state.use_cci
    params['cci_period'] = st.slider("CCI 周期", 5, 30, 20, key='cci_period') if params['use_cci'] else 20

    params['use_roc'] = st.session_state.use_roc
    params['roc_period'] = st.slider("ROC 周期", 5, 30, 12, key='roc_period') if params['use_roc'] else 12

    params['use_ma'] = st.session_state.use_ma
    params['ma_fast'] = st.slider("MA 快线周期", 2, 30, 5, key='ma_fast') if params['use_ma'] else 5
    params['ma_slow'] = st.slider("MA 慢线周期", 5, 120, 20, key='ma_slow') if params['use_ma'] else 20

    params['use_macd'] = st.session_state.use_macd
    params['macd_fast'] = st.slider("MACD 快线", 5, 30, 12, key='macd_fast') if params['use_macd'] else 12
    params['macd_slow'] = st.slider("MACD 慢线", 10, 40, 26, key='macd_slow') if params['use_macd'] else 26
    params['macd_signal'] = st.slider("MACD 信号线", 5, 15, 9, key='macd_signal') if params['use_macd'] else 9

    params['use_expma'] = st.session_state.use_expma
    params['expma_short'] = st.slider("EXPMA 短期", 5, 30, 12, key='expma_short') if params['use_expma'] else 12
    params['expma_long'] = st.slider("EXPMA 长期", 20, 60, 50, key='expma_long') if params['use_expma'] else 50

    params['use_boll'] = st.session_state.use_boll
    params['bb_period'] = st.slider("BOLL 周期", 10, 50, 20, key='bb_period') if params['use_boll'] else 20
    params['bb_std'] = st.slider("标准差倍数", 1, 4, 2, key='bb_std') if params['use_boll'] else 2

    params['use_dmi'] = st.session_state.use_dmi
    params['dmi_period'] = st.slider("DMI 周期", 5, 30, 14, key='dmi_period') if params['use_dmi'] else 14

    params['use_vol'] = st.session_state.use_vol
    params['vol_period'] = st.slider("均量周期", 5, 30, 20, key='vol_period') if params['use_vol'] else 20

    params['use_sar'] = st.session_state.use_sar
    params['use_obv'] = st.session_state.use_obv
    params['use_trend'] = st.session_state.use_trend

with st.sidebar.expander("⚡ 短线指标", expanded=True):
    use_kdj = st.checkbox("KDJ", key='use_kdj'); st.caption("超买超卖")
    use_skdj = st.checkbox("SKDJ", key='use_skdj'); st.caption("慢速KDJ")
    use_rsi = st.checkbox("RSI", key='use_rsi'); st.caption("相对强弱")
    use_wr = st.checkbox("WR", key='use_wr'); st.caption("威廉指标")
    use_bias = st.checkbox("BIAS", key='use_bias'); st.caption("乖离率")
    use_cci = st.checkbox("CCI", key='use_cci'); st.caption("商品通道指数")
    use_roc = st.checkbox("ROC", key='use_roc'); st.caption("变动速率")

with st.sidebar.expander("📊 长线指标", expanded=True):
    use_ma = st.checkbox("MA", key='use_ma'); st.caption("均线排列")
    use_macd = st.checkbox("MACD", key='use_macd'); st.caption("趋势动能")
    use_expma = st.checkbox("EXPMA", key='use_expma'); st.caption("指数均线")
    use_boll = st.checkbox("BOLL", key='use_boll'); st.caption("布林带")
    use_sar = st.checkbox("SAR", key='use_sar'); st.caption("抛物线转向")
    use_dmi = st.checkbox("DMI", key='use_dmi'); st.caption("趋向指标")
    use_obv = st.checkbox("OBV", key='use_obv'); st.caption("能量潮")
    use_vol = st.checkbox("量比", key='use_vol'); st.caption("放量缩量")
    use_trend = st.checkbox("短期趋势强度", key='use_trend'); st.caption("5日/20日")

if not any([st.session_state[k] for k in all_keys]):
    st.error("请在左侧至少选择一个技术指标！")
    st.stop()

# ========== 纯Python指标计算（无任何第三方库依赖） ==========
def compute_all_features(df, p):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    features = pd.DataFrame(index=df.index)

    if p['use_kdj']:
        n = p['kdj_n']
        low_list = pd.Series(low).rolling(n).min().values
        high_list = pd.Series(high).rolling(n).max().values
        rsv = np.where((high_list - low_list) != 0, (close - low_list) / (high_list - low_list) * 100, 0.0)
        k = np.zeros_like(rsv)
        d = np.zeros_like(rsv)
        for i in range(1, len(rsv)):
            k[i] = 2/3 * k[i-1] + 1/3 * rsv[i]
            d[i] = 2/3 * d[i-1] + 1/3 * k[i]
        features["kdj_k"] = k / 100.0
        features["kdj_d"] = d / 100.0
        features["kdj_j"] = (3 * k - 2 * d) / 100.0

    if p['use_skdj']:
        n, m = p['skdj_n'], p['skdj_m']
        low_n = pd.Series(low).rolling(n).min().values
        high_n = pd.Series(high).rolling(n).max().values
        rsv = np.where((high_n - low_n) != 0, (close - low_n) / (high_n - low_n) * 100, 0.0)
        k_raw = np.zeros_like(rsv)
        d_raw = np.zeros_like(rsv)
        for i in range(1, len(rsv)):
            k_raw[i] = 2/3 * k_raw[i-1] + 1/3 * rsv[i]
            d_raw[i] = 2/3 * d_raw[i-1] + 1/3 * k_raw[i]
        # 慢速平滑
        skdj_k = pd.Series(d_raw).ewm(alpha=1/m, adjust=False).mean().values
        skdj_d = pd.Series(skdj_k).ewm(alpha=1/m, adjust=False).mean().values
        features["skdj_k"] = skdj_k / 100.0
        features["skdj_d"] = skdj_d / 100.0
        features["skdj_kd_diff"] = (skdj_k - skdj_d) / 100.0

    if p['use_rsi']:
        period = p['rsi_period']
        delta = np.diff(close, prepend=close[0])
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features["rsi"] = rsi / 100.0

    if p['use_wr']:
        period = p['wr_period']
        high_n = pd.Series(high).rolling(period).max().values
        low_n = pd.Series(low).rolling(period).min().values
        wr = (high_n - close) / (high_n - low_n + 1e-10) * -100
        features["wr"] = wr / -100.0

    if p['use_bias']:
        ma = pd.Series(close).rolling(p['bias_period']).mean().values
        features["bias"] = (close - ma) / (ma + 1e-10)

    if p['use_cci']:
        tp = (high + low + close) / 3
        ma_tp = pd.Series(tp).rolling(p['cci_period']).mean().values
        mad = pd.Series(tp).rolling(p['cci_period']).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
        cci = (tp - ma_tp) / (0.015 * mad + 1e-10)
        features["cci"] = np.clip(cci, -200, 200) / 200.0

    if p['use_roc']:
        roc = (close[p['roc_period']:] - close[:-p['roc_period']]) / close[:-p['roc_period']] * 100
        roc_full = np.concatenate([np.full(p['roc_period'], np.nan), roc])
        features["roc"] = roc_full / 100.0

    if p['use_ma']:
        fast_ma = pd.Series(close).rolling(p['ma_fast']).mean().values
        slow_ma = pd.Series(close).rolling(p['ma_slow']).mean().values
        features["ma_fast_dist"] = (close - fast_ma) / (close + 1e-10)
        features["ma_slow_dist"] = (close - slow_ma) / (close + 1e-10)
        features["ma_cross"] = (fast_ma - slow_ma) / (close + 1e-10)

    if p['use_macd']:
        ema_fast = pd.Series(close).ewm(span=p['macd_fast']).mean().values
        ema_slow = pd.Series(close).ewm(span=p['macd_slow']).mean().values
        macd_line = ema_fast - ema_slow
        signal = pd.Series(macd_line).ewm(span=p['macd_signal']).mean().values
        macd_hist = macd_line - signal
        features["macd_hist_norm"] = macd_hist / (close + 1e-10)

    if p['use_expma']:
        ema_short = pd.Series(close).ewm(span=p['expma_short']).mean().values
        ema_long = pd.Series(close).ewm(span=p['expma_long']).mean().values
        features["expma_short_dist"] = (close - ema_short) / (close + 1e-10)
        features["expma_long_dist"] = (close - ema_long) / (close + 1e-10)
        features["expma_diff"] = (ema_short - ema_long) / (close + 1e-10)

    if p['use_boll']:
        mid = pd.Series(close).rolling(p['bb_period']).mean().values
        std = pd.Series(close).rolling(p['bb_period']).std().values
        upper = mid + p['bb_std'] * std
        lower = mid - p['bb_std'] * std
        features["bb_position"] = (close - lower) / (upper - lower + 1e-10)

    if p['use_sar']:
        af, max_af = 0.02, 0.2
        sar = np.zeros(len(close))
        ep = np.zeros(len(close))
        trend = np.ones(len(close))
        sar[0], ep[0] = close[0], low[0]
        for i in range(1, len(close)):
            sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
            if trend[i-1] == 1:
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = max(high[i], high[i-1])
                    ep[i] = low[i]
                    af = 0.02
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af = min(af + 0.02, max_af)
                    else:
                        ep[i] = ep[i-1]
            else:
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = min(low[i], low[i-1])
                    ep[i] = high[i]
                    af = 0.02
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af = min(af + 0.02, max_af)
                    else:
                        ep[i] = ep[i-1]
        features["sar_dist"] = (close - sar) / (close + 1e-10)

    if p['use_dmi']:
        period = p['dmi_period']
        up_move = np.diff(high, prepend=high[0])
        down_move = -np.diff(low, prepend=low[0])
        pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        mdm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        pdm_smooth = pd.Series(pdm).ewm(alpha=1/period, adjust=False).mean().values
        mdm_smooth = pd.Series(mdm).ewm(alpha=1/period, adjust=False).mean().values
        pdi = 100 * pdm_smooth / (atr + 1e-10)
        mdi = 100 * mdm_smooth / (atr + 1e-10)
        dx = np.abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        features["dmi_plus"] = pdi / 100.0
        features["dmi_minus"] = mdi / 100.0
        features["dmi_adx"] = adx / 100.0
        features["dmi_diff"] = (pdi - mdi) / 100.0

    if p['use_obv']:
        sign = np.sign(np.diff(close, prepend=close[0]))
        obv = np.cumsum(sign * volume)
        features["obv_change"] = pd.Series(obv).pct_change(5).values

    if p['use_vol']:
        vol_ma = pd.Series(volume).rolling(p['vol_period']).mean().values
        features["vol_ratio"] = volume / vol_ma

    if p['use_trend']:
        ma5 = pd.Series(close).rolling(5).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        features["trend_strength"] = (ma5 - ma20) / (close + 1e-10)

    return features

# ========== 分析按钮 ==========
if st.button("🔍 开始分析"):
    if not code:
        st.warning("请输入股票代码")
    else:
        features = compute_all_features(data, params)
        combined = pd.concat([data[["date","close"]], features], axis=1)

        valid_cols = ["date", "close"] + [col for col in features.columns if features[col].notna().any()]
        combined = combined[valid_cols]
        combined = combined.ffill().bfill().fillna(0)

        if len(combined) < 100:
            st.error(f"有效历史数据不足（当前仅 {len(combined)} 天）。\n数据范围：{data['date'].min().date()} 至 {data['date'].max().date()}，分析日期：{analysis_date}。")
            st.stop()

        target_date = pd.to_datetime(analysis_date)
        date_rows = combined[combined["date"] == target_date]
        if date_rows.empty:
            st.error(f"所选日期 {target_date.date()} 在数据中不存在。")
        else:
            target_idx = date_rows.index[0]
            target_close = combined.loc[target_idx, "close"]
            st.success(f"📌 {target_date.date()} 收盘价：{target_close:.2f} 元")

            feature_cols = [col for col in combined.columns if col not in ["date","close"]]
            current_feat = combined.loc[target_idx, feature_cols].values.reshape(1, -1)

            exclude_start = max(0, target_idx - 20)
            exclude_end = min(len(combined), target_idx + 21)
            hist_mask = np.ones(len(combined), dtype=bool)
            hist_mask[exclude_start:exclude_end] = False
            hist_feat = combined.loc[hist_mask, feature_cols].values

            if hist_feat.shape[1] > 0:
                std = np.std(hist_feat, axis=0)
                zero_var_mask = std == 0
                if zero_var_mask.any():
                    feature_cols = [col for i, col in enumerate(feature_cols) if not zero_var_mask[i]]
                    current_feat = combined.loc[target_idx, feature_cols].values.reshape(1, -1)
                    hist_feat = combined.loc[hist_mask, feature_cols].values

            if len(hist_feat) < 30 or hist_feat.shape[1] == 0:
                st.error("可用于分析的特征列不足，请尝试选择更多指标或调整参数。")
                st.stop()

            scaler = StandardScaler()
            scaler.fit(hist_feat)
            sim = cosine_similarity(scaler.transform(current_feat), scaler.transform(hist_feat))[0]
            top_k = min(30, len(sim))
            top_idx = np.argsort(sim)[-top_k:][::-1]
            hist_combined_idx = combined.loc[hist_mask].index.values
            matched_indices = hist_combined_idx[top_idx]
            sim_scores = sim[top_idx]

            with st.expander("📊 当前分析日期的技术指标数值"):
                cur_df = pd.DataFrame({"指标":feature_cols, "数值":combined.loc[target_idx, feature_cols].values})
                st.dataframe(cur_df.set_index("指标"), use_container_width=True)

            with st.expander("📊 最相似历史日期的技术指标数值（前5个）"):
                top5 = matched_indices[:5]
                sim_df = combined.loc[top5, ["date"]+feature_cols].copy()
                sim_df["日期"] = sim_df["date"].dt.date
                st.dataframe(sim_df.drop(columns="date").set_index("日期"), use_container_width=True)

            close_series = combined["close"].reset_index(drop=True)
            rets = []
            for idx in matched_indices:
                if idx + days_hold < len(close_series):
                    ret = (close_series.iloc[idx + days_hold] / close_series.iloc[idx]) - 1
                    rets.append(ret)

            if len(rets) < 5:
                st.error("有效相似样本太少，无法统计")
            else:
                ret_arr = np.array(rets)
                win_rate = (ret_arr > 0).mean()
                avg_ret = ret_arr.mean()
                pos = ret_arr[ret_arr > 0]; neg = ret_arr[ret_arr < 0]
                pl_ratio = pos.mean() / abs(neg.mean()) if len(pos) and len(neg) else np.inf
                col1, col2, col3 = st.columns(3)
                col1.metric("上涨概率", f"{win_rate:.1%}")
                col2.metric("平均收益", f"{avg_ret:.2%}")
                col3.metric("盈亏比", f"{pl_ratio:.2f}")
                if win_rate > 0.55 and avg_ret > 0:
                    st.success("✅ 概率买点信号")
                else:
                    st.info("ℹ️ 未达到高概率买点标准")
                fig = px.histogram(ret_arr, nbins=20, title=f"相似历史持有{days_hold}天收益分布")
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("相似历史日期及相似度"):
                    match_dates = combined.loc[matched_indices, "date"].reset_index(drop=True)
                    st.dataframe(pd.DataFrame({"历史日期":match_dates.values[:len(sim_scores)], "相似度":sim_scores}).head(20))

                st.warning("⚠️ 风险提示：历史表现不代表未来，本工具仅供参考，不构成投资建议。")
