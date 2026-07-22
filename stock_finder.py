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

# ---------- 自动提取文件名中的股票代码 ----------
def extract_code_from_filename(filename):
    match = re.search(r'(sh|sz)\d{6}', filename, re.IGNORECASE)
    if match:
        return match.group(0)
    match = re.search(r'(\d{6})', filename)
    if match:
        code = match.group(1)
        return 'sh' + code if code.startswith('6') else 'sz' + code
    return None

# 文件上传
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
                rows = [line.split(",") for line in klines]
                rows = [r[:6] for r in rows]
                df_upload = pd.DataFrame(rows, columns=["date","open","close","high","low","volume"])
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

        # 清洗数据（强化版）
        df_upload["date"] = pd.to_datetime(df_upload["date"], errors="coerce")
        # 强制转字符串再去除千分位逗号，然后转float
        for col in ["open","close","high","low","volume"]:
            df_upload[col] = df_upload[col].astype(str).str.replace(",", "").str.strip()
            df_upload[col] = pd.to_numeric(df_upload[col], errors="coerce")
        # 再次确保是 float 类型，避免 object 残留
        df_upload[["open","close","high","low","volume"]] = df_upload[["open","close","high","low","volume"]].astype(float)
        df_upload = df_upload.dropna(subset=["date","open","close","high","low","volume"]).sort_values("date").reset_index(drop=True)

        if len(df_upload) < 60:
            st.error("有效数据不足（少于60个交易日），请检查文件。")
            st.stop()

        st.session_state["data"] = df_upload
        st.success(f"✅ 上传成功，共 {len(df_upload)} 条有效数据。")
        st.write("📋 数据预览（前3行收盘价）：", df_upload["close"].head(3).tolist())
        st.write("📋 close列类型：", df_upload["close"].dtype)
    except Exception as e:
        st.error(f"文件解析失败：{e}")

# 股票代码输入
default_code = st.session_state.get("auto_code", "600887")
code = st.text_input("股票代码", value=default_code)

analysis_date = st.date_input("📅 分析日期（默认今天）", date.today())
days_hold = st.selectbox("持仓周期（天）", [5, 10, 20, 50, 100, 150, 200, 300, 400], index=2)

if "data" not in st.session_state:
    st.info("👆 请先上传历史数据文件（JSON 或 CSV）。")
    st.stop()
data = st.session_state["data"]

# ========== 固定搭配 ==========
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

# ========== 参数调整 ==========
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

# ========== 指标勾选区 ==========
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

# ========== 指标计算引擎 ==========
def compute_all_features(df, p):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    features = pd.DataFrame(index=df.index)

    if p['use_kdj']:
        n = p['kdj_n']
        low_min = low.rolling(n).min()
        high_max = high.rolling(n).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
        k_val = rsv.copy()
        d_val = rsv.copy()
        for i in range(1, len(k_val)):
            k_val.iloc[i] = 2/3 * k_val.iloc[i-1] + 1/3 * rsv.iloc[i]
            d_val.iloc[i] = 2/3 * d_val.iloc[i-1] + 1/3 * k_val.iloc[i]
        features["kdj_k"] = k_val / 100.0
        features["kdj_d"] = d_val / 100.0
        features["kdj_j"] = (3 * k_val - 2 * d_val) / 100.0

    # 后续指标计算保持不变（省略，与上一版相同）...
    # 这里由于篇幅，不重复所有指标，你替换时应保留完整的其他指标代码。
    # 请用之前提供的完整指标代码补充进来，确保所有指标逻辑完整。

    return features

# ========== 分析按钮 ==========
if st.button("🔍 开始分析"):
    if not code:
        st.warning("请输入股票代码")
    else:
        features = compute_all_features(data, params)
        combined = pd.concat([data[["date","close"]], features], axis=1)
        # 调试信息
        with st.expander("🔧 调试信息（如遇0天请展开）"):
            st.write(f"原始数据行数：{len(data)}")
            st.write(f"特征列：{list(features.columns)}")
            st.write(f"合并后总行数：{len(combined)}")
            st.write("前5行 close 值：", combined["close"].head(5).tolist())
            st.write("前5行 kdj_k 值：", combined["kdj_k"].head(10).tolist() if "kdj_k" in combined else "无")
            st.write("close列类型：", combined["close"].dtype)
            st.write("kdj_k列类型：", combined["kdj_k"].dtype if "kdj_k" in combined else "无")
        combined = combined.dropna()
        st.write("去NaN后行数：", len(combined))
        if len(combined) < 100:
            st.error(f"有效历史数据不足（当前仅 {len(combined)} 天）。\n数据范围：{data['date'].min().date()} 至 {data['date'].max().date()}，分析日期：{analysis_date}。")
            st.stop()

        # 后续分析逻辑（与之前相同）
        # 这里省略，请保留上一版完整分析代码。
