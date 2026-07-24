import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from datetime import datetime, timedelta, date
import calendar
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

# ---------- 初始化日期状态 ----------
if 'analysis_year' not in st.session_state:
    st.session_state.analysis_year = date.today().year
if 'analysis_month' not in st.session_state:
    st.session_state.analysis_month = date.today().month
if 'analysis_day' not in st.session_state:
    st.session_state.analysis_day = date.today().day

# ---------- 股票代码输入 ----------
default_code = st.session_state.get("auto_code", "600887")
code = st.text_input("股票代码", value=default_code)

# ---------- 数据下载按钮 ----------
secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
download_url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end=20500101&lmt=10000"
st.link_button("🌐 打开数据下载页面（右键另存为 .json）", download_url)
st.caption("点击上方按钮，在新页面中按 Ctrl+S 保存为 股票代码.json，然后上传至下方。")

# ---------- 文件上传 ----------
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
    except Exception as e:
        st.error(f"文件解析失败：{e}")

# ---------- 中文日期选择（年/月/日下拉框） ----------
# 年份范围：当前年份向前推30年
current_year = date.today().year
year_options = list(range(current_year - 30, current_year + 1))

# 月份中文名
month_names = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
month_values = list(range(1, 13))

# 星期中文名
weekday_cn = ["一", "二", "三", "四", "五", "六", "日"]

col1, col2, col3, col_today = st.columns([2, 2, 2, 1])
with col1:
    year = st.selectbox("年", year_options, index=year_options.index(st.session_state.analysis_year) if st.session_state.analysis_year in year_options else 0)
with col2:
    month = st.selectbox("月", month_values, format_func=lambda m: month_names[m-1], index=month_values.index(st.session_state.analysis_month) if st.session_state.analysis_month in month_values else 0)
with col3:
    # 根据年月计算当月最大天数
    max_day = calendar.monthrange(year, month)[1]
    day = st.selectbox("日", range(1, max_day+1), index=min(st.session_state.analysis_day, max_day)-1)
with col_today:
    st.markdown("### ")
    if st.button("📌 今天"):
        st.session_state.analysis_year = current_year
        st.session_state.analysis_month = date.today().month
        st.session_state.analysis_day = date.today().day
        st.rerun()

# 更新 session_state
st.session_state.analysis_year = year
st.session_state.analysis_month = month
st.session_state.analysis_day = day

# 显示完整中文日期
selected_date = date(year, month, day)
st.caption(f"📌 当前选择：{year}年{month}月{day}日 星期{weekday_cn[selected_date.weekday()]}")

days_hold = st.selectbox("持仓周期（天）", [5, 10, 20, 50, 100, 150, 200, 300, 400], index=2)

# ========== 侧边栏（不变） ==========
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

short_keys = ['use_kdj','use_skdj','use_rsi','use_wr','use_bias','use_cci','use_roc','use_trend']
long_keys = ['use_ma','use_macd','use_expma','use_boll','use_sar','use_dmi','use_obv','use_vol','use_trend_long']
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
    params['rsi_period'] = st.slider("RSI 周期（默认14）", 5, 30, 14, key='rsi_period') if params['use_rsi'] else 14

    params['use_wr'] = st.session_state.use_wr
    params['wr_period'] = st.slider("WR 周期", 5, 30, 14, key='wr_period') if params['use_wr'] else 14

    params['use_bias'] = st.session_state.use_bias
    params['bias_period'] = st.slider("BIAS 均线周期", 5, 60, 20, key='bias_period') if params['use_bias'] else 20

    params['use_cci'] = st.session_state.use_cci
    params['cci_period'] = st.slider("CCI 周期", 5, 30, 20, key='cci_period') if params['use_cci'] else 20

    params['use_roc'] = st.session_state.use_roc
    params['roc_period'] = st.slider("ROC 周期", 5, 30, 12, key='roc_period') if params['use_roc'] else 12

    params['use_trend'] = st.session_state.use_trend
    if params['use_trend']:
        st.markdown("**📈 短期趋势强度**")
        params['trend_fast'] = st.slider("短期快线周期", 2, 30, 5, key='trend_fast')
        params['trend_slow'] = st.slider("短期慢线周期", 5, 60, 20, key='trend_slow')
    else:
        params['trend_fast'] = 5
        params['trend_slow'] = 20

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

    params['use_trend_long'] = st.session_state.use_trend_long
    if params['use_trend_long']:
        st.markdown("**📉 长期趋势强度**")
        params['trend_long_fast'] = st.slider("长期快线周期", 5, 60, 20, key='trend_long_fast')
        params['trend_long_slow'] = st.slider("长期慢线周期", 10, 120, 60, key='trend_long_slow')
    else:
        params['trend_long_fast'] = 20
        params['trend_long_slow'] = 60

with st.sidebar.expander("⚡ 短线指标", expanded=True):
    use_kdj = st.checkbox("KDJ", key='use_kdj'); st.caption("K/D/J三线，判断超买超卖与金叉死叉")
    use_skdj = st.checkbox("SKDJ", key='use_skdj'); st.caption("慢速KDJ，信号更稳定，适合波段")
    use_rsi = st.checkbox("RSI", key='use_rsi'); st.caption("相对强弱，>70超买，<30超卖")
    use_wr = st.checkbox("WR", key='use_wr'); st.caption("威廉指标，<-80超卖，>-20超买")
    use_bias = st.checkbox("BIAS", key='use_bias'); st.caption("乖离率，价格远离均线时可能回归")
    use_cci = st.checkbox("CCI", key='use_cci'); st.caption("突破+100超买，-100超卖")
    use_roc = st.checkbox("ROC", key='use_roc'); st.caption("价格变动速率，衡量趋势强弱")
    use_trend = st.checkbox("短期趋势强度", key='use_trend'); st.caption("短均线差值，正值多头，负值空头")

with st.sidebar.expander("📊 长线指标", expanded=True):
    use_ma = st.checkbox("MA", key='use_ma'); st.caption("多周期均线，判断支撑压力与排列")
    use_macd = st.checkbox("MACD", key='use_macd'); st.caption("趋势动能，金叉死叉与背离")
    use_expma = st.checkbox("EXPMA", key='use_expma'); st.caption("指数均线，近期价格权重更高")
    use_boll = st.checkbox("BOLL", key='use_boll'); st.caption("布林带，判断波动区间与超买超卖")
    use_sar = st.checkbox("SAR", key='use_sar'); st.caption("抛物线转向，提供动态止损位")
    use_dmi = st.checkbox("DMI", key='use_dmi'); st.caption("趋向指标，ADX>25为强趋势")
    use_obv = st.checkbox("OBV", key='use_obv'); st.caption("能量潮，验证价格与成交量配合")
    use_vol = st.checkbox("量比", key='use_vol'); st.caption("当日量与5日均量之比，>1放量")
    use_trend_long = st.checkbox("长期趋势强度", key='use_trend_long'); st.caption("长均线差值，判断中长期趋势")

if not any([st.session_state[k] for k in all_keys]):
    st.error("请在左侧至少选择一个技术指标！")
    st.stop()

if "data" not in st.session_state:
    st.info("👆 请先上传历史数据文件（JSON 或 CSV）。")
    st.stop()
data = st.session_state["data"]

# ========== 指标计算引擎 ==========
def compute_all_features(df, p):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    features = pd.DataFrame(index=df.index)

    if p['use_kdj']:
        n = p['kdj_n']
        low_n = low.rolling(n).min()
        high_n = high.rolling(n).max()
        rsv = ((close - low_n) / (high_n - low_n + 1e-10)) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        features["KDJ_K"] = k
        features["KDJ_D"] = d
        features["KDJ_J"] = j

    if p['use_skdj']:
        n, m = p['skdj_n'], p['skdj_m']
        low_n = low.rolling(n).min()
        high_n = high.rolling(n).max()
        rsv = ((close - low_n) / (high_n - low_n + 1e-10)) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        skdj_k = d.ewm(alpha=1/m, adjust=False).mean()
        skdj_d = skdj_k.ewm(alpha=1/m, adjust=False).mean()
        features["SKDJ_K"] = skdj_k
        features["SKDJ_D"] = skdj_d
        features["SKDJ_KD差"] = skdj_k - skdj_d

    if p['use_rsi']:
        for period in [6, 12, 24]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features[f"RSI{period}"] = rsi

    if p['use_wr']:
        for period in [6, 10]:
            high_n = high.rolling(period).max()
            low_n = low.rolling(period).min()
            wr = (high_n - close) / (high_n - low_n + 1e-10) * -100
            features[f"WR{period}"] = wr

    if p['use_bias']:
        for period in [6, 12, 24]:
            ma = close.rolling(period).mean()
            features[f"BIAS{period}"] = (close - ma) / ma * 100

    if p['use_cci']:
        period = p['cci_period']
        tp = (high + low + close) / 3
        ma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - ma_tp) / (0.015 * mad + 1e-10)
        features["CCI"] = cci

    if p['use_roc']:
        period = p['roc_period']
        roc = close.pct_change(period) * 100
        features[f"ROC{period}"] = roc

    if p['use_trend']:
        fast = p['trend_fast']
        slow = p['trend_slow']
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()
        features["短期趋势强度"] = (ma_fast - ma_slow) / close * 100

    if p['use_trend_long']:
        fast = p['trend_long_fast']
        slow = p['trend_long_slow']
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()
        features["长期趋势强度"] = (ma_fast - ma_slow) / close * 100

    if p['use_ma']:
        for period in [5, 10, 20, 60]:
            ma_val = close.rolling(period).mean()
            features[f"MA{period}"] = ma_val
        for period in [5, 10, 20, 60]:
            ma_val = close.rolling(period).mean()
            features[f"距MA{period}"] = (close - ma_val) / close * 100

    if p['use_macd']:
        fast, slow, sig = p['macd_fast'], p['macd_slow'], p['macd_signal']
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=sig).mean()
        macd_hist = (dif - dea) * 2
        features["MACD_DIF"] = dif
        features["MACD_DEA"] = dea
        features["MACD_柱"] = macd_hist

    if p['use_expma']:
        for period in [12, 50]:
            ema_val = close.ewm(span=period).mean()
            features[f"EXPMA{period}"] = ema_val

    if p['use_boll']:
        period, std_mult = p['bb_period'], p['bb_std']
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        features["BOLL上轨"] = upper
        features["BOLL中轨"] = mid
        features["BOLL下轨"] = lower
        features["BOLL位置"] = (close - lower) / (upper - lower + 1e-10)

    if p['use_sar']:
        af, max_af = 0.02, 0.2
        sar = np.zeros(len(close))
        ep = np.zeros(len(close))
        trend = np.ones(len(close))
        sar[0], ep[0] = close.iloc[0], low.iloc[0]
        for i in range(1, len(close)):
            sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
            if trend[i-1] == 1:
                if low.iloc[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = max(high.iloc[i], high.iloc[i-1])
                    ep[i] = low.iloc[i]
                    af = 0.02
                else:
                    trend[i] = 1
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af = min(af + 0.02, max_af)
                    else:
                        ep[i] = ep[i-1]
            else:
                if high.iloc[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = min(low.iloc[i], low.iloc[i-1])
                    ep[i] = high.iloc[i]
                    af = 0.02
                else:
                    trend[i] = -1
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af = min(af + 0.02, max_af)
                    else:
                        ep[i] = ep[i-1]
        features["SAR"] = sar

    if p['use_dmi']:
        period = p['dmi_period']
        up_move = high.diff()
        down_move = -low.diff()
        pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        mdm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
        pdm_s = pd.Series(pdm).ewm(alpha=1/period, adjust=False).mean()
        mdm_s = pd.Series(mdm).ewm(alpha=1/period, adjust=False).mean()
        pdi = 100 * pdm_s / (atr + 1e-10)
        mdi = 100 * mdm_s / (atr + 1e-10)
        dx = np.abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        features["DMI_PDI"] = pdi
        features["DMI_MDI"] = mdi
        features["DMI_ADX"] = adx

    if p['use_obv']:
        sign = np.sign(close.diff())
        obv = (sign * volume).cumsum()
        features["OBV"] = obv

    if p['use_vol']:
        vol_ma5 = volume.rolling(5).mean()
        vol_ma10 = volume.rolling(10).mean()
        features["成交量"] = volume
        features["量比"] = volume / vol_ma5
        features["量比10"] = volume / vol_ma10

    features = features.ffill().bfill().fillna(0)
    return features

# ========== 分析按钮 ==========
if st.button("🔍 开始分析"):
    if not code:
        st.warning("请输入股票代码")
    else:
        features = compute_all_features(data, params)
        combined = pd.concat([data[["date","close"]], features], axis=1)
        combined = combined.ffill().bfill().fillna(0)

        if len(combined) < 100:
            st.error(f"有效历史数据不足（当前仅 {len(combined)} 天）。")
            st.stop()

        target_date = pd.to_datetime(selected_date)
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
                if zero_var_mask.all():
                    st.error("当前所选指标生成的特征全部为常数，无法进行相似度分析。请至少再添加一个其他指标。")
                    st.stop()
                elif zero_var_mask.any():
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
                cur_df = pd.DataFrame({"指标名称": feature_cols, "当前数值": combined.loc[target_idx, feature_cols].values})
                st.dataframe(cur_df.set_index("指标名称"), use_container_width=True)

            with st.expander("📊 最相似历史日期的技术指标数值（前5个）"):
                top5 = matched_indices[:5]
                sim_df = combined.loc[top5, ["date"]+feature_cols].copy()
                sim_df["日期"] = sim_df["date"].dt.date
                sim_df = sim_df.drop(columns="date").set_index("日期")
                sim_df.index.name = "历史日期"
                st.dataframe(sim_df, use_container_width=True)

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

                fig = px.histogram(
                    ret_arr,
                    nbins=20,
                    title=f"相似历史持有 {days_hold} 天的收益分布",
                    labels={"value": "收益率", "count": "出现次数"},
                    opacity=0.7,
                )
                fig.update_layout(
                    xaxis_title="收益率",
                    yaxis_title="出现次数",
                    bargap=0.05,
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="零收益线")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("相似历史日期及相似度"):
                    match_dates = combined.loc[matched_indices, "date"].reset_index(drop=True)
                    st.dataframe(pd.DataFrame({"历史日期": match_dates.values[:len(sim_scores)], "相似度": sim_scores}).head(20))

                st.warning("⚠️ 风险提示：历史表现不代表未来，本工具仅供参考，不构成投资建议。")
