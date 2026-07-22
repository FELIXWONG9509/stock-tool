import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from datetime import date
import io
import json
import re

st.set_page_config(page_title="多指标历史相似概率", layout="wide")
st.caption("请先上传JSON数据，然后选择指标组合进行分析。")

code = st.text_input("股票代码（如 600887）", "600887")
secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
download_url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end=20500101&lmt=10000"
st.link_button("🌐 打开数据下载页面（右键另存为 .json）", download_url)

uploaded_file = st.file_uploader("📤 上传 JSON 文件", type=["json"])

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        data_json = json.loads(content)
        klines = data_json["data"]["klines"]
        csv_text = "\n".join(klines)
        col_names = ["date","open","close","high","low","volume",
                     "amount","amplitude","pct_change","change","turnover"]
        df = pd.read_csv(io.StringIO(csv_text), header=None, names=col_names)
        df = df[["date","open","close","high","low","volume"]]
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open","close","high","low","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        st.session_state["data"] = df
        st.success(f"上传成功，{len(df)} 条数据")
    except Exception as e:
        st.error(f"解析失败：{e}")

if "data" not in st.session_state:
    st.info("请上传文件")
    st.stop()
data = st.session_state["data"]

analysis_date = st.date_input("分析日期", date.today())
days_hold = st.selectbox("持仓周期", [5,10,20,50,100,150,200,300,400], index=2)

# ====== 侧边栏（必定显示） ======
st.sidebar.header("🔧 技术指标")

use_kdj = st.sidebar.checkbox("KDJ", value=True)
use_rsi = st.sidebar.checkbox("RSI", value=False)
use_macd = st.sidebar.checkbox("MACD", value=False)
use_ma = st.sidebar.checkbox("MA", value=False)

params = {}
params['kdj_n'] = st.sidebar.slider("KDJ 周期", 5, 30, 9) if use_kdj else 9
params['rsi_n'] = st.sidebar.slider("RSI 周期", 5, 30, 14) if use_rsi else 14
if use_macd:
    params['macd_fast'] = st.sidebar.slider("MACD 快线", 5, 30, 12)
    params['macd_slow'] = st.sidebar.slider("MACD 慢线", 10, 40, 26)
    params['macd_sig'] = st.sidebar.slider("MACD 信号线", 5, 15, 9)
if use_ma:
    params['ma_fast'] = st.sidebar.slider("MA 快线", 2, 30, 5)
    params['ma_slow'] = st.sidebar.slider("MA 慢线", 5, 120, 20)

# 特征计算
close = data["close"]
high = data["high"]
low = data["low"]
vol = data["volume"]
feat = pd.DataFrame(index=data.index)

if use_kdj:
    n = params['kdj_n']
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = (close - low_n) / (high_n - low_n + 1e-10) * 100
    k = rsv.ewm(alpha=1/3).mean()
    d = k.ewm(alpha=1/3).mean()
    feat["kdj_k"] = k/100
    feat["kdj_d"] = d/100
if use_rsi:
    n = params['rsi_n']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    feat["rsi"] = rsi/100
if use_macd:
    f = params['macd_fast']
    s = params['macd_slow']
    sig = params['macd_sig']
    ema_f = close.ewm(span=f).mean()
    ema_s = close.ewm(span=s).mean()
    macd_line = ema_f - ema_s
    signal = macd_line.ewm(span=sig).mean()
    feat["macd_hist"] = (macd_line - signal) / close
if use_ma:
    f = params['ma_fast']
    s = params['ma_slow']
    ma_f = close.rolling(f).mean()
    ma_s = close.rolling(s).mean()
    feat["ma_cross"] = (ma_f - ma_s) / close

feat = feat.ffill().bfill().fillna(0)
combined = pd.concat([data[["date","close"]], feat], axis=1)

if st.button("开始分析"):
    target = pd.to_datetime(analysis_date)
    row = combined[combined["date"] == target]
    if row.empty:
        st.error("无此日期")
    else:
        idx = row.index[0]
        st.success(f"收盘价：{row['close'].values[0]:.2f}")
        st.write("特征列：", list(feat.columns))
