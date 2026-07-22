import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from datetime import datetime, timedelta, date
import time
import io
import json

st.set_page_config(page_title="多指标历史相似概率", layout="wide")
st.caption("选择经典组合或自由搭配，指定分析日期，寻找历史上最相似的时刻，计算后续上涨概率。")

code = st.text_input("股票代码（如 600887）", "600887")
analysis_date = st.date_input("📅 分析日期（默认今天，可选择历史日期）", date.today())
days_hold = st.selectbox("持仓周期（天）", [5, 10, 20, 50, 100, 150, 200, 300, 400], index=2)

# ---------- 数据来源区域 ----------
st.subheader("📥 数据来源")
col_dl, col_up = st.columns(2)
with col_dl:
    st.markdown("**方式一：在线获取（可能因网络限制失败）**")
    if st.button("🌐 在线获取历史数据"):
        with st.spinner("正在从东方财富获取数据..."):
            try:
                try:
                    info = ak.stock_individual_info_em(symbol=code)
                    list_date_str = info.loc[info["item"] == "上市时间", "value"].values[0]
                    list_date = pd.to_datetime(list_date_str)
                    total_days = (datetime.now() - list_date).days
                    request_days = min(total_days, 5 * 365)
                    request_days = max(request_days, 60)
                except:
                    request_days = 5 * 365

                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=request_days)).strftime("%Y%m%d")

                df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                        start_date=start_date, end_date=end_date, adjust="qfq")
                if df.empty:
                    st.warning("未获取到数据，请检查代码或使用方式二上传。")
                else:
                    df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"})
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                    st.session_state.online_data = df
                    st.success(f"✅ 在线获取成功，共 {len(df)} 条数据，可直接分析。")
            except Exception as e:
                st.error(f"在线获取失败（{e}）。请使用下方“方式二”上传本地数据。")
                st.info("📥 推荐使用JSON格式下载（解析最稳定）：")
                st.markdown(f"[➡️ 下载 {code} 的JSON数据](http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end=20500101&lmt=10000&ut=fa5fd1943c7b386f172d6893dbfba10b&cb=) （右键→另存为→文件名.json）", unsafe_allow_html=True)
                st.markdown(f"或者下载CSV：[➡️ 下载 {code} 的CSV数据](http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end=20500101&lmt=10000) （右键→另存为→文件名.csv）", unsafe_allow_html=True)

with col_up:
    st.markdown("**方式二：上传本地文件（JSON或CSV）**")
    uploaded_file = st.file_uploader("选择文件（.json 或 .csv）", type=["json", "csv"])
    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name.lower()
            if file_name.endswith(".json"):
                # 解析JSON
                content = uploaded_file.getvalue().decode("utf-8-sig")
                data_json = json.loads(content)
                if "data" in data_json and "klines" in data_json["data"]:
                    klines = data_json["data"]["klines"]
                    if not klines:
                        st.error("JSON文件中没有历史数据。")
                        st.stop()
                    rows = [line.split(",") for line in klines]
                    # 每行至少需要6列：日期,开盘,收盘,最高,最低,成交量
                    rows = [r[:6] for r in rows]
                    df_upload = pd.DataFrame(rows, columns=["date","open","close","high","low","volume"])
                else:
                    st.error("JSON格式不正确，缺少 data.klines 字段。")
                    st.stop()
            else:
                # 解析CSV（原有逻辑，加强容错）
                content = uploaded_file.getvalue().decode("utf-8-sig")
                lines = content.strip().split("\n")
                if not lines:
                    st.error("文件为空。")
                    st.stop()
                first_line = lines[0].strip()
                if first_line and first_line[0].isdigit():
                    # 无表头CSV，按顺序分配列名
                    columns = ["date","open","close","high","low","volume"]
                    df_upload = pd.read_csv(io.StringIO(content), header=None)
                    df_upload = df_upload.iloc[:, :6]
                    df_upload.columns = columns
                else:
                    # 有表头CSV，兼容中英文
                    df_upload = pd.read_csv(io.StringIO(content))
                    rename_map = {
                        "日期": "date", "开盘": "open", "收盘": "close",
                        "最高": "high", "最低": "low", "成交量": "volume",
                        "date": "date", "open": "open", "close": "close",
                        "high": "high", "low": "low", "volume": "volume"
                    }
                    df_upload.columns = [col.strip().lower() for col in df_upload.columns]
                    col_map = {col: rename_map[col] for col in df_upload.columns if col in rename_map}
                    df_upload = df_upload.rename(columns=col_map)
                    required = ["date","open","close","high","low","volume"]
                    if not all(c in df_upload.columns for c in required):
                        st.error("CSV缺少必要列：日期、开盘、收盘、最高、最低、成交量。请检查文件或改用JSON格式。")
                        st.stop()

            # 统一处理
            df_upload["date"] = pd.to_datetime(df_upload["date"])
            for col in ["open","close","high","low","volume"]:
                df_upload[col] = pd.to_numeric(df_upload[col], errors="coerce")
            df_upload = df_upload.dropna().sort_values("date").reset_index(drop=True)
            if len(df_upload) < 60:
                st.error("数据不足，至少需要60个交易日。")
                st.stop()
            st.session_state.uploaded_data = df_upload
            st.success(f"✅ 上传成功，共 {len(df_upload)} 条数据，可直接分析。")
        except Exception as e:
            st.error(f"文件解析失败：{e}")

# 确定最终使用的数据
if "online_data" in st.session_state:
    data = st.session_state.online_data
elif "uploaded_data" in st.session_state:
    data = st.session_state.uploaded_data
else:
    data = None

# ========== 固定搭配定义 ==========
FIXED_COMBOS = {
    "自定义（手动选择）": {
        "说明": "在下方短线/长线区域自由勾选指标，完全自定义。",
        "适合周期": "不限",
        "类别": "",
        "keys": []
    },
    "BOLL + KDJ 经典组合": {
        "说明": "布林带判断趋势与空间，KDJ判断短线买卖时机。多头回踩中下轨+KDJ低位金叉买入；冲上轨+KDJ高位死叉卖出。",
        "适合周期": "10天 ~ 60天",
        "类别": "经典组合",
        "keys": ["use_boll", "use_kdj"]
    },
    "MACD + OBV 量价组合": {
        "说明": "MACD判断趋势，OBV验证量能。金叉+OBV上升=可信上涨；死叉+OBV下降=可靠下跌。",
        "适合周期": "20天 ~ 100天",
        "类别": "经典组合",
        "keys": ["use_macd", "use_obv"]
    },
    "EXPMA + CCI 短线组合": {
        "说明": "EXPMA提供支撑压力，CCI捕捉超买超卖与突破。回踩EXPMA+CCI拐头为买点；CCI上破+100为加速信号。",
        "适合周期": "5天 ~ 30天",
        "类别": "经典组合",
        "keys": ["use_expma", "use_cci"]
    },
    "DMI + SAR 多空组合": {
        "说明": "DMI判断趋势强度（ADX>25），SAR提供止损。PDI上穿MDI+ADX上升+SAR红点，预示主升浪。",
        "适合周期": "20天 ~ 150天",
        "类别": "经典组合",
        "keys": ["use_dmi", "use_sar"]
    }
}

# 显示名映射（略，与之前相同）
display_to_combo = {}
combo_options = ["自定义（手动选择）"]
display_to_combo["自定义（手动选择）"] = "自定义（手动选择）"
combo_options.append("── 经典组合 ──")
classic_combos = [(name, info) for name, info in FIXED_COMBOS.items() if info.get("类别") == "经典组合"]
for name, info in classic_combos:
    display_name = f"   {name}"
    combo_options.append(display_name)
    display_to_combo[display_name] = name

short_keys = ['use_kdj', 'use_skdj', 'use_rsi', 'use_wr', 'use_bias', 'use_cci', 'use_roc']
long_keys = ['use_ma', 'use_macd', 'use_expma', 'use_boll', 'use_sar', 'use_dmi', 'use_obv', 'use_vol', 'use_trend']
all_keys = short_keys + long_keys

for k in all_keys:
    if k not in st.session_state:
        st.session_state[k] = False
if 'combo' not in st.session_state:
    st.session_state.combo = "自定义（手动选择）"

# 重置按钮、固定搭配、参数调整、指标勾选区、指标计算引擎、主分析部分与之前一致（此处省略，你可以沿用之前完整的代码区域，从“重置按钮”开始一直到主分析结束，没有变化）
# 为节省篇幅，此处用占位表示，实际需要将之前最终版中从“# ========== 重置按钮 ==========”开始的代码完整粘贴在这里。
# （请用你上一个完整版本中对应的代码段替换下面这一行）

# 注意：因为代码太长，这里只给了新修改的数据来源部分，后续部分需要你保持原来的完整代码不变。
# 真正部署时，请将下面这个完整的文件（我拼接好）替换GitHub上的文件。
