import streamlit as st
import pandas as pd
import json
import io

st.set_page_config(page_title="JSON 数据检查工具", layout="wide")
st.title("🔍 JSON 数据解析检查器")
st.caption("上传东方财富下载的 JSON 文件，查看解析后的数据是否正确。")

uploaded_file = st.file_uploader("📤 上传 JSON 文件", type=["json"])

if uploaded_file is not None:
    try:
        # 读取原始 JSON
        content = uploaded_file.getvalue().decode("utf-8-sig")
        raw_json = json.loads(content)

        st.subheader("📦 原始 JSON 结构")
        st.json(raw_json, expanded=False)

        # 检查是否包含 klines
        if "data" not in raw_json or "klines" not in raw_json["data"]:
            st.error("JSON 文件中没有 data.klines 字段，请确认是东方财富下载的文件。")
            st.stop()

        klines = raw_json["data"]["klines"]
        if not klines:
            st.error("klines 列表为空，文件可能损坏。")
            st.stop()

        st.success(f"✅ 成功读取 {len(klines)} 行原始数据。")

        # 将 klines 转为 CSV 文本
        csv_text = "\n".join(klines)

        # 显示原始文本（前500字符）
        with st.expander("📄 原始数据文本（前500字符）"):
            st.code(csv_text[:500], language="text")

        # 定义列名
        col_names = [
            "date", "open", "close", "high", "low", "volume",
            "amount", "amplitude", "pct_change", "change", "turnover"
        ]

        # 用 pandas 解析
        df = pd.read_csv(io.StringIO(csv_text), header=None, names=col_names)

        st.subheader("📊 解析后的数据表格（前100行）")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("📋 数据基本信息")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("总行数", len(df))
        with col_info2:
            st.metric("总列数", len(df.columns))
        with col_info3:
            st.metric("数据起始日期", str(df["date"].iloc[0]) if len(df) > 0 else "无")
            st.metric("数据结束日期", str(df["date"].iloc[-1]) if len(df) > 0 else "无")

        st.subheader("🔤 各列数据类型")
        dtype_df = pd.DataFrame({
            "列名": df.columns,
            "数据类型": [str(df[col].dtype) for col in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)

        st.subheader("📝 前5行数据样本")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("📝 后5行数据样本")
        st.dataframe(df.tail(), use_container_width=True)

        # 尝试转换数值列
        st.subheader("🔢 数值列统计（转换后）")
        numeric_cols = ["open", "close", "high", "low", "volume",
                        "amount", "amplitude", "pct_change", "change", "turnover"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        # 检查缺失值
        st.subheader("⚠️ 缺失值统计")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "列名": missing.index,
            "缺失数量": missing.values,
            "缺失比例": (missing / len(df) * 100).round(2).astype(str) + "%"
        })
        st.dataframe(missing_df, use_container_width=True)

        # 只保留需要的列
        st.subheader("✅ 最终用于分析的列（date, open, close, high, low, volume）")
        final_df = df[["date", "open", "close", "high", "low", "volume"]].copy()
        st.dataframe(final_df.head(100), use_container_width=True)

        # 下载解析后的 CSV
        csv_download = final_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 下载解析后的 CSV 文件",
            data=csv_download,
            file_name="parsed_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"解析失败：{e}")