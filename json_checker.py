import streamlit as st
import pandas as pd
import json
import io

st.set_page_config(page_title="JSON 数据检查工具", layout="wide")
st.title("🔍 JSON 数据解析检查器")
st.caption("上传东方财富下载的 JSON 文件，查看解析后的全部数据是否正确。")

uploaded_file = st.file_uploader("📤 上传 JSON 文件", type=["json"])

if uploaded_file is not None:
    try:
        # 读取原始 JSON
        content = uploaded_file.getvalue().decode("utf-8-sig")
        raw_json = json.loads(content)

        st.subheader("📦 原始 JSON 结构（可折叠）")
        with st.expander("点击展开原始 JSON"):
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

        # 定义中文列名
        col_names_cn = [
            "日期", "开盘价", "收盘价", "最高价", "最低价", "成交量",
            "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"
        ]
        # 内部英文列名（方便处理）
        col_names_en = [
            "date", "open", "close", "high", "low", "volume",
            "amount", "amplitude", "pct_change", "change", "turnover"
        ]

        # 用 pandas 解析
        df = pd.read_csv(io.StringIO(csv_text), header=None, names=col_names_en)

        # 显示中文列名的表格
        df_cn = df.copy()
        df_cn.columns = col_names_cn

        # ====== 显示全部数据 ======
        st.subheader(f"📊 解析后的全部数据（共 {len(df_cn)} 行）")
        st.dataframe(df_cn, use_container_width=True, height=600)

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
        dtype_data = {
            "中文名称": col_names_cn,
            "数据类型": [str(df[col].dtype) for col in col_names_en]
        }
        dtype_df = pd.DataFrame(dtype_data)
        st.dataframe(dtype_df, use_container_width=True)

        st.subheader("📝 前5行数据样本")
        st.dataframe(df_cn.head(), use_container_width=True)

        st.subheader("📝 后5行数据样本")
        st.dataframe(df_cn.tail(), use_container_width=True)

        # 尝试转换数值列
        st.subheader("🔢 数值列统计（转换后）")
        numeric_cols_en = ["open", "close", "high", "low", "volume",
                           "amount", "amplitude", "pct_change", "change", "turnover"]
        for col in numeric_cols_en:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 用中文列名显示统计
        stats_df = df[numeric_cols_en].describe()
        stats_df.columns = [col_names_cn[i+1] for i in range(len(numeric_cols_en))]
        st.dataframe(stats_df, use_container_width=True)

        st.caption("""
        **统计说明：**
        - **count**：有效数据数量
        - **mean**：平均值
        - **std**：标准差（波动程度）
        - **min**：最小值
        - **25%**：25%分位数（有四分之一的数据小于此值）
        - **50%**：中位数（一半数据小于此值）
        - **75%**：75%分位数（有四分之三的数据小于此值）
        - **max**：最大值
        """)

        # 检查缺失值
        st.subheader("⚠️ 缺失值统计")
        missing = df.isnull().sum()
        missing_data = {
            "中文名称": col_names_cn,
            "缺失数量": [missing[col] for col in col_names_en],
            "缺失比例": [f"{missing[col] / len(df) * 100:.2f}%" for col in col_names_en]
        }
        missing_df = pd.DataFrame(missing_data)
        st.dataframe(missing_df, use_container_width=True)

        # 只保留需要的列
        st.subheader("✅ 最终用于分析的列（日期、开盘价、收盘价、最高价、最低价、成交量）")
        final_cols_en = ["date", "open", "close", "high", "low", "volume"]
        final_cols_cn = ["日期", "开盘价", "收盘价", "最高价", "最低价", "成交量"]
        final_df = df[final_cols_en].copy()
        final_df.columns = final_cols_cn
        st.dataframe(final_df, use_container_width=True, height=600)

        # 下载解析后的 CSV
        csv_download = final_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 下载解析后的 CSV 文件",
            data=csv_download,
            file_name="解析后的数据.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"解析失败：{e}")
