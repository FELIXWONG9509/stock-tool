st.sidebar.header("预设组合")
preset = st.sidebar.selectbox("选择一个预设组合，快速勾选常用指标", ["自定义", "短线波段组合", "长线趋势组合", "全能组合"])

# 根据预设定义默认勾选集合
# 如果预设改变，更新session_state中的各个指标check值
# 初始化session_state
if 'preset' not in st.session_state:
    st.session_state.preset = "自定义"
    # 设置默认指标勾选
    for key in all_indicator_keys:
        st.session_state[key] = False

if preset != st.session_state.preset:
    st.session_state.preset = preset
    # 根据preset设置所有指标勾选状态
    if preset == "短线波段组合":
        # 勾选短线指标，不勾选长线
        for key in short_keys:
            st.session_state[key] = True
        for key in long_keys:
            st.session_state[key] = False
    elif preset == "长线趋势组合":
        for key in long_keys:
            st.session_state[key] = True
        for key in short_keys:
            st.session_state[key] = False
    elif preset == "全能组合":
        for key in all_keys:
            st.session_state[key] = True
    else: # 自定义，保持现状
        pass

# 然后短线指标区域expander
with st.sidebar.expander("短线指标区域", expanded=True):
    use_kdj = st.checkbox("KDJ (随机指标)", value=st.session_state.get('use_kdj', False), key='use_kdj')
    ...
# 长线指标区域expander
with st.sidebar.expander("长线指标区域", expanded=True):
    use_ma = st.checkbox("MA (均线排列)", ...)
    ...
