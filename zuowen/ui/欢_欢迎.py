import os
import streamlit as st

with open(os.path.join(os.path.dirname(__file__), "markdown", "welcome.md"), encoding="utf-8") as f:
    content = f.read()

st.sidebar.info("从边栏中选择一个工具")
st.markdown(content, unsafe_allow_html=True)
