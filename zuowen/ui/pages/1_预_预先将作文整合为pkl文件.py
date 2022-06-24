import os
import pickle

import streamlit as st
from transformers import CpmTokenizer

from zuowen.utils import absolute_path

st.title("作文整合器")
st.subheader("吴凡的作文整合器")
datas = st.file_uploader(
    "选择所有需要训练的作文文件",
    accept_multiple_files=True,
    type=["txt"],
    help="所有需要训练的作文文件，是txt文件的格式")
win_size = st.slider(
    "滑动窗口的大小",
    min_value=100,
    max_value=1000,
    step=50,
    help="滑动窗口的大小，相当于每条数据的最大长度")
step = st.slider(
    "滑动窗口的滑动步幅",
    min_value=100,
    max_value=1000,
    step=50,
    help="滑动窗口的滑动步幅")

tokenizer = CpmTokenizer(
    vocab_file=os.path.join(
        absolute_path,
        "vocab",
        "chinese_vocab.model"))
eod_id = tokenizer.convert_tokens_to_ids("<eod>")
sep_id = tokenizer.sep_token_id

start = st.button("开始训练", help="当按下这个按钮后，电脑就会将你上传的所有作文转换为pickle格式的训练集")
train_list = []
if start:
    st.info("开始标记数据")
    progress_bar = st.progress(0)
    each_file_value = 100 / len(datas)
    for file in datas:
        lines = file.readlines()
        title = lines[1][3:].strip()
        lines = lines[7:]
        article = ""
        for line in lines:
            if line.strip() != "":
                article += line.decode("utf-8")
        title_ids = tokenizer.encode(
            title.decode("utf-8"),
            add_special_tokens=False)
        article_ids = tokenizer.encode(article, add_special_tokens=False)
        token_ids = title_ids + [sep_id] + article_ids + [eod_id]

        win_size = win_size
        start_index = 0
        end_index = win_size
        data = token_ids[start_index:end_index]
        train_list.append(data)
        start_index += step
        end_index += step
        while end_index + 50 < len(token_ids):
            data = token_ids[start_index:end_index]
            train_list.append(data)
            start_index += step
            end_index += step
        progress_bar.progress(datas.index(file) * each_file_value)

    data = pickle.dumps(train_list)
    st.download_button(
        "下载整合完毕的文件",
        data,
        mime="application/octet-stream",
        file_name="train.pkl",
        help="下载整合完毕的文件，即pkl格式的数据集")
    st.balloons()
    st.success("祝贺，您已成功转换数据集！下一步，您可以转到训练页面，转换为可用的数据集")
