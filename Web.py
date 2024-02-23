import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# # 加载机器学习模型
# model = GaussianNB()
# # 机器学习模型训练
# data = np.load('E:\python\day02\MSIST.\mnist.npz',allow_pickle=True)
# x_train, y_train = data['x_train'], data['y_train']
# x_test, y_test = data['x_test'], data['y_test']
# model.fit(X = x_train, y = y_train)

# ChatGLM3大模型加载
# from transformers import AutoTokenizer
# from transformers import AutoModel

# 全局设置
st.set_page_config(page_title = "乳腺疾病分类模型" , 
                   page_icon = "👩‍⚕️",
                   layout = "wide",
                   initial_sidebar_state = "expanded")
# icon网址https://getemoji.com/

# ChatGLM3大模型加载
# MODEL_PATH = "E:\python\day02"
# @st.cache_resource
# def load_model(MODEL_PATH):
#     tokennizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = MODEL_PATH,
#                                                trust_remote_code = True)
#     model = AutoModel.from_pretrained(pretrained_model_name_or_path = MODEL_PATH,
#                                       trust_remote_code = True,
#                                       device_map = True).eval()
#     return tokennizer,model
# tokennizer,model = load_model(MODEL_PATH = MODEL_PATH)

# 侧边栏布局
with st.sidebar:
    st.markdown(body = "## 乳腺疾病分类模型")
    st.divider()
    st.write("**模型信息**：基于机器学习构建乳腺疾病分类模型，用于乳腺医学图片的自动分类")
    # st.divider()
    st.write("**乳腺疾病分类**：")
    st.write("(1)乳腺炎症：包括哺乳期的乳腺炎、非哺乳期的乳腺炎，非哺乳期的乳腺炎又包括肉芽肿性小叶性乳腺炎、浆细胞性乳腺炎等。")
    st.write("(2)乳腺肿瘤：包括良性肿瘤、恶性肿瘤，其中良性肿瘤中包括乳腺纤维瘤、乳腺增生性结节、乳房囊肿等，恶性肿瘤包括乳腺癌以及乳房肉瘤。")
    st.write("(3)乳腺增生：既不属于炎症，也不属于肿瘤，是乳腺组织增生以及乳腺组织修复不全的一种病症。")
    st.divider()
    st.write("作者AI4904")
    st.write(time.ctime())

# 主显示区布局
st.title('👩‍⚕️ 乳腺疾病分类 👩‍⚕️')
st.divider()

# 列式布局
col1, col2, col3 = st.columns(3)
with col1:
    st.write("**图片一**")
    # 上传图片
    image1 = st.file_uploader("上传图片1", type = ["jpg", "png"])
    if image1:
    # 读取图片
        img = st.image(image1, use_column_width = True)
    # 保存图片
        with open("img.jpg", "wb") as f:
            f.write(image1.read())
    st.divider()
    # # 输入图片1进行预测
    # out1 = model.predict(image1)
    # 输出图片1的分类结果
    text1 = st.write("**图片一结果为：**")
    # text1 = st.write("**图片一结果为：**", out1)
    st.divider()

with col2:
    st.write("**图片二**")
    # 上传图片
    image2 = st.file_uploader("上传图片2", type = ["jpg", "png"])
    if image2:
    # 读取图片
        img = st.image(image2, use_column_width = True)
    # 保存图片
        with open("img.jpg", "wb") as f:
            f.write(image2.read())
    st.divider()
    # # 输入图片2进行预测
    # out2 = model.predict(image2)
    # 输出图片1的分类结果
    text2 = st.write("**图片二结果为：**")
    # text2 = st.write("**图片二结果为：**", out2)
    st.divider()

with col3:
    st.write("**图片三**")
    # 上传图片
    image3 = st.file_uploader("上传图片3", type = ["jpg", "png"])
    if image3:
    # 读取图片
        img = st.image(image3, use_column_width = True)
    # 保存图片
        with open("img.jpg", "wb") as f:
            f.write(image3.read())
    st.divider()
    # # 输入图片3进行预测
    # out3 = model.predict(image3)
    # 输出图片3的分类结果
    text3 = st.write("**图片三结果为：**")
    # text3 = st.write("**图片三结果为：**", out3)
    st.divider()

# 聊天输入框(默认在最下面)
query = st.chat_input(placeholder = "请输入您的问题......")
# 如果有输入内容
if query:
    # 用户提问渲染
    with st.chat_message(name = "user"):
        st.markdown(query)

     # 预设回答渲染
    with st.chat_message(name = "assistant"):
        out = st.empty()
        txt = ""
        text = "建立良好的生活方式，调整好生活节奏，保持心情舒畅。坚持体育锻炼，积极参加社交活动，避免和减少精神、心理紧张因素"
        for i in text:
            txt += str(i)
            out.markdown(txt)
            time.sleep(0.1)

    # ChatGLM3大模型回答渲染
    # with st.chat_message(name = "assistant"):
    #     out = st.empty()
    #     for response,history in model.stream_chat(tokenizer=tokenizer.
    #                                               model=model,
    #                                               query=query):
    #     out.markdown(response)
    

