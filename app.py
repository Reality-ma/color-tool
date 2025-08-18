import cv2
import numpy as np
import streamlit as st

st.title("颜色轮廓检测与面积测量（只保留所选颜色）")

# 上传图片
uploaded_file = st.file_uploader("上传图片（JPG/PNG）", type=["jpg","png"])

if uploaded_file is not None:
    # 转换为 OpenCV 图像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)

    st.sidebar.subheader("HSV 颜色范围设置")
    h_min = st.sidebar.slider("H 最小值", 0, 179, 0)
    h_max = st.sidebar.slider("H 最大值", 0, 179, 10)
    s_min = st.sidebar.slider("S 最小值", 0, 255, 50)
    s_max = st.sidebar.slider("S 最大值", 0, 255, 255)
    v_min = st.sidebar.slider("V 最小值", 0, 255, 50)
    v_max = st.sidebar.slider("V 最大值", 0, 255, 255)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # 颜色检测
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # 保留所选颜色区域
    result_img = cv2.bitwise_and(img, img, mask=mask)

    # 轮廓检测
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    st.subheader("轮廓检测结果（只保留所选颜色）")

    total_area = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 100:  # 忽略太小轮廓
            total_area += area
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            st.write(f"轮廓 {i+1} 面积: {area:.0f} px²")

    st.write(f"总保留面积: {total_area:.0f} px²")

    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="只保留所选颜色", use_container_width=True)



