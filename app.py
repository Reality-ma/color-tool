import cv2
import numpy as np
import streamlit as st

st.title("颜色轮廓检测与面积测量")

# 上传图片
uploaded_file = st.file_uploader("上传图片（JPG/PNG）", type=["jpg","png"])

if uploaded_file is not None:
    # 将上传文件转换为 OpenCV 图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="原图", use_column_width=True)

    # 颜色检测（示例红色）
    st.subheader("轮廓检测结果")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 50])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img.copy()
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 100:  # 忽略太小的轮廓
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            st.write(f"轮廓 {i+1} 面积: {area:.0f} px²")

    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="检测结果", use_column_width=True)

