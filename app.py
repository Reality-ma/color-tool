import cv2
import numpy as np
import streamlit as st
import io, zipfile
import matplotlib.pyplot as plt

st.title("多区域颜色识别与分区面积比例计算 (HSV)")

# 上传图像
uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "png", "jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="原始图像", use_container_width=True)

    # HSV 转换
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    st.subheader("颜色区域设置")
    num_ranges = st.number_input("选择颜色区间数量", min_value=1, max_value=5, value=1, step=1)

    total_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    stats = []  # 存储每个区间的比例

    for i in range(num_ranges):
        st.markdown(f"### 区间 {i+1}")
        h_min = st.slider(f"H{i+1} 最小值", 0, 179, 0, key=f"hmin_{i}")
        h_max = st.slider(f"H{i+1} 最大值", 0, 179, 179, key=f"hmax_{i}")
        s_min = st.slider(f"S{i+1} 最小值", 0, 255, 50, key=f"smin_{i}")
        s_max = st.slider(f"S{i+1} 最大值", 0, 255, 255, key=f"smax_{i}")
        v_min = st.slider(f"V{i+1} 最小值", 0, 255, 50, key=f"vmin_{i}")
        v_max = st.slider(f"V{i+1} 最大值", 0, 255, 255, key=f"vmax_{i}")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, lower, upper)
        total_mask = cv2.bitwise_or(total_mask, mask)

        # 单个区间比例
        total_pixels = image.shape[0] * image.shape[1]
        selected_pixels = np.sum(mask > 0)
        ratio = (selected_pixels / total_pixels) * 100
        stats.append((f"区间 {i+1}", ratio, lower, upper))

    # 合并结果
    result = cv2.bitwise_and(image, image, mask=total_mask)

    # 总比例
    selected_pixels_total = np.sum(total_mask > 0)
    ratio_total = (selected_pixels_total / (image.shape[0]*image.shape[1])) * 100
    ratio_str = f"{ratio_total:.2f}"

    col1, col2 = st.columns(2)
    with col1:
        st.image(total_mask, caption="合并掩膜 (白色=选中)", use_container_width=True)
    with col2:
        st.image(result, caption="合并结果", use_container_width=True)

    st.subheader("统计结果")
    st.success(f"选中区域总占比: {ratio_str}%")
    for name, ratio, l, u in stats:
        st.write(f"{name}: 占比 {ratio:.2f}% (H={l[0]}-{u[0]}, S={l[1]}-{u[1]}, V={l[2]}-{u[2]})")

    # 绘制饼图
    st.subheader("分区比例饼图")
    labels = [name for name, ratio, _, _ in stats if ratio > 0]
    values = [ratio for _, ratio, _, _ in stats if ratio > 0]
    chart_img = None
    if values:
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        # 保存饼图到内存
        chart_buffer = io.BytesIO()
        fig.savefig(chart_buffer, format="png")
        chart_img = chart_buffer.getvalue()
        plt.close(fig)

    # 批量导出 ZIP
    st.subheader("批量导出")
    if st.button("打包下载 ZIP"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipf:
            # 原图
            _, img_bytes = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            zipf.writestr("original.png", img_bytes.tobytes())

            # 掩膜
            mask_bgr = cv2.cvtColor(total_mask, cv2.COLOR_GRAY2BGR)
            _, mask_png = cv2.imencode(".png", mask_bgr)
            zipf.writestr(f"mask_total_{ratio_str}%.png", mask_png.tobytes())

            # 结果图
            _, result_png = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            zipf.writestr(f"result_total_{ratio_str}%.png", result_png.tobytes())

            # 信息文件
            info_text = f"总选中区域占比: {ratio_str}%\n\n"
            for name, ratio, l, u in stats:
                info_text += f"{name}: 占比 {ratio:.2f}% (H={l[0]}-{u[0]}, S={l[1]}-{u[1]}, V={l[2]}-{u[2]})\n"
            zipf.writestr("info.txt", info_text)

            # 饼图
            if chart_img:
                zipf.writestr("chart.png", chart_img)

        st.download_button(
            label="下载 ZIP 包",
            data=buffer.getvalue(),
            file_name=f"color_selection_{ratio_str}%.zip",
            mime="application/zip"
        )
