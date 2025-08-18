import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print("拍照失败")
        return None

def process_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 设置颜色范围（这里示例红色）
    lower = np.array([0, 50, 50])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"Area: {area:.0f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

def save_image(image):
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG files","*.jpg"),("PNG files","*.png")])
    if save_path:
        cv2.imwrite(save_path, image)
        print(f"图像已保存到: {save_path}")

if __name__ == "__main__":
    img = capture_image()
    if img is not None:
        processed_img = process_image(img)
        cv2.imshow("Result", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(processed_img)
