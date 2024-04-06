import cv2
import numpy as np


def process_image(image):
    image = np.array(image, dtype=np.float32)
    x = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0]  # 只要明度
    y = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 1:]  # 只要颜色信息
    y = y/128.   # 明度归一化处理
    return x, y
