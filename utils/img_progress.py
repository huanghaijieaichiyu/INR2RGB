import cv2
import numpy as np


def process_image(image, batch_size=1):
    image = np.array(image, dtype=np.float32)
    x = cv2.cvtColor(1 / 255. * image, cv2.COLOR_RGB2Lab)[:, :, 0]  # 只要明度
    y = cv2.cvtColor(1 / 255. * image, cv2.COLOR_RGB2Lab)[:, :, 1:]  # 只要颜色信息

    # 颜色值归一处理
    lamb = np.max(y)    # 最高权重
    y = y / lamb  # 明度归一化处理

    x = x.reshape(batch_size, image.shape[0], image.shape[1], 1)
    y = y.reshape(batch_size, image.shape[0], image.shape[1], 2)
    return x, y, lamb
