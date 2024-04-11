import cv2
import numpy as np


def process_image(image):
    image_size = image.size
    image = np.array(image, dtype=np.float32)
    x = cv2.cvtColor((1 / 255. * image),
                     cv2.COLOR_RGB2LAB)[:, :, 0]  # 只要第一列：灰度
    y = cv2.cvtColor((1 / 255. * image),
                     cv2.COLOR_RGB2LAB)[:, :, 1:]  # 不要第一列 ：AB

    y /= 128
    print('y: ', y)
    x = x.reshape(1, image_size[0], image_size[1], 1)
    y = y.reshape(1, image_size[0], image_size[1], 2)
    return x, y, image_size
