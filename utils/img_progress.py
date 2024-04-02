import cv2
import numpy as np


def process_image(image):
    image = np.array(image, dtype=np.float32)
    image /= 255.0
    image = cv2.cvtCOLOR(image, cv2.COLOR_RGB2GRAY)
    return image
