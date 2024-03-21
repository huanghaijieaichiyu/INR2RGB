from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def get_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''
    dir_path = Path(dir_path)
    classes = []  # 类别名列表

    for category in dir_path.iterdir():
        if category.is_dir():
            classes.append(category.name)
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表

    for index, name in enumerate(classes):
        class_path = dir_path / name
        if not class_path.is_dir():
            continue
        for img_path in class_path.glob('*.jpg'):
            images_list.append(str(img_path))
            labels_list.append(int(index))
    return images_list, labels_list


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0], std=[0.229, 0.224, 0.225])
])


class MyDataset(Dataset):
    def __init__(self, dir_path, transform=img_transform):
        super(MyDataset, self).__init__()
        self.dir_path = dir_path  # 数据集根目录
        self.transform = transform
        self.images, self.labels = get_images_and_labels(self.dir_path)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        img = cv2.imread(img_path)

        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        """
        cv2.imshow('img', img)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
        """
        img = np.array(img, np.float32)
        img = self.transform(img)
        return img, label
