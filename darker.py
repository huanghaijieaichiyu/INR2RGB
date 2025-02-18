'''
Code by: 黄小海
    这是一个将任意图像转换为低光照图像的程序，旨在制作你自己的低照度增强数据集。
这个程序的数据集保存结构将仿照LOLdataset，即：
LOLdataset
├── eval15
│   ├── high
│   └── low
└── our485
    ├── high
    └── low
    
    事先准备好你的数据集，仿照上述结构，将你的数据集放在LOLdataset目录下，
    并且将high子文件夹中放置你的高光照图像，low子文件夹中保持空白即可。
'''


from operator import index
import cv2
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.data_set import LowLightDataset
from utils.color_trans import PSlab2rgb, PSrgb2lab, RGB_HSV
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def get_dataloader(data_dir, batch_size, img_size, num_workers, phase="train"):
    """
    获取数据加载器。
    Args:
        data_dir (str): 数据集目录。
        batch_size (int): 批大小。
        img_size (tuple): 图像大小。
        num_workers (int): 数据加载工作线程数。
        phase (str): 阶段，可选值为"train"或"test"。

    Returns:
        DataLoader: 数据加载器。
    """
    # 定义图像转换
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 先转换为PIL Image, 因为一些transform需要PIL Image作为输入
        transforms.Resize((375, 1242)),  # 调整大小
        transforms.ToTensor(),  # 转换为Tensor
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 可选：归一化
    ])

    # 创建数据集实例
    dataset = LowLightDataset(
        image_dir=data_dir, transform=transform, phase=phase)

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader


def darker(dataloder, batch_size=8, ratio=0.05, phase="train", data_dir="../datasets/LOLdataset"):
    """
    低光照增强。
    Args:
        dataloder (DataLoader): 数据加载器。

    Returns:
        DataLoader: 增强后的数据加载器。
    """
    # 获取数据集中的图片后缀名
    sample_image_path = next(os.path.join(root, file) for root, _, files in os.walk(
        data_dir) for file in files if file.endswith(('png', 'jpg', 'jpeg', 'bmp')))
    img_format = sample_image_path.split('.')[-1]
    # 获取图片名称
    img_name = sample_image_path.split('/')[-1].split('.')[0]
    print("开始处理数据集...")
    log = SummaryWriter(log_dir='runs/darker', comment='darker')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pbar = tqdm(enumerate(dataloder), total=len(dataloder))
    index = 0
    hsvtrans = RGB_HSV()
    with torch.no_grad():
        for _, (_, high_img) in pbar:
            # 低光照增强
            high_img = high_img.to(device)
            log.add_images('origin_high_img', high_img)
            # 转换为HSV ， 低光照增强只对V通道进行操作 v取值范围为0-1
            high_img = hsvtrans.rgb_to_hsv(high_img)
            high_img[:, 2, :, :] = high_img[:, 2, :, :] * ratio
            high_img = hsvtrans.hsv_to_rgb(high_img)
            log.add_images('high_img', high_img)

            dark_img = high_img.detach().squeeze(
                0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            # 可以修改为你需要的格式，例如 'jpg', 'jpeg', 'bmp' 等
            path = os.path.join(
                data_dir, f"our485/low/{index}.{img_format}") if phase == "train" else os.path.join(data_dir, f"eval15/low/{index}.{img_format}")
            cv2.imwrite(path, cv2.cvtColor(dark_img * 255., cv2.COLOR_RGB2BGR))
            index += 1


if __name__ == '__main__':
    # 1. 设置数据集目录
    data_dir = "../datasets/kitti_LOL"  # 替换成你的数据集路径

    # 2. 创建 DataLoader
    # batch_size 必须为 1
    batch_size = 1
    assert batch_size == 1, "batch_size must be 1."
    ratio = 0.05
    img_size = (256, 256)
    num_workers = 0
    train_dataloder = get_dataloader(
        data_dir, batch_size, img_size, num_workers, phase="train")
    test_dataloder = get_dataloader(
        data_dir, batch_size, img_size, num_workers, phase="test")
    # 3. 低光照增强
    darker(train_dataloder, ratio=ratio, batch_size=batch_size,
           phase="train", data_dir=data_dir)
    darker(test_dataloder, ratio=ratio, batch_size=batch_size,
           phase="test", data_dir=data_dir)

    print("处理完成！请手动检查数据集。")
    # ...
