import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import tensorboard
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.base_mode import ConvertV1
from train import process_image
from utils.model_map import model_structure
from utils.save_path import Path


def parse_args():
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str, help="path to dataset", required=True)
    parser.add_argument("--model", type=str, help="path to model", required=True)
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=int, default=640, help="size of the image")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of data loading workers, if in windows, must be 0"
                        )
    parser.add_argument("--device", type=str, default="cuda", choices='["cpu", "cuda"]',
                        help="select your device to train, if you have a gpu, use 'cuda:0'!")  # 训练设备
    parser.add_argument("--save_path", type=str, default='runs/', help="where to save your data")  # 保存位置
    parser.add_argument("--sample_interval", type=int, default=10, help="how often to sample the img")
    opt = parser.parse_args()
    print(opt)
    return opt


def predict(self):
    # 防止同名覆盖
    path = Path(self.save_path, model='predict')
    # 数据准备
    if self.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    log = tensorboard.SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard'),
                                    filename_suffix=str('val'),
                                    flush_secs=180)
    model = ConvertV1()
    model_structure(model)
    checkpoint = torch.load(self.model)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    test_data = ImageFolder(root=self.data)
    pbar = tqdm(total=len(test_data))
    model.eval()
    torch.no_grad()
    i = 0
    if not os.path.exists(os.path.join(path, 'predictions')):
        os.makedirs(os.path.join(path, 'predictions'))
    for img, _ in test_data:
        img = img.resize((640, 640))  # 是否需要resize取决于新图片格式与训练时的是否一致

        x, _, image_size = process_image(img)
        x_trans = torch.tensor(x, dtype=torch.float).to(device)
        # 训练前交换维度
        x_trans = torch.permute(x_trans, (0, 3, 1, 2))
        outputs = model(x_trans)

        outputs = outputs.cpu().data.numpy()
        outputs = outputs.astype(np.float32)
        tmp = np.zeros((640, 640, 3), dtype=np.float32)
        # 训练后复原再拼接
        tmp[:, :, 0] = 128 * x[0][:, :, 0]
        tmp[:, :, 1:] = 128 * outputs[0]
        print('tmp is ', tmp)
        if i > 10 and i % 10 == 0:  # 图片太多，十轮保存一次
            img_save_path = os.path.join(path, 'predictions', str(i) + '.jpg')
            cv2.imwrite(img_save_path, cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB))
        i = i + 1
        pbar.set_description('Processed %d images' % i)
        pbar.update(1)
    pbar.close()


def predict_live(self):
    if self.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    log = tensorboard.SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard'),
                                    filename_suffix=str('val'),
                                    flush_secs=180)
    model = ConvertV1()
    model_structure(model)
    checkpoint = torch.load(self.model)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    cap = cv2.VideoCapture(2)  # 读取图像
    model.eval()
    torch.no_grad()
    if not os.path.exists(os.path.join(self.save_path, 'predictions')):
        os.makedirs(os.path.join(self.save_path, 'predictions'))
    while cap.isOpened():
        rect, frame = cap.read()
        frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = cv2.resize(frame_pil, (640, 640))  # 是否需要resize取决于新图片格式与训练时的是否一致
        frame_pil = Image.fromarray(frame_pil)
        x, _, image_size = process_image(frame_pil)

        x_trans = torch.tensor(x, dtype=torch.float).to(device)
        # 训练前交换维度
        x_trans = torch.permute(x_trans, (0, 3, 1, 2))
        outputs = model(x_trans)
        outputs = outputs.cpu().data.numpy()
        outputs = outputs.astype(np.float32)
        tmp = np.zeros((640, 640, 3), dtype=np.float32)
        print('x[0] is ', x[0])
        print('outputs[0] is ', outputs[0])
        tmp[:, :, 0] = x[0][:, :, 0]
        tmp[:, :, 1:] = 128. * outputs[0]

        cv2.imshow('fake', cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB))
        cv2.imshow('real', frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cap.release()


if __name__ == '__main__':
    opt = parse_args()
    # predict(opt)pyth
    predict_live(opt)
