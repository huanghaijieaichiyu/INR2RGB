import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.data_set import MyDataset
from models.base_mode import ConvertV1, Generator
from utils.model_map import model_structure
from utils.save_path import Path


def parse_args():
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str, help="path to dataset", required=True)
    parser.add_argument("--model", type=str, help="path to model", required=True)
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=tuple, default=(480, 480), help="size of the image")
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
    model_structure(model, (1, self.img_size[0], self.img_size[1]))
    checkpoint = torch.load(self.model)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    test_data = MyDataset(self.data)
    trans = transforms.ToPILImage()
    img_2gray = transforms.Grayscale(3)
    test_loader = DataLoader(test_data,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             drop_last=True)
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), bar_format='{l_bar}{bar:10}| {n_fmt}/{'
                                                                           'total_fmt} {elapsed}')
    model.eval()
    torch.no_grad()
    i = 0
    if not os.path.exists(os.path.join(path, 'predictions')):
        os.makedirs(os.path.join(path, 'predictions'))
    for data in pbar:
        target, (img, label) = data
        img = img.to(device)
        img = img_2gray(img)
        fake = model(img)

        fake = fake.detach().cpu().numpy()
        fake = fake.astype(np.float32)
        fake = fake[0]
        print('fake is ', fake)
        if i > 10 and i % 10 == 0:  # 图片太多，十轮保存一次
            img_save_path = os.path.join(path, 'predictions', str(i) + '.jpg')
            cv2.imwrite(img_save_path, fake)
        i = i + 1
        pbar.set_description('Processed %d images' % i)
    pbar.close()


def predict_live(self):
    if self.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = Generator()
    model_structure(model, (1, self.img_size[0], self.img_size[1]))
    checkpoint = torch.load(self.model)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    cap = cv2.VideoCapture(2)  # 读取图像
    img_2gray = transforms.Grayscale()
    model.eval()
    torch.no_grad()
    if not os.path.exists(os.path.join(self.save_path, 'predictions')):
        os.makedirs(os.path.join(self.save_path, 'predictions'))
    while cap.isOpened():
        rect, frame = cap.read()
        frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = cv2.resize(frame_pil, self.img_size)  # 是否需要resize取决于新图片格式与训练时的是否一致

        frame_pil = torch.tensor(frame_pil, dtype=torch.float32).to(device)
        frame_pil = torch.permute(frame_pil, (2, 0, 1))  # HWC 2 CHW
        frame_pil = img_2gray(frame_pil)
        frame_pil = torch.unsqueeze(frame_pil, 0)  # 提升维度

        fake = model(frame_pil/255.)
        fake = fake.permute(0, 2, 3, 1)  # CHW2HWC
        fake = fake.detach().cpu().numpy()
        fake = fake[0]  # 降维度
        fake = fake.astype(np.float32)
        # fake *= 255.
        fake = cv2.resize(fake, (640, 480))  # 维度还没降下来

        cv2.imshow('fake', fake)

        cv2.imshow('real', frame)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cap.release()


if __name__ == '__main__':
    opt = parse_args()
    if opt.data == '0':

        predict_live(opt)
    else:
        predict(opt)