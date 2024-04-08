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
from models.base_mode import Generator
from utils.color_trans import PSrgb2lab, PSlab2rgb
from utils.model_map import model_structure
from utils.save_path import Path


def parse_args():
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str,
                        help="path to dataset", required=True)
    parser.add_argument("--model", type=str,
                        help="path to model", required=True)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=tuple,
                        default=(480, 480), help="size of the image")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of data loading workers, if in windows, must be 0"
                        )
    parser.add_argument("--device", type=str, default="cuda", choices='["cpu", "cuda"]',
                        help="select your device to train, if you have a gpu, use 'cuda:0'!")  # 训练设备
    parser.add_argument("--save_path", type=str, default='runs/',
                        help="where to save your data")  # 保存位置
    parser.add_argument("--sample_interval", type=int,
                        default=10, help="how often to sample the img")
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
    model = Generator()
    model_structure(model, (1, self.img_size[0], self.img_size[1]))
    checkpoint = torch.load(self.model)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    test_data = MyDataset(self.data, self.img_size)
    img_pil = transforms.ToPILImage()
    test_loader = DataLoader(test_data,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             drop_last=True)
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), bar_format='{l_bar}{bar:10}| {n_fmt}/{'
                                                                           'total_fmt} {elapsed}')
                                                                           'total_fmt} {elapsed}')
    model.eval()
    torch.no_grad()
    i= 0
    if not os.path.exists(os.path.join(path, 'predictions')):
        os.makedirs(os.path.join(path, 'predictions'))
    for data in pbar:
        target, (img1, label) = data
        img = img1[0]  # c=此步去除tensor中的bach-size 4维降3
        img = trans(img)
        x, y, image_size = process_image(img)

        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
        # 训练前交换维度
        x_trans = torch.permute(x, (0, 3, 1, 2))
        outputs = model(x_trans)

        x = x.cpu().data.numpy()
        x = x.astype(np.float32)
        outputs = outputs.cpu().data.numpy()
        outputs = outputs.astype(np.float32)
        tmp = np.zeros(
            (self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        # 训练后复原再拼接
        tmp[:, :, 0] = x[0][:, :, 0]
        tmp[:, :, 1:] = 128 * outputs[0]
        print('tmp is ', tmp)
        target, (img, label) = data

        img_lab = PSrgb2lab(img)
        gray, _, _ = torch.split(img_lab, [1, 1, 1], 1)

        lamb = 128.  # 取绝对值最大值，避免负数超出索引
        gray = gray.to(device)

        fake = model(gray)
        fake_tensor = torch.zeros((self.batch_size, 3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
        fake_tensor[:, 0, :, :] = gray[:, 0, :, :]  # 主要切片位置
        fake_tensor[:, 1:, :, :] = lamb * fake
        for j in range(self.batch_size):
            fake_img = np.array(img_pil(PSlab2rgb(fake_tensor)[j]), dtype=np.float32)

            if i > 10 and i % 10 == 0:  # 图片太多，十轮保存一次
                img_save_path = os.path.join(path, 'predictions', str(i) + '.jpg')
                cv2.imwrite(img_save_path, fake_img)
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
        # 是否需要resize取决于新图片格式与训练时的是否一致
        frame_pil = cv2.resize(frame_pil, self.img_size)
        frame_pil = Image.fromarray(frame_pil)
        x, _, image_size = process_image(frame_pil)

        x_trans = torch.tensor(x, dtype=torch.float).to(device)
        # 训练前交换维度
        x_trans = torch.permute(x_trans, (0, 3, 1, 2))
        outputs = model(x_trans)
        outputs = outputs.cpu().data.numpy()
        outputs = outputs.astype(np.float32)
        tmp = np.zeros(
            (self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        print('x[0] is ', x[0])
        print('outputs[0] is ', outputs[0])
        tmp[:, :, 0] = x[0][:, :, 0]
        tmp[:, :, 1:] = 128. * outputs[0]
        tmp = cv2.resize(tmp, (640, 480))

        cv2.imshow('fake', cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB))
        frame_pil = cv2.resize(frame_pil, self.img_size)  # 是否需要resize取决于新图片格式与训练时的是否一致

        frame_pil = torch.tensor(np.array(frame_pil, np.float32) / 255., dtype=torch.float32).to(device)  # 转为tensor
        frame_pil = torch.unsqueeze(frame_pil, 0).permute(0, 3, 1, 2)  # 提升维度--转换维度
        frame_lab = PSrgb2lab(frame_pil)  # 转为LAB
        gray, _, _ = torch.split(frame_lab, [1, 1, 1], 1)

        fake_ab = model(gray)
        fake_ab = fake_ab.permute(0, 2, 3, 1).detach().cpu().numpy()[0].astype(np.float32)
        fake = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        fake[:, :, 0] = gray.permute(0, 2, 3, 1).detach().cpu().numpy()[0][:, :, 0]
        fake[:, :, 1:] = fake_ab * 128
        fake = cv2.cvtColor(fake, cv2.COLOR_Lab2RGB)
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
