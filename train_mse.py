import argparse
import math
import os
import random
import time

import numpy as np
import torch
from rich import print
from skimage.metrics import peak_signal_noise_ratio
from timm.optim import Lion, RMSpropTF
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.data_set import MyDataset
from models.base_mode import Convert
from utils.color_trans import PSlab2rgb, PSrgb2lab
from utils.loss import BCEBlurWithLogitsLoss, FocalLoss
from utils.model_map import model_structure
from utils.save_path import save_path


# 初始化随机种子
def set_random_seed(seed=10, deterministic=False, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


def train(self):
    # 避免同名覆盖
    path = save_path(self.save_path)
    os.makedirs(path)
    # 创建训练日志文件
    train_log = path + '/log.txt'
    train_log_txt_formatter = '{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n'

    args_dict = self.__dict__
    print(args_dict)

    # 训练前数据准备
    device = torch.device('cpu')
    if self.device == 'cuda':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    log = tensorboard.SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard'),
                                    filename_suffix=str(self.epochs),
                                    flush_secs=180)
    set_random_seed(self.seed, deterministic=self.deterministic,
                    benchmark=self.benchmark)

    # 选择模型参数

    model = Convert(self.depth, self.weight)
    print('model info: \n')
    params, macs = model_structure(
        model, img_size=(1, self.img_size[0], self.img_size[1]))
    model = model.to(device)
    # 打印配置
    with open(path + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
        f.writelines('\n' + 'The parameters of model: {:.2f} M'.format(params) + '\n' + 'The Gflops of '
                                                                                        'model: {:.2f}'
                                                                                        ' G'.format(macs))
        f.writelines('\n' + '-------------------------------------------')
    print('train models at the %s device' % device)
    os.makedirs(path, exist_ok=True)

    # 加载数据集
    train_data = MyDataset(self.data, img_size=self.img_size)

    train_loader = DataLoader(train_data,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              shuffle=True,
                              drop_last=True)
    assert len(train_loader) != 0, 'no data loaded'

    if self.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

    elif self.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

    elif self.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=1e-4)

    elif self.optimizer == 'lion':
        optimizer = Lion(params=model.parameters(),
                         lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'rmp':
        optimizer = RMSpropTF(params=model.parameters(), lr=self.lr)
    else:
        raise ValueError('No such optimizer: {}'.format(self.optimizer))

    # 学习率退火
    if self.lr_deduce == 'coslr':
        LR = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.epochs, 1e-6)

    if self.lr_deduce == 'llamb':
        assert self.lr_deduce != 'coslr', 'do not using tow stagics at the same time!'

        def lf(x): return (
                (1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - 0.2) + 0.2

        LR = LambdaLR(
            optimizer, lr_lambda=lf, last_epoch=-1, verbose=False)

    if self.lr_deduce == 'reduceLR':
        assert self.lr_deduce != 'coslr', 'do not using tow stagics at the same time!'
        LR = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.2, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel',
            cooldown=10, min_lr=1e-6, eps=1e-5)

    # 损失函数
    if self.loss == 'BCEBlurWithLogitsLoss':
        loss = BCEBlurWithLogitsLoss()
    elif self.loss == 'mse':
        loss = nn.MSELoss()
    elif self.loss == 'FocalLoss':
        loss = FocalLoss(nn.BCEWithLogitsLoss())
    elif self.loss == 'bce':
        loss = nn.BCEWithLogitsLoss()
    else:
        print('no such Loss Function!')
        raise NotImplementedError
    loss = loss.to(device)
    img_pil = transforms.ToPILImage()

    # 储存loss 判断模型好坏
    loss_all = [99.]

    # 寄存器判断模型提前终止条件
    per_loss = 99
    toleration = 0
    # 此处开始训练
    # 使用cuDNN加速训练
    if self.cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
    model.train()
    for epoch in range(self.epochs):
        # 参数储存
        PSN = []
        # 断点训练参数设置
        if self.resume != ['']:
            path_checkpoint = self.resume[0]

            checkpoint = torch.load(path_checkpoint)  # 加载断点
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']  # 设置开始的epoch
            loss.load_state_dict = checkpoint['loss']

            epoch = epoch

            print('继续第：{}轮训练'.format(epoch + 1))

            self.resume = ['']  # 跳出循环
        print('第{}轮训练'.format(epoch + 1))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}', postfix=dict, colour='#8762A5')
        for data in pbar:
            target, (img, _) = data
            # 对输入图像进行处理
            img_lab = PSrgb2lab(img)
            gray, a, b = torch.split(img_lab, 1, 1)
            color = torch.cat([a, b], dim=1)
            lamb = 128.  # 取绝对值最大值，避免负数超出索引
            gray = gray.to(device)
            color = color.to(device)

            with autocast(enabled=self.amp):
                optimizer.zero_grad()
                fake = model(gray)
                output = loss(fake, color/lamb)
                output.backward()
                optimizer.step()

            # 判断模型是否需要提前终止
            if per_loss == np.mean(output.mean().item()):
                toleration += 1
            if toleration > 99:
                break

            per_loss = output.mean().item()

            if output.mean().item() == 0.:
                break
            # 图像拼接还原
            fake_tensor = torch.zeros_like(img, dtype=torch.float32)
            fake_tensor[:, 0, :, :] = gray[:, 0, :, :]  # 主要切片位置
            fake_tensor[:, 1:, :, :] = lamb * fake

            fake_img = np.array(
                img_pil(PSlab2rgb(fake_tensor)[0]), dtype=np.float32)
            # print(fake_img)
            # 加入新的评价指标：PSN,SSIM

            real_pil = img_pil(img[0])
            psn = peak_signal_noise_ratio(
                np.array(real_pil, dtype=np.float32) / 255., fake_img / 255., data_range=1)

            PSN.append(psn)

            pbar.set_description("Epoch [%d/%d] ----------- Batch [%d/%d] ----------- loss: %.4f "
                                 "-----------PSN: %.4f-------learning ratio: %.4f"
                                 % (
                                     epoch + 1, self.epochs, target + 1, len(train_loader),
                                     np.mean(output.mean().item()),
                                     np.mean(PSN),
                                     optimizer.state_dict()['param_groups'][0]['lr']))

        # 学习率退火
        if self.lr_deduce == 'llamb' or 'coslr':
            LR.step()
        elif self.lr_deduce == 'reduceLR':
            LR.step(output)
        else:
            pass

        checkpoint = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss.state_dict() if loss is not None else None
        }

        # 保持最佳模型

        if np.mean(output.mean().item()) < min(loss_all):
            torch.save(checkpoint, path + '/best.pt')
        loss_all.append(np.mean(output.mean().item()))

        # 保持训练权重
        torch.save(checkpoint, path + '/last.pt')

        # 写入日志文件
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch + 1,
                                                  loss_str=" ".join(
                                                      ["{:4f}".format(np.mean(output.mean().item()))]))
        with open(train_log, "a") as f:
            f.write(to_write)

            # 5 epochs for saving another model
        if (epoch + 1) % 10 == 0 and (epoch + 1) >= 10:
            torch.save(checkpoint, path + '/%d.pt' % (epoch + 1))
        # 可视化训练结果

        log.add_scalar('generation loss', np.mean(output.mean().item()), epoch + 1)
        log.add_scalar('PSN', np.mean(PSN), epoch + 1)
        log.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch + 1)

        log.add_images('real', img, epoch + 1)
        log.add_images('fake', PSlab2rgb(fake_tensor), epoch + 1)

    log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str,
                        default='../datasets/coco5000', help="path to dataset")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs of training")  # 迭代次数
    parser.add_argument("--batch_size", type=int, default=8,
                        help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=tuple,
                        default=(400, 400), help="size of the image")
    parser.add_argument("--optimizer", type=str, default='AdamW',
                        choices=['AdamW', 'SGD', 'Adam', 'lion', 'rmp'])
    parser.add_argument("--num_workers", type=int, default=20,
                        help="number of data loading workers, if in windows, must be 0"
                        )
    parser.add_argument("--seed", type=int, default=1999, help="random seed")
    parser.add_argument("--resume", type=tuple,
                        default=[''], help="path to two latest "
                                           "checkpoint,yes or no")
    parser.add_argument("--amp", type=bool, default=True,
                        help="Whether to use amp in mixed precision")
    parser.add_argument("--cuDNN", type=bool, default=True,
                        help="Wether use cuDNN to celerate your program")
    parser.add_argument("--loss", type=str, default='mse',
                        choices=['BCEBlurWithLogitsLoss', 'mse', 'bce',
                                 'FocalLoss', 'wgb'],
                        help="loss function")
    parser.add_argument("--lr", type=float, default=4.5e-4,
                        help="learning rate, for adam is 1-e3, SGD is 1-e2")  # 学习率
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="momentum for adam and SGD")
    parser.add_argument("--depth", type=float, default=1,
                        help="depth of the generator")
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of the generator")
    parser.add_argument("--model", type=str, default="train",
                        help="train or test model")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")  # 动量梯度下降第一个参数
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")  # 动量梯度下降第二个参数
    parser.add_argument("--lr_deduce", type=str, default='llamb',
                        choices=['coslr', 'llamb', 'reduceLR', 'no'], help='using a lr tactic')

    parser.add_argument("--device", type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="select your device to train, if you have a gpu, use 'cuda:0'!")  # 训练设备
    parser.add_argument("--save_path", type=str, default='runs/',
                        help="where to save your data")  # 保存位置
    parser.add_argument("--benchmark", type=bool, default=False, help="whether using torch.benchmark to accelerate "
                                                                      "training(not working in interactive mode)")
    parser.add_argument("--deterministic", type=bool, default=True,
                        help="whether to use deterministic initialization")

    #  此处开始训练
    arges = parser.parse_args()

    train(arges)
