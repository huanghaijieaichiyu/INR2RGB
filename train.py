from utils.save_path import Path
from utils.model_map import model_structure
from utils.loss import BCEBlurWithLogitsLoss
from models.base_mode import Generator, Discriminator
from utils.colorful import PSrgb2lab, PSlab2rgb
from models.base_mode import ConvertV2
from datasets.data_set import MyDataset
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torch.backends import cudnn
from torch.cuda.amp import autocast
from torch import nn
from timm.optim import Lion, RMSpropTF
from timm.loss import SoftTargetCrossEntropy
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import argparse
import math
import math
import os
import random
import time

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from timm.optim import Lion, RMSpropTF
from torch import nn
from torch.cuda.amp import autocast
from torch.backends import cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms
from tqdm import tqdm

from datasets.data_set import MyDataset
from models.base_mode import Generator, Discriminator
from utils.color_trans import PSlab2rgb, PSrgb2lab
from utils.loss import BCEBlurWithLogitsLoss, FocalLoss
from models.base_mode import Generator, Discriminator
from utils.color_trans import PSlab2rgb, PSrgb2lab
from utils.loss import BCEBlurWithLogitsLoss, FocalLoss
from utils.model_map import model_structure
from utils.save_path import Path


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
    path = Path(self.save_path)
    os.makedirs(os.path.join(path, 'generator'))
    os.makedirs(os.path.join(path, 'discriminator'))
    os.makedirs(os.path.join(path, 'generator'))
    os.makedirs(os.path.join(path, 'discriminator'))
    # 创建训练日志文件
    train_log = path + '/log.txt'
    train_log_txt_formatter = '{time_str} [Epoch] {epoch:03d} [gLoss] {gloss_str} [dLoss] {dloss_str}\n'
    train_log_txt_formatter = '{time_str} [Epoch] {epoch:03d} [gLoss] {gloss_str} [dLoss] {dloss_str}\n'

    args_dict = self.__dict__
    print(args_dict)

    # 训练前数据准备
    device = torch.device('cpu')
    if self.device == 'cuda':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if self.cuDNN:
        assert device != 'cuda', 'cuDNN only work on cuda!'
        cudnn.benchmark = True

    log = tensorboard.SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard'),
                                    filename_suffix=str(self.epochs),
                                    flush_secs=180)
    set_random_seed(self.seed, deterministic=self.deterministic,
                    benchmark=self.benchmark)
    set_random_seed(self.seed, deterministic=self.deterministic,
                    benchmark=self.benchmark)

    # 选择模型参数

    generator = Generator()
    discriminator = Discriminator()

    generator = Generator()
    discriminator = Discriminator()

    print('-' * 100)
    print('Drawing model graph to tensorboard, you can check it with:http://127.0.0.1:6006 after running tensorboard '
          '--logdir={}'.format(os.path.join(self.save_path, 'tensorboard')))
    log.add_graph(generator, torch.randn(
        1, 1, self.img_size[0], self.img_size[1]))
    log.add_graph(generator, torch.randn(
        1, 1, self.img_size[0], self.img_size[1]))
    print('Drawing dnoe!')
    print('-' * 100)
    print('Generator model info: \n')
    g_params, g_macs = model_structure(
        generator, img_size=(1, self.img_size[0], self.img_size[1]))
    print('Discriminator model info: \n')
    d_params, d_macs = model_structure(
        discriminator, img_size=(2, self.img_size[0], self.img_size[1]))
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    # 打印配置
    with open(path + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
        f.writelines('\n' + 'The parameters of generator: {:.2f} M'.format(g_params) + '\n' + 'The Gflops of '
                                                                                              'generator: {:.2f}'
                                                                                              ' G'.format(g_macs))
        f.writelines('\n' + 'The parameters of discriminator: {:.2f} M'.format(d_params) + '\n' + 'The Gflops of '
                                                                                                  ' discriminator: {:.2f}'
                                                                                                  ' G'.format(d_macs))
        f.writelines('\n' + '-------------------------------------------')
    print('train models at the %s device' % device)
    f.writelines('\n' + 'The parameters of generator: {:.2f} M'.format(g_params) + '\n' + 'The Gflops of '
                 'generator: {:.2f}'
                 ' G'.format(g_macs))
    f.writelines('\n' + 'The parameters of discriminator: {:.2f} M'.format(d_params) + '\n' + 'The Gflops of '
                 ' discriminator: {:.2f}'
                 ' G'.format(d_macs))
    f.writelines('\n' + '-------------------------------------------')
    print('train models at the %s device' % device)
    os.makedirs(path, exist_ok=True)

    # 加载数据集
    # 加载数据集
    train_data = MyDataset(self.data, img_size=self.img_size)

    train_loader = DataLoader(train_data,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              drop_last=True)
    assert len(train_loader) != 0, 'no data loaded'

    if self.optimizer == 'AdamW' or self.optimizer == 'Adam':
        g_optimizer = torch.optim.AdamW(
            params=generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = torch.optim.AdamW(
            params=discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    if self.optimizer == 'AdamW':
        g_optimizer = torch.optim.AdamW(
            params=generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = torch.optim.AdamW(
            params=discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        g_optimizer = torch.optim.AdamW(
            params=generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = torch.optim.AdamW(
            params=discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'Adam':
        g_optimizer = torch.optim.Adam(
            params=generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = torch.optim.Adam(
            params=discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        g_optimizer = torch.optim.Adam(
            params=generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = torch.optim.Adam(
            params=discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'SGD':
        g_optimizer = torch.optim.SGD(
            params=generator.parameters(), lr=self.lr, momentum=self.momentum)
        d_optimizer = torch.optim.SGD(
            params=discriminator.parameters(), lr=self.lr, momentum=self.momentum)
        g_optimizer = torch.optim.SGD(
            params=generator.parameters(), lr=self.lr, momentum=self.momentum)
        d_optimizer = torch.optim.SGD(
            params=discriminator.parameters(), lr=self.lr, momentum=self.momentum)
    elif self.optimizer == 'lion':
        g_optimizer = Lion(params=generator.parameters(),
                           lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = Lion(params=discriminator.parameters(),
                           lr=self.lr, betas=(self.b1, self.b2))
        g_optimizer = Lion(params=generator.parameters(),
                           lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = Lion(params=discriminator.parameters(),
                           lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'rmp':
        g_optimizer = RMSpropTF(params=generator.parameters(), lr=self.lr, momentum=self.momentum,
                                lr_in_momentum=self.lr * self.momentum)
        d_optimizer = RMSpropTF(params=discriminator.parameters(), lr=self.lr, momentum=self.momentum,
                                lr_in_momentum=self.lr * self.momentum)
        g_optimizer = RMSpropTF(params=generator.parameters(), lr=self.lr, momentum=self.momentum,
                                lr_in_momentum=self.lr * self.momentum)
        d_optimizer = RMSpropTF(params=discriminator.parameters(), lr=self.lr, momentum=self.momentum,
                                lr_in_momentum=self.lr * self.momentum)
    else:
        raise ValueError('No such optimizer: {}'.format(self.optimizer))

    # 学习率退火
    if self.coslr:
        assert not self.llamb, 'do not using tow stagics at the same time!'
        LR_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer, len(train_loader) * self.epochs, 1e-6)
        LR_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            g_optimizer, len(train_loader) * self.epochs, 1e-6)

    if self.llamb:
        assert not self.coslr, 'do not using tow stagics at the same time!'

        def lf(x): return (
            (1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - 0.2) + 0.2
        LR_G = LambdaLR(
            g_optimizer, lr_lambda=lf, last_epoch=-1, verbose=False)
        LR_D = LambdaLR(d_optimizer, lr_lambda=lf,
                        last_epoch=-1, verbose=False)

    # 损失函数
    if self.loss == 'BCEBlurWithLogitsLoss':
        loss = BCEBlurWithLogitsLoss()
    elif self.loss == 'mse':
        loss = nn.MSELoss()
    elif self.loss == 'FocalLoss':
        loss = FocalLoss(nn.BCEWithLogitsLoss())
    elif self.loss == 'FocalLoss':
        loss = FocalLoss(nn.BCEWithLogitsLoss())
    elif self.loss == 'bce':
        loss = nn.BCEWithLogitsLoss()
    else:
        print('no such Loss Function!')
        raise NotImplementedError
    loss = loss.to(device)
    mse = nn.MSELoss()
    mse = mse.to(device)
    mse = nn.MSELoss()
    mse = mse.to(device)

    img_pil = transforms.ToPILImage()

    # 储存loss 判断模型好坏
    loss_all = [99.]
    # 寄存器判断模型提前终止条件
    per_G_loss = 99
    per_D_loss = 99

    toleration = 0
    loss_all = [99.]
    # 寄存器判断模型提前终止条件
    per_G_loss = 99
    per_D_loss = 99

    toleration = 0
    # 此处开始训练
    # 使用cuDNN加速训练
    if self.cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
    generator.train()
    discriminator.train()
    # 使用cuDNN加速训练
    if self.cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
    generator.train()
    discriminator.train()
    for epoch in range(self.epochs):
        d_epoch_loss = 0
        g_epoch_loss = 0
        d_epoch_loss = 0
        g_epoch_loss = 0
        # 断点训练参数设置
        if self.resume != ['']:

            g_path_checkpoint = self.resume[0]
            d_path_checkpoint = self.resume[1]
        if self.resume != ['']:

            g_path_checkpoint = self.resume[0]
            d_path_checkpoint = self.resume[1]

            g_checkpoint = torch.load(g_path_checkpoint)  # 加载断点
            generator.load_state_dict(g_checkpoint['net'])
            g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            g_epoch = g_checkpoint['epoch']  # 设置开始的epoch
            loss.load_state_dict = g_checkpoint['loss']

            d_checkpoint = torch.load(d_path_checkpoint)  # 加载断点
            discriminator.load_state_dict(d_checkpoint['net'])
            d_optimizer.load_state_dict(d_checkpoint['optimizer'])
            d_epoch = d_checkpoint['epoch']  # 设置开始的epoch
            loss.load_state_dict = d_checkpoint['loss']

            if g_epoch != d_epoch:
                print('given models are mismatched')
                raise NotImplementedError

            epoch = g_epoch

            g_checkpoint = torch.load(g_path_checkpoint)  # 加载断点
            generator.load_state_dict(g_checkpoint['net'])
            g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            g_epoch = g_checkpoint['epoch']  # 设置开始的epoch
            loss.load_state_dict = g_checkpoint['loss']

            d_checkpoint = torch.load(d_path_checkpoint)  # 加载断点
            discriminator.load_state_dict(d_checkpoint['net'])
            d_optimizer.load_state_dict(d_checkpoint['optimizer'])
            d_epoch = d_checkpoint['epoch']  # 设置开始的epoch
            loss.load_state_dict = d_checkpoint['loss']

            if g_epoch != d_epoch:
                print('given models are mismatched')
                raise NotImplementedError

            epoch = g_epoch

            print('继续第：{}轮训练'.format(epoch + 1))

            self.resume = ['']  # 跳出循环
            self.resume = ['']  # 跳出循环
        print('第{}轮训练'.format(epoch + 1))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}| {n_fmt}/{'
                                                                                 'total_fmt} {elapsed}')
        for data in pbar:
            target, (img, label) = data
            # print(img)
            # 对输入图像进行处理
            # print(img)
            # 对输入图像进行处理
            img_lab = PSrgb2lab(img)
            gray, a, b = torch.split(img_lab, [1, 1, 1], 1)
            color = torch.cat([a, b], dim=1)
            lamb = 128.  # 取绝对值最大值，避免负数超出索引
            gray = gray.to(device)
            color = color.to(device)

            '''img = img.to(device)
            img_gray = img_2gray(img)
            img_gray = img_gray.to(device)'''

            '''img = img.to(device)
            img_gray = img_2gray(img)
            img_gray = img_gray.to(device)'''

            with autocast(enabled=self.amp):
                '''---------------训练判别模型---------------'''
                real_outputs = discriminator(color / lamb)
                fake = generator(gray)  # 记得输入要换成明度！！！
                fake_outputs = discriminator(fake)
                d_optimizer.zero_grad()

                d_real_output = loss(real_outputs, torch.ones_like(
                    real_outputs))  # D 希望 real_loss 为 1

                d_fake_output = loss(fake_outputs, torch.zeros_like(
                    fake_outputs))  # D 希望 fake_loss 为 0

                d_output = (d_real_output + d_fake_output)*0.5
                d_output.backward()
                d_optimizer.step()
                if self.llamb or self.coslr:
                    LR_D.step()
                '''--------------- 训练生成器 ----------------'''
                fake = generator(gray)
                g_optimizer.zero_grad()
                fake_inputs = discriminator(fake)
                g_output = loss(fake_inputs, torch.ones_like(
                    fake_inputs))  # G 希望 fake_loss 为 1
                g_output.backward()
                g_optimizer.step()
                if self.llamb or self.coslr:
                    LR_G.step()

            # 判断模型是否需要提前终止
            if per_G_loss < g_output.item() and per_D_loss > d_output.item():
                toleration += 1
            if toleration > 99:
                break

            # 图像拼接还原
            fake_tensor = torch.zeros_like(img.detach(), dtype=torch.float32)
            fake_tensor[:, 0, :, :] = gray[:, 0, :, :]  # 主要切片位置
            fake_tensor[:, 1:, :, :] = lamb * fake
            fake_img = np.array(
                img_pil(PSlab2rgb(fake_tensor)[0]), dtype=np.float32)
            # print(fake_img)
            # 加入新的评价指标：PSN,SSIM
            real_pil = img_pil(img[0])
            psn = peak_signal_noise_ratio(
                np.array(real_pil, dtype=np.float32) / 255., fake_img / 255., data_range=1)
            psn = peak_signal_noise_ratio(
                np.array(real_pil, dtype=np.float32) / 255., fake_img / 255., data_range=1)

            pbar.set_description("Epoch [%d/%d] ----------- Batch [%d/%d] -----------  Generator loss: %.4f "
                                 "-----------  Discriminator loss: %.4f-----------"
                                 "-----------PSN: %.4f-------loss: %.4f"
                                 % (epoch + 1, self.epochs, target + 1, len(train_loader), g_output.item(),
                                    d_output.item(), psn, g_optimizer.state_dict()['param_groups'][0]['lr']))

        g_checkpoint = {
            'net': generator.state_dict(),
            'optimizer': g_optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss.state_dict()
        }
        d_checkpoint = {
            'net': discriminator.state_dict(),
            'optimizer': d_optimizer.state_dict(),
            g_checkpoint = {
                'net': generator.state_dict(),
                'optimizer': g_optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss.state_dict()
            }
            d_checkpoint = {
                'net': discriminator.state_dict(),
                'optimizer': d_optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss.state_dict()
            }
            # 保持最佳模型
            # 保持最佳模型

            if g_output.item() < min(loss_all):
            torch.save(g_checkpoint, path + '/generator/best.pt')
            loss_all.append(g_output.item())
            if g_output.item() < min(loss_all):
            torch.save(g_checkpoint, path + '/generator/best.pt')
            loss_all.append(g_output.item())

            # 保持训练权重
            torch.save(g_checkpoint, path + '/generator/last.pt')
            torch.save(d_checkpoint, path + '/discriminator/last.pt')
            torch.save(g_checkpoint, path + '/generator/last.pt')
            torch.save(d_checkpoint, path + '/discriminator/last.pt')

            # 写入日志文件
            to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                      epoch=epoch + 1,
                                                      gloss_str=" ".join(
                                                      ["{:4f}".format(g_output.item())]),
                                                      gloss_str=" ".join(
                                                      ["{:4f}".format(g_output.item())]),
                                                      dloss_str=" ".join(["{:4f}".format(d_output.item())]))
            with open(train_log, "a") as f:
            f.write(to_write)

            # 5 epochs for saving another model
            if (epoch + 1) % 10 == 0 and (epoch + 1) >= 10:
            torch.save(g_checkpoint, path + '/generator/%d.pt' % (epoch + 1))
            torch.save(d_checkpoint, path + '/discriminator/%d.pt' %
                       (epoch + 1))
            torch.save(d_checkpoint, path + '/discriminator/%d.pt' %
                       (epoch + 1))
            # 可视化训练结果
            log.add_images('real', img, epoch + 1)
            log.add_images('fake', PSlab2rgb(fake_tensor), epoch + 1)

            log.close()


            def parse_args():
            parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
            parser.add_argument("--data", type=str,
                                default='../datasets/coco5000', help="path to dataset")
            default = '../datasets/coco5000', help = "path to dataset")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of epochs of training")  # 迭代次数
    parser.add_argument("--batch_size", type=int, default=8,
                        help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=tuple,
                        default=(256, 256), help="size of the image")
    parser.add_argument("--optimizer", type=str, default='Adam',
                        choices=['AdamW', 'SGD', 'Adam', 'lion', 'rmp'])
    parser.add_argument("--num_workers", type=int, default=20,
                        help="number of data loading workers, if in windows, must be 0"
                        )
    parser.add_argument("--seed", type=int, default=1999, help="random seed")
    parser.add_argument("--resume", type=tuple,
                        default=[''], help="path to two latest checkpoint,yes or no")
    parser.add_argument("--amp", type=bool, default=True,
                        help="Whether to use amp in mixed precision")
    parser.add_argument("--cuDNN", type=bool, default=True,
                        help="Wether use cuDNN to celerate your program")
    parser.add_argument("--loss", type=str, default='bce',
                        choices=['BCEBlurWithLogitsLoss', 'mse', 'bce',
                                 'FocalLoss'],
                        'FocalLoss'],
    help = "loss function")
        parser.add_argument("--lr", type=float, default=3.5e-4,
    parser.add_argument("--lr", type=float, default=3.5e-4,
                        help="learning rate, for adam is 1-e3, SGD is 1-e2")  # 学习率
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="momentum for adam and SGD")
    parser.add_argument("--model", type=str, default="train",
                        help="train or test model")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")  # 动量梯度下降第一个参数
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")  # 动量梯度下降第二个参数
    parser.add_argument("--coslr", type=bool, default=False,
                        help="using cosine learning decay")
    parser.add_argument("--llamb", type=bool, default=False,
                        help="using yolo tactic")
    parser.add_argument("--coslr", type=bool, default=False,
                        help="using cosine learning decay")
    parser.add_argument("--llamb", type=bool, default=False,
                        help="using yolo tactic")
    parser.add_argument("--device", type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="select your device to train, if you have a gpu, use 'cuda:0'!")  # 训练设备
    parser.add_argument("--save_path", type=str, default='runs/',
                        help="where to save your data")  # 保存位置
    parser.add_argument("--benchmark", type=bool, default=False, help="whether using torch.benchmark to accelerate "
                                                                      "training(not working in interactive mode)")
    parser.add_argument("--deterministic", type=bool, default=True,
                        help="whether to use deterministic initialization")
    arges = parser.parse_args()

    return arges


if __name__ == '__main__':
    opt = parse_args()
    train(opt)
