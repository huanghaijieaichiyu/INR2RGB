import argparse
import math
import os
import random
import time

import numpy as np
import torch
from torcheval.metrics.functional import peak_signal_noise_ratio
from timm.optim import Lion, RMSpropTF
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from rich import print

from datasets.data_set import MyDataset
from models.base_mode import Generator, Discriminator, Generator_lite
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
    os.makedirs(os.path.join(path, 'generator'))
    os.makedirs(os.path.join(path, 'discriminator'))
    # 创建训练日志文件
    train_log = path + '/log.txt'
    train_log_txt_formatter = (
        '{time_str} \t [Epoch] \t {epoch:03d} \t [gLoss] \t {gloss_str} \t [dLoss] \t {dloss_str} \t {Dx_str} \t [Dgz0] \t {Dgz0_str} \t [Dgz1] \t {Dgz1_str}\n')

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

    generator = Generator(self.depth, self.weight)
    discriminator = Discriminator(
        batch_size=self.batch_size, img_size=self.img_size[0])

    if self.draw_model:
        print('-' * 50)
        print('Drawing model graph to tensorboard, you can check it with:http://127.0.0.1:6006 in tensorboard '
              '--logdir={}'.format(os.path.join(self.save_path, 'tensorboard')))
        log.add_graph(generator, torch.randn(
            self.batch_size, 1, self.img_size[0], self.img_size[1]))
        print('Drawing doe!')
        print('-' * 50)
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
        g_optimizer = torch.optim.AdamW(
            params=generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = torch.optim.AdamW(
            params=discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'Adam':
        g_optimizer = torch.optim.Adam(
            params=generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = torch.optim.Adam(
            params=discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'SGD':
        g_optimizer = torch.optim.SGD(
            params=generator.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=1e-4)
        d_optimizer = torch.optim.SGD(
            params=discriminator.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=1e-4)
    elif self.optimizer == 'lion':
        g_optimizer = Lion(params=generator.parameters(),
                           lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = Lion(params=discriminator.parameters(),
                           lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'rmp':
        g_optimizer = RMSpropTF(params=generator.parameters(), lr=self.lr)
        d_optimizer = RMSpropTF(params=discriminator.parameters(), lr=self.lr)
    else:
        raise ValueError('No such optimizer: {}'.format(self.optimizer))

    # 学习率退火
    LR_D = None
    LR_G = None
    if self.lr_deduce == 'coslr':
        LR_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer, self.epochs, 1e-6)
        LR_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            g_optimizer, self.epochs, 1e-6)

    if self.lr_deduce == 'llamb':
        assert self.lr_deduce != 'coslr', 'do not using tow stagics at the same time!'

        def lf(x): return (
            (1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - 0.2) + 0.2

        LR_G = LambdaLR(
            g_optimizer, lr_lambda=lf, last_epoch=-1, verbose=False)
        LR_D = LambdaLR(d_optimizer, lr_lambda=lf,
                        last_epoch=-1, verbose=False)

    if self.lr_deduce == 'reduceLR':
        assert self.lr_deduce != 'coslr', 'do not using tow stagics at the same time!'
        LR_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
            d_optimizer, 'min', factor=0.2, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel',
            cooldown=10, min_lr=1e-6, eps=1e-5)
        LR_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            g_optimizer, 'min', factor=0.2, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel',
            cooldown=10, min_lr=1e-6, eps=1e-5)
    if self.lr_deduce == 'no':
        pass

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

    # 此处开始训练
    # 使用cuDNN加速训练
    if self.cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True

    # 开始训练
    discriminator.train()
    generator.train()
    for epoch in range(self.epochs):
        # 参数储存
        PSN = []
        fake_tensor = torch.zeros(
            (self.batch_size, 3, self.img_size[0], self.img_size[1]))
        d_g_z2 = 0.
        d_output = 0
        g_output = 0
        # 储存loss 判断模型好坏
        loss_all = [99.]
        gen_loss = []
        dis_loss = []
        # 断点训练参数设置
        if self.resume != ['']:
            g_path_checkpoint = self.resume[0]
            g_checkpoint = torch.load(g_path_checkpoint)  # 加载断点
            generator.load_state_dict(g_checkpoint['net'])
            g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            g_epoch = g_checkpoint['epoch']  # 设置开始的epoch
            loss.load_state_dict = g_checkpoint['loss']
            epoch = g_epoch
            print('继续第：{}轮训练'.format(epoch + 1))
            self.resume = ['']  # 跳出循环
        print('第{}轮训练'.format(epoch + 1))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}', colour='#8762A5')
        for data in pbar:
            target, (img, _) = data
            # 对输入图像进行处理
            img_lab = PSrgb2lab(img)
            gray, a, b = torch.split(img_lab, 1, 1)
            color = torch.cat([a, b], dim=1)
            # lamb = color.abs().max()  # 取绝对值最大值，避免负数超出索引
            lamb = 128.
            gray = gray.to(device)
            color = color.to(device)

            with autocast(enabled=self.amp):
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()

                '''---------------训练判别模型---------------'''
                fake = generator(gray)
                fake_inputs = discriminator(fake.detach())
                real_outputs = discriminator(color / lamb)
                real_lable = torch.ones_like(fake_inputs, requires_grad=False)
                fake_lable = torch.zeros_like(fake_inputs, requires_grad=False)
                # D 希望 real_loss 为 1
                d_real_output = loss(real_outputs, real_lable)
                d_real_output.backward()
                d_x = real_outputs.mean().item()
                # D希望 fake_loss 为 0
                d_fake_output = loss(fake_inputs, fake_lable)
                d_fake_output.backward()
                d_g_z1 = fake_inputs.mean().item()
                d_output = (d_real_output.item() + d_fake_output.item()) / 2.
                d_optimizer.step()

                '''--------------- 训练生成器 ----------------'''
                fake_inputs = discriminator(fake)
                # G 希望 fake 为 1
                g_output = loss(fake_inputs, real_lable)
                g_output.backward()
                d_g_z2 = fake_inputs.mean().item()
                g_optimizer.step()

            with torch.no_grad():

                gen_loss.append(g_output.item())
                dis_loss.append(d_output)
                # 图像拼接还原
                fake_tensor = torch.zeros_like(img, dtype=torch.float32)
                fake_tensor[:, 0, :, :] = gray[:, 0, :, :]  # 主要切片位置
                fake_tensor[:, 1:, :, :] = lamb * fake

                fake_img = PSlab2rgb(fake_tensor)
                psn = peak_signal_noise_ratio(
                    fake_img, img, data_range=255.)
                PSN.append(psn)
                pbar.set_description('Epoch: [%d/%d]\t Batch: [%d/%d]\t Loss_D: %.4f\t Loss_G: %.4f\t D(x): %.4f\t D(G('
                                     'z)): %.4f / %.4f\t PSN: %.4f\t learning ratio: %.4f'
                                     % (epoch + 1, self.epochs, target + 1, len(train_loader),
                                        d_output, g_output.item(), d_x, d_g_z1, d_g_z2, np.mean(PSN),
                                        g_optimizer.state_dict()['param_groups'][0]['lr']))
        # 判断模型是否提前终止
        if torch.eq(fake_tensor, torch.zeros_like(fake_tensor)).all():
            print('fake tensor is zero!')
            break
        if d_g_z2 == 0.:
            break

        # 学习率退火
        if self.lr_deduce == 'no':
            pass
        elif self.lr_deduce == 'reduceLR':
            assert LR_D is not None, 'no such lr deduce'
            assert LR_G is not None, 'no such lr deduce'
            LR_D.step(d_output)
            LR_G.step(g_output)
        elif self.lr_deduce == 'coslr' or 'lamb':
            assert LR_D is not None, 'no such lr deduce'
            assert LR_G is not None, 'no such lr deduce'
            LR_D.step()
            LR_G.step()
        g_checkpoint = {
            'net': generator.state_dict(),
            'optimizer': g_optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss.state_dict() if loss is not None else None
        }
        d_checkpoint = {
            'net': discriminator.state_dict(),
            'optimizer': d_optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss.state_dict() if loss is not None else None
        }
        # 保持最佳模型
        assert min(loss_all) > 0., 'loss_all is zero!'
        if np.mean(g_output.item()) < min(loss_all):
            torch.save(g_checkpoint, path + '/generator/best.pt')
        loss_all.append(np.mean(g_output.item()))

        # 保持训练权重
        torch.save(g_checkpoint, path + '/generator/last.pt')
        torch.save(d_checkpoint, path + '/discriminator/last.pt')

        # 写入日志文件
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch + 1,
                                                  gloss_str=" ".join(
                                                      ["{:4f}".format(np.mean(gen_loss))]),
                                                  dloss_str=" ".join(
                                                      ["{:4f}".format(np.mean(dis_loss))]),
                                                  Dx_str=" ".join(
                                                      ["{:4f}".format(d_x)]),
                                                  Dgz0_str=" ".join(
                                                      ["{:4f}".format(d_g_z1)]),
                                                  Dgz1_str=" ".join(
                                                      ["{:4f}".format(d_g_z2)]),
                                                  PSN_str=" ".join(["{:4f}".format(np.mean(PSN))]))
        with open(train_log, "a") as f:
            f.write(to_write)

            # 5 epochs for saving another model
        if (epoch + 1) % 10 == 0 and (epoch + 1) >= 10:
            torch.save(g_checkpoint, path + '/generator/%d.pt' % (epoch + 1))
            torch.save(d_checkpoint, path + '/discriminator/%d.pt' %
                       (epoch + 1))
        # 可视化训练结果

        log.add_scalar('generation loss', np.mean(gen_loss), epoch + 1)
        log.add_scalar('discrimination loss', np.mean(dis_loss), epoch + 1)
        log.add_scalar('PSN', np.mean(PSN), epoch + 1)
        log.add_scalar('learning rate', g_optimizer.state_dict()
                       ['param_groups'][0]['lr'], epoch + 1)

        log.add_images('real', img, epoch + 1)
        log.add_images('fake', fake_img, epoch + 1)
    pbar.close()
    log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str,
                        default='../datasets/coco5000', help="path to dataset")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of epochs of training")  # 迭代次数
    parser.add_argument("--batch_size", type=int, default=8,
                        help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=tuple,
                        default=(256, 256), help="size of the image")
    parser.add_argument("--optimizer", type=str, default='Adam',
                        choices=['AdamW', 'SGD', 'Adam', 'lion', 'rmp'])
    parser.add_argument("--num_workers", type=int, default=10,
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
    parser.add_argument("--loss", type=str, default='bce',
                        choices=['BCEBlurWithLogitsLoss', 'mse', 'bce',
                                 'FocalLoss', 'wgb'],
                        help="loss function")
    parser.add_argument("--lr", type=float, default=3.5e-4,
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
    parser.add_argument("--draw_model", type=bool, default=False,
                        help="whether to draw model graph to tensorboard")

    #  此处开始训练
    arges = parser.parse_args()

    train(arges)
