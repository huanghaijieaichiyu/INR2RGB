import argparse
import os
import random
import time

import numpy as np
import torch
from timm.loss import SoftTargetCrossEntropy
from timm.optim import Lion, RMSpropTF
from torch import nn
from torch.cuda.amp import autocast
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from skimage.metrics import structural_similarity
from tqdm import tqdm

from datasets.data_set import MyDataset
from models.base_mode import ConvertV1, ConvertV2, BaseModel
from utils.img_progress import process_image
from utils.loss import BCEBlurWithLogitsLoss
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
    set_random_seed(self.seed, deterministic=self.deterministic, benchmark=self.benchmark)

    # 选择模型参数

    mode = ConvertV2()
    print('-' * 100)
    print('Drawing model graph to tensorboard, you can check it with:http://127.0.0.1:6006 after running tensorboard '
          '--logdir={}'.format(os.path.join(self.save_path, 'tensorboard')))
    log.add_graph(mode, torch.randn(1, 1, self.img_size[0], self.img_size[1]))
    print('Drawing dnoe!')
    print('-' * 100)
    params, macs = model_structure(mode, img_size=(1, self.img_size[0], self.img_size[1]))
    mode = mode.to(device)
    # 打印配置
    with open(path + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
        f.writelines('\n' + 'The parameters of Model ConvertV1: {:.2f} M'.format(params) + '\n' + 'The Gflops of '
                                                                                                  'ConvertV1: {:.2f}'
                                                                                                  ' G'.format(macs))
    print('train model at the %s device' % device)
    os.makedirs(path, exist_ok=True)

    train_data = MyDataset(self.data, img_size=self.img_size)

    train_loader = DataLoader(train_data,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              drop_last=True)
    assert len(train_loader) != 0, 'no data loaded'
    if self.optimizer == 'AdamW' or self.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(params=mode.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=mode.parameters(), lr=self.lr, momentum=self.momentum)
    elif self.optimizer == 'lion':
        optimizer = Lion(params=mode.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'rmp':
        optimizer = RMSpropTF(params=mode.parameters(), lr=self.lr, momentum=self.momentum,
                              lr_in_momentum=self.lr * self.momentum)
    else:
        raise ValueError('No such optimizer: {}'.format(self.optimizer))

    if self.loss == 'BCEBlurWithLogitsLoss':
        loss = BCEBlurWithLogitsLoss()
    elif self.loss == 'mse':
        loss = nn.MSELoss()
    elif self.loss == 'SoftTargetCrossEntropy':
        loss = SoftTargetCrossEntropy()
    elif self.loss == 'bce':
        loss = nn.BCEWithLogitsLoss()
    else:
        print('no such Loss Function!')
        raise NotImplementedError
    loss = loss.to(device)

    img_transform = transforms.ToPILImage()
    img_gray = transforms.Grayscale(1)
    # 储存loss 判断模型好坏
    Loss = [1.0]

    # 此处开始训练
    mode.train()
    for epoch in range(self.epochs):

        # 断点训练参数设置
        if self.resume != '':
            path_checkpoint = self.resume

            checkpoint = torch.load(path_checkpoint)  # 加载断点
            mode.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']  # 设置开始的epoch
            loss.load_state_dict = checkpoint['loss']
            print('继续第：{}轮训练'.format(epoch + 1))

        print('第{}轮训练'.format(epoch + 1))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}| {n_fmt}/{'
                                                                                 'total_fmt} {elapsed}')
        for data in pbar:
            target, (img, label) = data

            img = img.to(device)
            gray = img_gray(img)

            optimizer.zero_grad()
            with autocast(enabled=self.amp):
                fake = mode(gray)
                output = loss(fake, img)  # ---大坑--损失函数计算必须全是tensor
                output.backward()
                optimizer.step()

            with torch.no_grad(): # 不需要梯度操作，节省显存空间

                # 加入新的评价指标：PSNR,SSIM
                '''max_pix = 255.
                psnr = 10 * np.log10((max_pix ** 2) / output.item())

                ssim = structural_similarity(np.array(img_transform(fake[0]), dtype=np.float32),
                                             np.array(img_transform(img[0]),
                                                      dtype=np.float32), channel_axis=2, data_range=1)'''

                pbar.set_description("Epoch [%d/%d] ---------------  Batch [%d/%d] ---------------  loss: %.4f "
                                     "---------------"

                                     % (epoch + 1, self.epochs, target + 1, len(train_loader), output.item()))

        checkpoint = {
            'net': mode.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss.state_dict()
        }
        log.add_scalar('total loss', output.item(), epoch)

        # 依据损失和相似度来判断最佳模型

        if output.item() <= min(Loss):
            torch.save(checkpoint, path + '/best.pt')

        Loss.append(output.item())
        # 保持训练权重
        torch.save(checkpoint, path + '/last.pt')

        # 写入日志文件
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch + 1,
                                                  loss_str=" ".join(["{:4f}".format(output.item())]))
        with open(train_log, "a") as f:
            f.write(to_write)

            # 5 epochs for saving another model
        if (epoch + 1) % 10 == 0 and (epoch + 1) >= 10:
            torch.save(checkpoint, path + '/%d.pt' % (epoch + 1))
        log.add_images('fake', fake, epoch)
        log.add_images('real', img, epoch)
    log.close()


def parse_args():
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str, help="path to dataset", required=True)
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs of training")  # 迭代次数
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=tuple, default=(480, 480), help="size of the image")
    parser.add_argument("--optimizer", type=str, default='lion', choices=['AdamW', 'SGD', 'Adam', 'lion', 'rmp'])
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of data loading workers, if in windows, must be 0"
                        )
    parser.add_argument("--seed", type=int, default=1999, help="random seed")
    parser.add_argument("--resume", type=str, default='', help="path to latest checkpoint,yes or no")
    parser.add_argument("--amp", type=bool, default=True, help="Whether to use amp in mixed precision")
    parser.add_argument("--loss", type=str, default='mse', choices=['BCEBlurWithLogitsLoss', 'mse', 'bce',
                                                                    'SoftTargetCrossEntropy'],
                        help="loss function")
    parser.add_argument("--lr", type=float, default=6.4e-5, help="learning rate, for adam is 1-e3, SGD is 1-e2")  # 学习率
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for adam and SGD")
    parser.add_argument("--model", type=str, default="train", help="train or test model")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")  # 动量梯度下降第一个参数
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")  # 动量梯度下降第二个参数
    parser.add_argument("--device", type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="select your device to train, if you have a gpu, use 'cuda:0'!")  # 训练设备
    parser.add_argument("--save_path", type=str, default='runs/', help="where to save your data")  # 保存位置
    parser.add_argument("--benchmark", type=bool, default=False, help="whether using torch.benchmark to accelerate "
                                                                      "training(not working in interactive mode)")
    parser.add_argument("--deterministic", type=bool, default=True, help="whether to use deterministic initialization")
    arges = parser.parse_args()

    return arges


if __name__ == '__main__':
    opt = parse_args()
    train(opt)
