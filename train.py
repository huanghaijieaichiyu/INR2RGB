import argparse
import os
import random

import numpy as np
import torch
from timm.loss import SoftTargetCrossEntropy
from timm.optim import Lion
from torch import nn
from torch.cuda.amp import autocast
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets.data_set import MyDataset
from models.base_mode import BaseModel, ConvertV1, ConvertV2
from utils.img_progress import process_image
from utils.loss import BCEBlurWithLogitsLoss
from utils.model_map import model_structure


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

    args_dict = self.__dict__
    # 打印配置
    with open(opt.save_path + 'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')



    # 训练前数据准备
    device = torch.device('cpu')
    if self.device == 'cuda':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    log = tensorboard.SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard'),
                                    filename_suffix=str(self.epochs),
                                    flush_secs=180)
    set_random_seed(self.seed, deterministic=self.deterministic, benchmark=self.benchmark)

    # 选择模型参数

    mode = ConvertV1()
    mode = mode.to(device)
    print(mode)
    print('train model at the %s device' % device)
    os.makedirs(self.save_path, exist_ok=True)
    # assert self.batch_size ==1,'batch-size > 1 may led some question when detecting by caps'
    train_data = MyDataset(self.data)

    train_loader = DataLoader(train_data,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              drop_last=False)
    assert len(train_loader) != 0, 'no data loaded'
    # print(train_loader)
    if self.optimizer == 'AdamW' or self.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(params=mode.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    elif self.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=mode.parameters(), lr=self.lr, momentum=self.momentum)
    elif self.optimizer == 'lion':
        optimizer = Lion(params=mode.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    else:
        raise ValueError('No such optimizer: {}'.format(self.optimizer))

    if self.loss == 'BCEBlurWithLogitsLoss':
        loss = BCEBlurWithLogitsLoss()
    elif self.loss == 'mse':
        loss = nn.MSELoss()
    elif self.loss == 'SoftTargetCrossEntropy':
        loss = SoftTargetCrossEntropy()
    else:
        print('no such Loss Function!')
        raise NotImplementedError
    loss = loss.to(device)
    # log.add_graph(mode)
    img_transform = transforms.Compose([
        transforms.ToPILImage()
    ])
    model_structure(mode)
    print('begin training...')
    # 此处开始训练
    mode.train()
    for epoch in range(self.epochs):
        # 断点训练参数设置
        if self.resume:
            if self.model_path == '':
                print('No model offered to resume, using last.pt!')
                path_checkpoint = 'runs/last.pt'
            else:
                path_checkpoint = self.model_path  # 断点路径

            checkpoint = torch.load(path_checkpoint)  # 加载断点
            mode.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']  # 设置开始的epoch
            loss.load_state_dict = checkpoint['loss']
            print('继续第：{}轮训练'.format(epoch + 1))
            self.resume = False
        else:
            pass

        print('第{}轮训练'.format(epoch + 1))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}| {n_fmt}/{'
                                                                                 'total_fmt} {elapsed}')
        for data in pbar:
            target, (img1, label) = data
            img = img1[0]  # c=此步去除tensor中的bach-size 4维降3
            img = img_transform(img)
            x, y, image_size = process_image(img)

            x = torch.tensor(x, dtype=torch.float16)
            x = x.to(device)
            y = torch.tensor(y, dtype=torch.float16)
            y = y.to(device)

            optimizer.zero_grad()
            with autocast(enabled=self.amp):
                # 训练前交换维度
                x_trans = x.transpose(1, 3)  # 这样交换 是 (B H W C) to (B C W H) , 需要再做一次转化2，3维度:(B C H W)
                x_trans = x_trans.transpose(2, 3)
                x_trans = mode(x_trans)
                output = loss(x_trans, y)  # ---大坑--损失函数计算必须全是tensor
                output.backward()
                optimizer.step()
            accuracy = torch.eq(output, y).float().mean()
            pbar.set_description("Epoch [%d/%d] ---------------  Batch [%d/%d] ---------------  loss: %.4f "
                                 "---------------"
                                 "accuracy: %.4f"
                                 % (epoch, self.epochs, target, len(train_loader), output.item(), accuracy))

            checkpoint = {
                'net': mode.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss.state_dict()
            }
            log.add_scalar('total loss', output.item(), epoch)
            torch.save(checkpoint, self.save_path + 'last.pt')
        # 5 epochs for saving another model
            if epoch % 10 == 0 and epoch >= 10:
                torch.save(checkpoint, self.save_path + '%d.pt' % epoch)

    log.close()


def parse_args():
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str, help="path to dataset", required=True)
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")  # 迭代次数
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=int, default=640, help="size of the image")
    parser.add_argument("--optimizer", type=str, default='lion', choices=['AdamW', 'SGD', 'Adam', 'lion'])
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of data loading workers, if in windows, must be 0"
                        )
    parser.add_argument("--seed", type=int, default=1999, help="random seed")
    parser.add_argument("--resume", type=bool, default=False, help="path to latest checkpoint,yes or no")
    parser.add_argument("--model_path", type=str, default='', help="path to saved model")
    parser.add_argument("--amp", type=bool, default=True, help="Whether to use amp in mixed precision")
    parser.add_argument("--loss", type=str, default='mse', choices=['BCEBlurWithLogitsLoss', 'mse',
                                                                    'SoftTargetCrossEntropy'],
                        help="loss function")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate, for adam is 1-e3, SGD is 1-e2")  # 学习率
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
    opt = parser.parse_args()
    print(opt)


    return opt


if __name__ == '__main__':
    opt = parse_args()
    train(opt)
