import os
import random
from copy import copy
from math import exp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from timm.optim import Lion, RMSpropTF
from torch.autograd import Variable

from utils.loss import BCEBlurWithLogitsLoss, FocalLoss


def set_random_seed(seed=10, deterministic=False, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
    return seed


def get_opt(self, generator, discriminator):
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

    return d_optimizer, g_optimizer


def get_loss(loss_name):
    # 损失函数
    if loss_name == 'BCEBlurWithLogitsLoss':
        loss = BCEBlurWithLogitsLoss()
    elif loss_name == 'mse':
        loss = nn.MSELoss()
    elif loss_name == 'FocalLoss':
        loss = FocalLoss(nn.BCEWithLogitsLoss())
    elif loss_name == 'bce':
        loss = nn.BCEWithLogitsLoss()
    else:
        print('no such Loss Function!')
        raise NotImplementedError
    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def save_path(path, model='train'):
    file_path = os.path.join(path, model)
    i = 1
    while os.path.exists(file_path):
        file_path = os.path.join(path, model + '(%i)' % i)
        i += 1

    return file_path


def model_structure(model, img_size):
    model_name = copy(model)
    blank = ' '
    print('-' * 142)
    print('|' + ' ' * 10 + 'weight name' + ' ' * 32 + '|'
          + ' ' * 21 + 'weight shape' + ' ' * 21 + '|'
          + ' ' * 5 + 'number' + ' ' * 5 + '|')
    print('-' * 142)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4
    macs, _ = get_model_complexity_info(model_name, img_size, as_strings=False, print_per_layer_stat=False,
                                        verbose=False)
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 67:
            key = key + (57 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 45:
            shape = shape + (42 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 142)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:.2f} M'.format(
        model._get_name(), num_para * type_size / 1000 / 1000))
    print('The Gflops of {}: {:.2f} G'.format(
        model._get_name(), (2 * int(macs) * 1e-9)))
    print('-' * 142)

    return num_para * 1e-6, 2 * macs * 1e-9
