import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from timm.optim import Lion, RMSpropTF

from utils.loss import BCEBlurWithLogitsLoss, FocalLoss


def learning_rate_scheduler(self, g_optimizer, d_optimizer):
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
    else:
        raise ValueError('lr_deduce must be one of "coslr", "llamb", "reduceLR"')

    return LR_D, LR_G


def set_random_seed(seed=10, deterministic=False, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


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


def get_loss(self):
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
    return loss