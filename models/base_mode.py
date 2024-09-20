import math

import torch.nn as nn

from models.MobileVit3 import MobileViTv3_block
from models.common import C2f, SPPELAN, Concat, Disconv, Gencov, C2fCIB, PSA, SCDown


class Generator(nn.Module):

    def __init__(self, depth=0.8, weight=1) -> None:
        super(Generator, self).__init__()
        depth = depth
        weight = weight
        self.conv1 = Gencov(1, math.ceil(8 * depth))
        self.conv2 = nn.Sequential(
            Gencov(math.ceil(8 * depth), math.ceil(16 * depth),
                   math.ceil(weight), 2),
            C2f(math.ceil(16 * depth), math.ceil(32 * depth), 1, True)
        )

        self.conv3 = nn.Sequential(
            Gencov(math.ceil(32 * depth),
                   math.ceil(64 * depth), math.ceil(weight), 2),
            C2f(math.ceil(64 * depth), math.ceil(128 * depth), 1, True)
        )

        self.conv4 = nn.Sequential(
            SCDown(math.ceil(128 * depth),
                   math.ceil(256 * depth), math.ceil(weight), 2),
            C2f(math.ceil(256 * depth), math.ceil(512 * depth), 1, True)
        )

        self.conv5 = nn.Sequential(
            SPPELAN(math.ceil(512 * depth), math.ceil(512 * depth),
                    math.ceil(256 * depth)),
            PSA(math.ceil(512 * depth), math.ceil(512 * depth)),
            Gencov(math.ceil(512 * depth), math.ceil(256 * depth)), )
        self.conv6 = nn.Sequential(
            Gencov(math.ceil(256 * depth),
                   math.ceil(128 * depth), math.ceil(3 * weight)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.conv7 = nn.Sequential(
            Gencov(math.ceil(256 * depth),
                   math.ceil(64 * depth), math.ceil(weight)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv8 = nn.Sequential(
            C2fCIB(math.ceil(96 * depth), math.ceil(32 * depth)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.conv9 = Gencov(math.ceil(32 * depth), 2,
                            math.ceil(weight), act=False, bn=False)
        self.tanh = nn.Tanh()
        self.concat = Concat()

    def forward(self, x):
        # head net
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        # neck net

        x7 = self.conv7(self.concat([x6, x3]))
        x8 = self.conv8(self.concat([x2, x7]))
        x9 = self.tanh(self.conv9(x8))

        return x9.view(-1, 2, x.shape[2], x.shape[3])


class Discriminator(nn.Module):
    """
    Discriminator model with no activation function
    """

    def __init__(self, batch_size=8, img_size=256):
        """
        :param batch_size: batch size\
        """
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        ratio = img_size / 256.
        self.conv_in = nn.Sequential(Disconv(2, 8, ),
                                     Disconv(8, 16, 3, 2),  # 128
                                     )
        self.conv1 = nn.Sequential(Disconv(16, 32, 3, 2),  # 64
                                   Disconv(32, 64),
                                   Disconv(64, 32, 3, 2),  # 32
                                   Disconv(32, 16),
                                   Disconv(16, 8, 3, 2),  # 16
                                   Disconv(8, 4),
                                   MobileViTv3_block(4, 4)
                                   )
        self.conv_out = Disconv(4, 1, bn=False, act=False)

        self.flat = nn.Flatten()

        self.liner = nn.Sequential(nn.Linear(math.ceil(16 * ratio) ** 2, 16 * 16),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(16 * 16),
                                   nn.Linear(16 * 16, 8 * 16),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(8 * 16),
                                   nn.Linear(8 * 16, 8 * 8),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(8 * 8),
                                   nn.Linear(8 * 8, 4 * 8),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(4 * 8),
                                   nn.Linear(4 * 8, 8),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(8),
                                   nn.Linear(8, 1))

        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: input image
        :return: output
        """
        # x = self.act(self.liner(self.flat(self.conv_out(self.conv1(self.conv_in(x)))))).view(
        #    self.batch_size if x.shape[0] == self.batch_size else x.shape[0], -1)
        x = self.act(self.conv_out(self.conv1(self.conv_in(x)))).view(
            self.batch_size if x.shape[0] == self.batch_size else x.shape[0], -1)
        return x


class DiscriminatorVit(nn.Module):
    def __init__(self, batch_size=8, img_size=256):
        super(DiscriminatorVit, self).__init__()
        self.batch_size = batch_size
        ratio = img_size / 256.
        self.conv_in = Disconv(2, 4, 1, 2)
        self.layer1 = MobileViTv3_block(4, 4)
        self.layer2 = Disconv(4, 1, 3, 1, bn=False, act=False)
        self.act = nn.Sigmoid()

    def forward(self, x):

        return self.act(self.layer2(self.layer1(self.conv_in(x))))
