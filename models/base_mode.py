import math

import torch.nn as nn

from models.common import Conv, C2f, SPPELAN, Concat, Disconv, Gencov
from utils.model_map import model_structure


class Convert(nn.Module):

    def __init__(self, depth=1, weight=1) -> None:
        super(Convert, self).__init__()
        depth = depth
        weight = weight
        self.conv1 = Conv(1, math.ceil(8 * depth))
        self.conv2 = nn.Sequential(Conv(math.ceil(8 * depth), math.ceil(16 * depth), math.ceil(3 * weight), 2),  # 128
                                   C2f(math.ceil(16 * depth), math.ceil(32 * depth), math.ceil(weight), shortcut=True)
                                   )
        self.conv3 = nn.Sequential(Conv(math.ceil(32 * depth), math.ceil(64 * depth), math.ceil(3 * weight), 2),  # 64
                                   C2f(math.ceil(64 * depth), math.ceil(128 * depth), math.ceil(weight), shortcut=True)
                                   )
        self.conv4 = nn.Sequential(Conv(math.ceil(128 * depth), math.ceil(256 * depth), math.ceil(3 * weight), 2),  # 32
                                   C2f(math.ceil(256 * depth), math.ceil(512 * depth), math.ceil(weight),
                                       shortcut=True))
        self.conv5 = nn.Sequential(Conv(math.ceil(512 * depth), math.ceil(1024 * depth), math.ceil(3 * weight), 2),
                                   # 16
                                   Conv(math.ceil(1024 * depth), math.ceil(512 * depth), math.ceil(3 * weight))
                                   )
        self.conv6 = nn.Sequential(SPPELAN(math.ceil(512 * depth), math.ceil(512 * depth), math.ceil(256 * depth)),
                                   Conv(math.ceil(512 * depth), math.ceil(256 * depth), math.ceil(3 * weight)))
        self.conv7 = nn.Sequential(
            C2f(math.ceil(256 * depth), math.ceil(128 * depth), math.ceil(weight), shortcut=False),
            nn.Upsample(scale_factor=2),  # 32
            Conv(math.ceil(128 * depth), math.ceil(64 * depth)))
        self.conv8 = nn.Sequential(
            C2f(math.ceil(576 * depth), math.ceil(256 * depth), math.ceil(weight), shortcut=True),
            nn.Upsample(scale_factor=2),  # 64
            Conv(math.ceil(256 * depth), math.ceil(128 * depth), math.ceil(3 * weight)),
            C2f(math.ceil(128 * depth), math.ceil(64 * depth), math.ceil(weight), shortcut=False))
        self.conv9 = nn.Sequential(C2f(math.ceil(64 * depth), math.ceil(64 * depth), math.ceil(weight), shortcut=True),
                                   nn.Upsample(scale_factor=2),  # 128
                                   Conv(math.ceil(64 * depth), math.ceil(128 * depth))
                                   )
        self.conv10 = nn.Sequential(
            C2f(math.ceil(128 * depth), math.ceil(64 * depth), math.ceil(weight), shortcut=False),
            nn.Upsample(scale_factor=2),  # 256
            Conv(math.ceil(64 * depth), math.ceil(32 * depth))
        )
        self.conv11 = nn.Sequential(
            C2f(math.ceil(32 * depth), math.ceil(16 * depth), math.ceil(weight), shortcut=False),
            Conv(math.ceil(16 * depth), math.ceil(8 * depth)),
            Conv(math.ceil(8 * depth), 2, math.ceil(3 * weight), act=False)
        )
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
        x7 = self.conv7(x6)
        # neck net

        x8 = self.conv8(self.concat([x7, x4]))
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.tanh(x11)

        x = x12.view(-1, 2, x.shape[2], x.shape[3])

        return x


class Generator(nn.Module):

    def __init__(self, depth=0.8, weight=1) -> None:
        super(Generator, self).__init__()
        depth = depth
        weight = weight
        self.conv1 = Conv(1, math.ceil(8 * depth))
        self.conv2 = nn.Sequential(Gencov(math.ceil(8 * depth), math.ceil(16 * depth), math.ceil(3 * weight), 2),
                                   C2f(math.ceil(16 * depth), math.ceil(32 * depth), math.ceil(weight), shortcut=True)
                                   )
        self.conv3 = nn.Sequential(Gencov(math.ceil(32 * depth), math.ceil(64 * depth), math.ceil(3 * weight), 2),
                                   C2f(math.ceil(64 * depth), math.ceil(128 * depth), math.ceil(weight), shortcut=True)
                                   )
        self.conv4 = nn.Sequential(Gencov(math.ceil(128 * depth), math.ceil(256 * depth), math.ceil(3 * weight), 2),
                                   C2f(math.ceil(256 * depth), math.ceil(512 * depth), math.ceil(weight), shortcut=True),
                                   Gencov(math.ceil(512 * depth), math.ceil(1024 * depth), math.ceil(3 * weight)),
                                   Gencov(math.ceil(1024 * depth), math.ceil(512 * depth))
                                   )
        self.conv5 = nn.Sequential(SPPELAN(math.ceil(512 * depth), math.ceil(512 * depth), math.ceil(256 * depth)),
                                   Gencov(math.ceil(512 * depth), math.ceil(256 * depth), math.ceil(3 * weight)))
        self.conv6 = nn.Sequential(C2f(math.ceil(256 * depth), math.ceil(128 * depth), math.ceil(weight), shortcut=False),
                                   nn.Upsample(scale_factor=2),
                                   Gencov(math.ceil(128 * depth), math.ceil(64 * depth)))
        self.conv7 = nn.Sequential(C2f(math.ceil(192 * depth), math.ceil(96 * depth), math.ceil(weight), shortcut=True),
                                   nn.Upsample(scale_factor=2),
                                   Gencov(math.ceil(96 * depth), math.ceil(64 * depth)))
        self.conv8 = nn.Sequential(C2f(math.ceil(96 * depth), math.ceil(64 * depth), math.ceil(weight), shortcut=True),
                                   nn.Upsample(scale_factor=2),
                                   Gencov(math.ceil(64 * depth), math.ceil(128 * depth))
                                   )
        self.conv9 = nn.Sequential(
            C2f(math.ceil(128 * depth), math.ceil(64 * depth), math.ceil(weight), shortcut=False),
            Gencov(math.ceil(64 * depth), math.ceil(32 * depth))
        )
        self.conv10 = nn.Sequential(
            C2f(math.ceil(32 * depth), math.ceil(16 * depth), math.ceil(weight), shortcut=False),
            Gencov(math.ceil(16 * depth), math.ceil(8 * depth)),
            Gencov(math.ceil(8 * depth), 2, math.ceil(3 * weight), act=False, bn=False)
        )
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
        x8 = self.conv8(self.concat([x7, x2]))
        x10 = self.conv9(x8)
        x11 = self.conv10(x10)
        x12 = self.tanh(x11)

        x = x12.view(-1, 2, x.shape[2], x.shape[3])

        return x


class Discriminator(nn.Module):
    """
    Discriminator model with no activation function
    """

    def __init__(self, depth=1, batch_size=8):
        """
        :param batch_size: batch size
        """
        self.batch_size = batch_size
        super(Discriminator, self).__init__()
        self.conv_in = nn.Sequential(Disconv(2, 16, 3)
                                     )
        self.conv1 = nn.Sequential(Disconv(16, 32, 3, 2, bn=False),  # 128
                                   Disconv(32, 64, 3),
                                   Disconv(64, 128, 3, 2, bn=False),  # 64
                                   Disconv(128, 64, 3),
                                   Disconv(64, 32, 3, 2, bn=False),  # 32
                                   Disconv(32, 16, 3),
                                   Disconv(16, 8, 3, 2, bn=False),  # 16
                                   Disconv(8, 4, 3)
                                   )
        self.conv_out = Disconv(4, 1, 3, bn=False, act=False)

        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: input image
        :return: output
        """
        x1 = self.act(self.conv_out(self.conv1(self.conv_in(x))))

        return x1.view(self.batch_size if x.shape[0] == self.batch_size else x.shape[0], -1)


if __name__ == '__main__':
    # model = Discriminator()
    model_ = Generator(1, 1)
    #d_params, d_macs = model_structure(model, (2, 256, 256))
    d_params, d_macs = model_structure(model_, (1, 256, 256))
    print(d_params, d_macs)
