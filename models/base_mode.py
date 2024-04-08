import torch
import torch.nn as nn

from models.common import Conv, C2f, CA, SPPELAN, RepNCSPELAN4, Concat, EMA, ADown, SPPF
from utils.model_map import model_structure


class Generator(nn.Module):

    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.conv1 = Conv(1, 8, 3)
        self.conv2 = RepNCSPELAN4(8, 16, 16, 8)
        self.conv3 = RepNCSPELAN4(16, 32, 32, 16)
        self.conv4 = nn.Sequential(RepNCSPELAN4(32, 64, 64, 32)
                                   )
        self.conv5 = SPPELAN(64, 64, 32)
        self.conv6 = nn.Sequential(C2f(64, 64))

        self.conv7 = nn.Sequential(Conv(80, 64),
                                   ADown(64, 64),
                                   nn.Upsample(scale_factor=2))
        self.conv8 = nn.Sequential(Conv(128, 64, 3))
        self.conv9 = nn.Sequential(C2f(64, 32),
                                   Conv(32, 16, 3))
        self.conv10 = nn.Sequential(C2f(16, 8),
                                    Conv(8, 2, 3, act=False)
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
        x7 = self.concat([x2, x6])
        x8 = self.conv7(x7)
        x9 = self.concat([x5, x8])
        x10 = self.conv8(x9)
        x11 = self.conv9(x10)
        x12 = self.conv10(x11)
        x13 = self.tanh(x12)

        x = x13.view(-1, 2, x.shape[2], x.shape[3])

        return x


class GeneratorV2(nn.Module):

    def __init__(self) -> None:
        super(GeneratorV2, self).__init__()
        self.conv1 = Conv(1, 8, 3)
        self.conv2 = Conv(8, 16, 5, 2)
        self.conv3 = RepNCSPELAN4(16, 32, 32, 16)
        self.conv4 = nn.Sequential(Conv(32, 64, 5, 2),
                                   RepNCSPELAN4(64, 128, 128, 64)
                                   )
        self.conv5 = SPPELAN(128, 128, 64)
        self.conv6 = nn.Sequential(C2f(128, 64),
                                   Conv(64, 64, 5, 2))
        self.conv7 = nn.Sequential(RepNCSPELAN4(64, 64, 64, 32),
                                   nn.Upsample(scale_factor=2))
        self.conv8 = nn.Sequential(Conv(192, 64, 3),
                                   nn.Upsample(scale_factor=2))
        self.conv9 = nn.Sequential(C2f(96, 48),
                                   nn.Upsample(scale_factor=2),
                                   Conv(48, 16, 3))
        self.conv10 = nn.Sequential(C2f(16, 8),
                                    Conv(8, 2, 3, act=False)
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

        x7 = self.conv7(x6)
        x8 = self.concat([x5, x7])
        x9 = self.conv8(x8)
        x10 = self.concat([x3, x9])
        x11 = self.conv9(x10)
        x12 = self.conv10(x11)
        x13 = self.tanh(x12)

        x = x13.view(-1, 2, x.shape[2], x.shape[3])

        return x


class Discriminator(nn.Module):
    """
    Discriminator model with no activation function
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_in = nn.Sequential(Conv(2, 16, 3, act=False),
                                     nn.LeakyReLU())
        self.conv1 = nn.Sequential(Conv(16, 32, 3, 2, act=False),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(3, 2, 1),
                                   Conv(32, 16, 3, 2, act=False),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(3, 2, 1),
                                   Conv(16, 8, 3, 2, act=False),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(3, 2, 1),
                                   Conv(8, 4, 3, 2, act=False),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(3, 2, 1)
                                   )
        self.conv_out = Conv(4, 1, 3, act=False)  # 记得替换激活函数
        self.flatten = nn.Flatten()
        self.sig = nn.Sigmoid()
        self.linear = nn.Sequential(nn.Linear(1024, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 32),
                                    nn.LeakyReLU(),
                                    nn.Linear(32, 16),
                                    nn.LeakyReLU(),
                                    nn.Linear(16, 8),
                                    nn.LeakyReLU(),
                                    nn.Linear(8, 4),
                                    nn.LeakyReLU(),
                                    nn.Linear(4, 1)
                                    )

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.conv1(x1)
        x3 = self.conv_out(x2)
        # x4 = self.flatten(x3)
        # x5 = self.linear(x4)  # 此处切换全连接 或者 深度卷积层
        x = self.sig(x3)

        return x


if __name__ == '__main__':
    # model = Discriminator()
    model_ = GeneratorV2()
    # d_params, d_macs = model_structure(model, (1, 256, 256))
    d_params, d_macs = model_structure(model_, (1, 256, 256))
    print(d_params, d_macs)
