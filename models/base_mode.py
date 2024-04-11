import torch
import torch.nn as nn

from models.common import Conv, C2f, CA, SPPELAN, RepNCSPELAN4, Concat, EMA, ADown, SPPF, SimAM, CBAM
from utils.model_map import model_structure


class Generator(nn.Module):

    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.conv1 = Conv(1, 8, 3)
        self.conv2 = nn.Sequential(Conv(8, 16, 5),
                                   ADown(16, 16))
        self.conv3 = C2f(16, 32, 3)
        self.conv4 = nn.Sequential(Conv(32, 64, 5),
                                   ADown(64, 64),
                                   C2f(64, 128, 3)
                                   )
        self.conv5 = nn.Sequential(SPPELAN(128, 128, 64),
                                   ADown(128, 128))
        self.conv6 = nn.Sequential(C2f(128, 64),
                                   nn.Upsample(scale_factor=2),
                                   Conv(64, 64, 5))
        self.conv7 = nn.Sequential(C2f(64, 64),
                                   nn.Upsample(scale_factor=2))
        self.conv8 = nn.Sequential(Conv(96, 64, 3),
                                   nn.Upsample(scale_factor=2),
                                   )
        self.conv9 = nn.Sequential(C2f(64, 48),
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
        x10 = self.concat([x3, x7])
        x10 = self.conv8(x10)
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
                                   Conv(32, 16, 3, 2, act=False),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(3, 2, 1),
                                   Conv(16, 8, 3, 2, act=False),
                                   nn.LeakyReLU(),
                                   Conv(8, 4, 3, 2, act=False),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(3, 2, 1)
                                   )
        self.conv_out = Conv(4, 1, 3, act=False)  # 记得替换激活函数
        self.flat = nn.Flatten()
        self.liner = nn.Sequential(nn.Linear(16, 8),
                                   nn.LeakyReLU(),
                                   nn.Linear(8, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.conv1(x1)
        x3 = self.conv_out(x2)
        x4 = self.flat(x3)
        x5 = self.liner(x4)

        x = self.sig(x5)

        return x


if __name__ == '__main__':
    model = Discriminator()
    # model_ = Generator()
    d_params, d_macs = model_structure(model, (2, 256, 256))
    # d_params, d_macs = model_structure(model_, (1, 256, 256))
    print(d_params, d_macs)
