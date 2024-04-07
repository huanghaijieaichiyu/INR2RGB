import torch
import torch.nn as nn

from models.common import Conv, C2f, CA, SPPELAN, RepNCSPELAN4, Concat, EMA, ADown


class Generator(nn.Module):

    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.conv1 = Conv(1, 8, 3)
        self.conv2 = RepNCSPELAN4(8, 16, 16, 8)
        self.conv3 = RepNCSPELAN4(16, 32, 32, 16)
        self.conv4 = nn.Sequential(RepNCSPELAN4(32, 64, 64, 32),
                                   EMA(64),
                                   Conv(64, 128, 3))
        self.conv5 = SPPELAN(128, 128, 64)
        self.conv6 = nn.Sequential(Conv(128, 64, 5),
                                   ADown(64, 64),
                                   nn.Upsample(scale_factor=2))
        self.conv7 = nn.Sequential(Conv(80, 64, 3),
                                   ADown(64, 64),
                                   nn.Upsample(scale_factor=2))
        self.conv8 = nn.Sequential(Conv(192, 64, 3),
                                   ADown(64, 64),
                                   nn.Upsample(scale_factor=2))
        self.conv9 = nn.Sequential(Conv(64, 32, 3),
                                   Conv(32, 16, 3))
        self.conv10 = nn.Sequential(Conv(16, 8, 3),
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
        x7 = self.conv7(x7)
        x7 = self.concat([x5, x7])
        x = self.conv8(x7)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.tanh(x)

        x = x.view(-1, 2, x.shape[2], x.shape[3])

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_in = Conv(2, 8, 3, 2)
        self.conv1 = nn.Sequential(Conv(8, 16, 3),
                                   ADown(16, 16),
                                   nn.Upsample(scale_factor=2),
                                   SPPELAN(16, 16, 8),
                                   Conv(16, 8, 3),
                                   ADown(8, 8),
                                   nn.Upsample(scale_factor=2)
                                   )
        self.conv_out = Conv(8, 3, 3, 2, act=False)  # 记得替换激活函数
        self.sig = nn.Sigmoid()
        self.linear1 = nn.Linear(32, 1)

    def forward(self, x):
        x1 = self.conv_in(x)
        x4 = self.conv_out(x1)
        x = self.linear1(x4)
        x = self.sig(x)

        return x
