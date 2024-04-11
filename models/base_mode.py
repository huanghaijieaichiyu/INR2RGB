from models.common import Conv, C2f, CA, SPPELAN, RepNCSPELAN4, Concat, EMA, ADown, SPPF, SimAM, CBAM
from utils.model_map import model_structure
import torch
import torch.nn as nn

from models.common import Conv, C2f, CA, SPPELAN, RepNCSPELAN4, Concat, EMA, ADown, Conv2dSame


# 模型体量太小的话，显卡占用会很低
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Sequential(
            Conv(1, 8, 3, 2),
            nn.LeakyReLU(),
            Conv(8, 16, 3),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(),
            Conv(16, 32, 5, 2),
            nn.LeakyReLU(),
            Conv(32, 32, 3),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(),
            Conv(32, 16, 3, 2),
            nn.LeakyReLU(),
            Conv(16, 8, 3),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(),
            Conv(8, 2, 3),
            nn.Tanh()  # 预测值映射
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 2, 480, 480)
        x = torch.permute(x, (0, 2, 3, 1))

        return x


class ConvertV1(nn.Module):

    def __init__(self):
        super(ConvertV1, self).__init__()
        self.conv1 = Conv(1, 8, 3)
        self.conv2 = CA(8, 8)
        self.conv3 = RepNCSPELAN4(8, 16, 16, 8)
        self.conv4 = RepNCSPELAN4(16, 32, 32, 16)
        self.conv5 = SPPELAN(32, 32, 16)
        self.conv6 = RepNCSPELAN4(32, 16, 16, 8)
        self.conv7 = RepNCSPELAN4(24, 12, 12, 6)
        self.conv8 = Conv(44, 8, 3)
        self.conv9 = Conv(8, 2, 3, act=False)
        self.tanh = nn.Tanh()
        self.concat = Concat()
        self.upsample = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool2d(4, 2, 1)

    def forward(self, x):
        # head net
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = self.upsample(x3)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = self.pool(x5)
        x6 = self.conv6(x5)

        # neck net
        x7 = self.concat([x2, x6])
        x7 = self.conv7(x7)
        x7 = self.concat([x5, x7])
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.tanh(x9)

        x = x10.view(-1, 2, x.shape[2], x.shape[3])

        return x


class Generator(nn.Module):
    """
    this is a light net for V1
    """


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
        self.liner = nn.Linear(8192, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.conv1(x1)
        x3 = self.conv_out(x2)
        x4 = x3.view(x.shape[0], -1)

        x = self.sig(x4)

        return x


if __name__ == '__main__':
    model = Discriminator()
    # model_ = Generator()
    d_params, d_macs = model_structure(model, (2, 256, 256))
    # d_params, d_macs = model_structure(model_, (1, 256, 256))
    print(d_params, d_macs)
