import torch
import torch.nn as nn
import torch.nn.functional as F

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

        x7 = self.conv7(x6)
        x10 = self.concat([x3, x7])
        x10 = self.conv8(x10)
        x11 = self.conv9(x10)
        x12 = self.conv10(x11)
        x13 = self.tanh(x12)

        x = x13.view(-1, 2, x.shape[2], x.shape[3])

        return x


class ConvertV2(nn.Module):
    """
    this is a light net for V1
    """

    def __init__(self) -> None:
        super(ConvertV2, self).__init__()
        self.conv1 = Conv(1, 8, 3)
        self.conv2 = nn.Sequential(RepNCSPELAN4(8, 16, 16, 8),
                                   ADown(16, 16),
                                   C2f(16, 16, 2))
        self.conv3 = nn.Sequential(RepNCSPELAN4(16, 32, 32, 16),
                                   ADown(32, 32),
                                   C2f(32, 32))
        self.conv4 = nn.Sequential(RepNCSPELAN4(32, 64, 64, 32),
                                   ADown(64, 64),
                                   Conv(64, 128, 3),
                                   C2f(128, 128, 3))
        self.conv5 = SPPELAN(128, 128, 64)
        self.conv6 = nn.Sequential(C2f(128, 64),
                                   nn.Upsample(scale_factor=2))
        self.conv7 = nn.Sequential(Conv(64, 64, 3),
                                   nn.Upsample(scale_factor=2))
        self.conv8 = nn.Sequential(Conv(80, 64, 3),
                                   nn.Upsample(scale_factor=2))
        self.conv9 = nn.Sequential(C2f(64, 32, 3),
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
        x7 = self.conv7(x6)
        x7 = self.concat([x2, x7])
        x = self.conv8(x7)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.tanh(x)

        x = x.view(-1, 2, x.shape[2], x.shape[3])

        return x
