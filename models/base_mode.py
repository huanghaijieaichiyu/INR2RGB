import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Conv2dSAME import Conv2dSame
from models.common import Conv, C2f, CA, SPPELAN, RepNCSPELAN4, Concat


# 模型体量太小的话，显卡占用会很低
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2dSame(1, 8, 3, 2),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            CA(16, 16),
            Conv2dSame(16, 16, 3, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            CA(32, 32),
            Conv2dSame(32, 32, 3, 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 2, 3, padding='same'),
            nn.Tanh()  # 预测值映射
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 2, 640, 640)
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
        self.conv8 = Conv(28, 8, 3)
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
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = self.upsample(x5)
        x6 = self.conv6(x5)
        x6 = self.pool(x6)

        # neck net
        x7 = self.concat([x2, x6])
        x7 = self.conv7(x7)
        x8 = self.concat([x3, x7])
        x = self.conv8(x8)
        x = self.conv9(x)
        x = self.tanh(x)

        x = x.view(-1, 2, 640, 640)
        x = torch.permute(x, (0, 2, 3, 1))

        return x


class ConvertV2(nn.Module):
    """
    this is a light net for V1
    """

    def __init__(self) -> None:
        super(ConvertV2, self).__init__()
        self.conv1 = Conv(1, 8, 3)
        self.conv2 = CA(8, 8)
        self.conv3 = C2f(8, 16, shortcut=True)
        self.conv4 = C2f(16, 32)
        self.conv5 = SPPELAN(32, 32, 16)
        self.conv6 = C2f(48, 16)
        self.conv7 = C2f(16, 8, shortcut=True)
        self.conv8 = Conv(16, 8, 3)
        self.conv9 = Conv(8, 2, 3, act=False)
        self.tanh = nn.Tanh()
        self.concat = Concat()
        self.upsample = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool2d(4, 2, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = self.upsample(x4)
        x5 = self.conv5(x4)
        x5 = self.pool(x5)
        x5 = self.concat([x3, x5])
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x7 = self.concat([x2, x7])
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        # x = self.tanh(x9)
        x = F.tanh(x9)

        x = x.view(-1, 2, 640, 640)
        x = torch.permute(x, (0, 2, 3, 1))

        return x
