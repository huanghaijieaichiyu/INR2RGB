import torch
import torch.nn as nn

from models.Conv2dSAME import Conv2dSame
from models.common import Conv, C2f, SimAM, CA, SPPELAN, RepNCSPELAN4, SPPF, Concat


## 模型体量太小的话，显卡占用会很低
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
        self.conv1 = nn.Sequential(
            Conv(1, 8),
            CA(8, 8),
            RepNCSPELAN4(8, 16, 16, 8),
            RepNCSPELAN4(16, 32, 32, 16),
            RepNCSPELAN4(32, 64, 64, 32),
            SPPELAN(64, 64, 23),
            Conv(64, 32),
            RepNCSPELAN4(32, 16, 16, 8),
            RepNCSPELAN4(16, 8, 8, 4),
            Conv(8, 2, act=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 2, 640, 640)
        x = torch.permute(x, (0, 2, 3, 1))

        return x


class ConvertV2(nn.Module):
    """
    this is a light net for V1
    """

    def __init__(self) -> None:
        super(ConvertV2, self).__init__()
        self.conv1 = nn.Sequential(
            Conv(1, 8),
            CA(8, 8),
            C2f(8, 16),
            C2f(16, 32),
            C2f(32, 64),
            SPPELAN(64, 64, 23),
            Conv(64, 32),
            C2f(32, 16),
            C2f(16, 8),
            Conv(8, 2, act=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 2, 640, 640)
        x = torch.permute(x, (0, 2, 3, 1))

        return x


class ConvertV3(nn.Module):
    """
    此版本尝试残差结构，需要concat与SPP结构
    """

    def __init__(self):
        super(ConvertV3, self).__init__()
        self.conv_in = Conv(1, 8, 3)
        self.c2f_1 = C2f(8, 16)
        self.c2f_2 = C2f(16, 32)
        self.sppf = SPPELAN(32, 32, 16)
        self.conv_down = Conv(48, 16, 3)
        self.concat = Concat()
        self.c2f_3 = C2f(16, 8)
        self.c2f_4 = C2f(40, 8)
        self.conv_out = Conv(8, 2, 3, act=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.c2f_1(x1)
        x3 = self.c2f_2(x2)
        x4 = self.sppf(x3)
        x5 = self.concat([x2, x4])
        x6 = self.conv_down(x5)
        x7 = self.c2f_3(x6)
        x7 = self.concat([x3, x7])
        x8 = self.c2f_4(x7)
        x9 = self.conv_out(x8)

        x = self.tanh(x9)
        x = x.view(-1, 2, 640, 640)
        x = torch.permute(x, (0, 2, 3, 1))

        return x
