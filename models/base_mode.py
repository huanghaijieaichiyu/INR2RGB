import torch
import torch.nn as nn

from models.Conv2dSAME import Conv2dSame
from models.common import Conv, C2f, SimAM, CA, SPPELAN


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
        #  print(x)
        x = x.reshape(1, 640, 640, 2)
        return x


class ConvertV1(nn.Module):
    def __init__(self, in_ch, out_ch, e=0.5):
        super(ConvertV1, self).__init__()
        c_dnow = in_ch * e  # 下采样
        c_up = in_ch * (1 - e)  # 上采样
        self.conv1 = Conv(in_ch, 8 * in_ch)  # 1x8
        self.conv2 = C2f(in_ch, c_up)  # 8x16
        self.conv3 = C2f(in_ch, c_dnow)  # 16x8
        self.conv4 = Conv(in_ch, out_ch, act=False)  # 8x out_ch 此层激活函数需替换为tanh，从而将预测值映射到[-1,1]
        self.tanh = nn.Tanh()
        self.att = CA(out_ch, out_ch)
        self.upsample = nn.Upsample(scale_factor=2)
        self.sppe = SPPELAN(in_ch, in_ch, in_ch)

    def forward(self, x):
        x1 = self.conv1(x)  # input *8
        x2 = self.att(x1)
        x3 = self.conv2(x2)  # 16
        x4 = self.conv2(x3)  # 32
        x5 = self.upsample(x4)
        x6 = self.conv2(x5)  # 64
        x7 = self.upsample(x6)
        x8 = self.conv2(x7)  # 128
        # 开始下采样
        x9 = self.conv3(x8)  # 64
        x10 = self.conv3(x9)  # 32
        x11 = self.conv3(x10)  # 16
        x12 = self.conv3(x11)  # 8
        x13 = self.sppe(x6, x7, x8, x9, x10, x11)
        x14 = torch.cat([x12, x13], dim=1)
        x15 = self.conv4(x14)
        output = self.tanh(x15)

        return output
