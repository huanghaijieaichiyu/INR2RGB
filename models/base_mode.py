import torch
import torch.nn as nn

from models.Conv2dSAME import Conv2dSame
from models.common import Conv, C2f, SimAM, CA, SPPELAN, RepNCSPELAN4


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

    def __init__(self):
        super(ConvertV1, self).__init__()
        self.conv1 = nn.Sequential(
            Conv(1,8),
            CA(8,8),
            RepNCSPELAN4(8,16,16,8),
            RepNCSPELAN4(16,32,32,16),
            RepNCSPELAN4(32,64,64,32),
            SPPELAN(64,64,23),
            Conv(64,32),
            RepNCSPELAN4(32,16,16,8),
            RepNCSPELAN4(16,8,8,4),
            Conv(8,2),
            nn.Tanh()
                                )




    def forward(self, x):
        x = self.conv1(x)
        x = x.contiguous().view(-1,640,640,2)

        return x



class ConvertV2(nn.Module):


    """
    this is a light net for V1
    """
    def __init__(self) -> None:
        super(ConvertV2, self).__init__()
        self.conv1 = nn.Sequential(
            Conv(1,8),
            CA(8,8),
            C2f(8,16),
            C2f(16,32),
            C2f(32,64),
            nn.MaxPool2d(4,2,1),
            SPPELAN(64,64,23),
            Conv(64,32),
            C2f(32,16),
            nn.Maxpool2d(4,2,1),
            C2f(16,8),
            Conv(8,2),
            nn.Tanh()
                                )




    def forward(self, x):
        x = self.conv1(x)
        x = x.contiguous().view(-1,640,640,2)

        return x