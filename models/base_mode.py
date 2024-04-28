import torch.nn as nn

from models.common import Conv, C2f, SPPELAN, Concat, EMA
from utils.model_map import model_structure


class Generator(nn.Module):

    def __init__(self, depth=0.8, weight=1) -> None:
        super(Generator, self).__init__()
        depth = depth
        weight = weight
        self.conv1 = Conv(1, int(8 * depth))
        self.conv2 = nn.Sequential(Conv(int(8 * depth), int(16 * depth), int(3 * weight), 2),
                                   C2f(int(16 * depth), int(32 * depth), int(weight), shortcut=True)
                                   )
        self.conv3 = nn.Sequential(Conv(int(32 * depth), int(64 * depth), int(3 * weight), 2),
                                   C2f(int(64 * depth), int(128 * depth), int(weight), shortcut=True)
                                   )
        self.conv4 = nn.Sequential(Conv(int(128 * depth), int(256 * depth), int(3 * weight), 2),
                                   C2f(int(256 * depth), int(512 * depth), int(weight), shortcut=True),
                                   Conv(int(512 * depth), int(1024 * depth), int(3 * weight)),
                                   Conv(int(1024 * depth), int(512 * depth)),
                                   EMA(int(512 * depth)),
                                   )
        self.conv5 = nn.Sequential(SPPELAN(int(512 * depth), int(512 * depth), int(256 * depth)),
                                   Conv(int(512 * depth), int(256 * depth), int(3 * weight)))
        self.conv6 = nn.Sequential(C2f(int(256 * depth), int(128 * depth), int(weight), shortcut=False),
                                   nn.Upsample(scale_factor=2),
                                   Conv(int(128 * depth), int(64 * depth)))
        self.conv7 = nn.Sequential(C2f(int(192 * depth), int(96 * depth), int(weight), shortcut=True),
                                   nn.Upsample(scale_factor=2),
                                   Conv(int(96 * depth), int(64 * depth)))
        self.conv8 = nn.Sequential(C2f(int(64 * depth), int(64 * depth), int(weight), shortcut=True),
                                   nn.Upsample(scale_factor=2),
                                   Conv(int(64 * depth), int(128 * depth))
                                   )
        self.conv9 = nn.Sequential(C2f(int(128 * depth), int(64 * depth), int(weight), shortcut=False),
                                   Conv(int(64 * depth), int(32 * depth))
                                   )
        self.conv10 = nn.Sequential(C2f(int(32 * depth), int(16 * depth), int(weight), shortcut=False),
                                    Conv(int(16 * depth), int(8 * depth)),
                                    Conv(int(8 * depth), 2, int(3 * weight), act=False)
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
        x9 = self.conv8(x7)
        x10 = self.conv9(x9)
        x11 = self.conv10(x10)
        x12 = self.tanh(x11)

        x = x12.view(-1, 2, x.shape[2], x.shape[3])

        return x


class Discriminator(nn.Module):
    """
    Discriminator model with no activation function
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_in = nn.Sequential(Conv(2, 16, 3)
                                     )
        self.conv1 = nn.Sequential(C2f(16, 32),  # 256
                                   Conv(32, 64, 3, 2),  # 128
                                   Conv(64, 128, 3, 2),  # 64
                                   Conv(128, 64, 3, 2),  # 32
                                   Conv(64, 32, 3, 2),  # 16
                                   Conv(32, 16, 3),
                                   C2f(16, 8),
                                   Conv(8, 4, 3),
                                   )
        self.conv_out = Conv(4, 1, 3)
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(16 * 16 * 1, 8 * 8),
                                    nn.LeakyReLU(),
                                    nn.Linear(8 * 8, 16),
                                    nn.LeakyReLU(),
                                    nn.Linear(16, 1)
                                    )
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.conv1(x1)
        x3 = self.conv_out(x2)
        x4 = self.flatten(x3)
        x5 = self.linear(x4)
        x = self.act(x5)

        return x


if __name__ == '__main__':
    model = Discriminator()
    # model_ = Generator()
    d_params, d_macs = model_structure(model, (2, 256, 256))
    #  d_params, d_macs = model_structure(model_, (1, 256, 256))
    print(d_params, d_macs)
