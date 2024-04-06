import torch


def F(X):  # X为任意形状的张量
    FX = 7.787 * X + 0.137931
    index = X > 0.008856
    FX[index] = torch.pow(X[index], 1.0 / 3.0)
    return FX


def anti_F(X):  # 逆操作。
    tFX = (X - 0.137931) / 7.787
    index = X > 0.206893
    tFX[index] = torch.pow(X[index], 3)
    return tFX

def gamma(r):
    r2 = r / 12.92
    index = r > 0.04045  # pow:0.0031308072830676845,/12.92:0.0031308049535603713
    r2[index] = torch.pow((r[index] + 0.055) / 1.055, 2.4)
    return r2


def anti_g(r):
    r2 = r * 12.92
    index = r > 0.0031308072830676845
    r2[index] = torch.pow(r[index], 1.0 / 2.4) * 1.055 - 0.055
    return r2


# Mps = np.array([[0.436052025, 0.385081593, 0.143087414],
#               [0.222491598, 0.716886060, 0.060621486],
#               [0.013929122, 0.097097002, 0.714185470]])
# Mpst = np.linalg.inv(Mps)
# [[ 3.13405134 -1.61702771 -0.49065221]
#  [-0.97876273  1.91614223  0.03344963]
#  [ 0.07194258 -0.22897118  1.40521831]]

def myPSrgb2lab(img):  # RGB img:[b,3,h,w]->lab,L[0,100],AB[-127,127]
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]

    r = gamma(r)
    g = gamma(g)
    b = gamma(b)

    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470
    X = X / 0.964221
    Z = Z / 0.825211

    F_X = F(X)
    F_Y = F(Y)
    F_Z = F(Z)

    # L = 903.3*Y
    # index = Y > 0.008856
    # L[index] = 116 * F_Y[index] - 16 # [0,100]
    L = 116 * F_Y - 16.0
    a = 500 * (F_X - F_Y)  # [-127,127]
    b = 200 * (F_Y - F_Z)  # [-127,127]

    # L = L
    # a = (a+128.0)
    # b = (b+128.0)
    return torch.stack([L, a, b], dim=1)


def myPSlab2rgb(Lab):
    fY = (Lab[:, 0, :, :] + 16.0) / 116.0
    fX = Lab[:, 1, :, :] / 500.0 + fY
    fZ = fY - Lab[:, 2, :, :] / 200.0

    x = anti_F(fX)
    y = anti_F(fY)
    z = anti_F(fZ)
    x = x * 0.964221
    z = z * 0.825211
    #
    r = 3.13405134 * x - 1.61702771 * y - 0.49065221 * z
    g = -0.97876273 * x + 1.91614223 * y + 0.03344963 * z
    b = 0.07194258 * x - 0.22897118 * y + 1.40521831 * z
    #
    r = anti_g(r)
    g = anti_g(g)
    b = anti_g(b)
    return torch.stack([r, g, b], dim=1)

