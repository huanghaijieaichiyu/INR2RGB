import torch


def cal_gp(D, real_imgs, fake_imgs, cuda):  # 定义函数，计算梯度惩罚项gp
    # 真假样本的采样比例r，batch size个随机数，服从区间[0,1)的均匀分布
    r = torch.rand(size=(real_imgs.shape[0], 1, 1, 1))
    if cuda:  # 如果使用cuda
        r = r.cuda()  # r加载到GPU
    # 输入样本x，由真假样本按照比例产生，需要计算梯度
    x = (r * real_imgs + (1 - r) * fake_imgs).requires_grad_(True)
    d = D(x)  # 判别网络D对输入样本x的判别结果D(x)
    fake = torch.ones_like(d)  # 定义与d形状相同的张量，代表梯度计算时每一个元素的权重
    if cuda:  # 如果使用cuda
        fake = fake.cuda()  # fake加载到GPU
    g = torch.autograd.grad(  # 进行梯度计算
        outputs=d,  # 计算梯度的函数d，即D(x)
        inputs=x,  # 计算梯度的变量x
        grad_outputs=fake,  # 梯度计算权重
        create_graph=True,  # 创建计算图
        retain_graph=True  # 保留计算图
    )[0]  # 返回元组的第一个元素为梯度计算结果
    gp = ((g.norm(2, dim=1) - 1) ** 2).mean()  # (||grad(D(x))||2-1)^2 的均值
    return gp  # 返回梯度惩罚项gp
