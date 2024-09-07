import os

import time

import numpy as np
import torch
from torcheval.metrics.functional import peak_signal_noise_ratio

from utils.misic import learning_rate_scheduler, set_random_seed, get_opt, get_loss, ssim, model_structure, save_path
from torch.backends import cudnn
from torch.cuda.amp import autocast

from torch.utils import tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print

from datasets.data_set import MyDataset
from models.base_mode import Generator, Discriminator
from utils.color_trans import PSlab2rgb, PSrgb2lab


def train(self):
    # 避免同名覆盖
    if self.resume != ['']:
        path = os.path.dirname(os.path.dirname(self.resume[0]))
    else:
        path = save_path(self.save_path)
        os.makedirs(os.path.join(path, 'generator'))
        os.makedirs(os.path.join(path, 'discriminator'))
    # 创建训练日志文件
    train_log = path + '/log.txt'
    train_log_txt_formatter = (
        '{time_str} \t [Epoch] \t {epoch:03d} \t [gLoss] \t {gloss_str} \t [dLoss] \t {dloss_str} \t {Dx_str} \t ['
        'Dgz0] \t {Dgz0_str} \t [Dgz1] \t {Dgz1_str}\n')

    args_dict = self.__dict__
    print(args_dict)

    # 训练前数据准备
    device = torch.device('cpu')
    if self.device == 'cuda':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    log = tensorboard.SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard'),
                                    filename_suffix=str(self.epochs),
                                    flush_secs=180)
    set_random_seed(self.seed, deterministic=self.deterministic,
                    benchmark=self.benchmark)

    # 选择模型参数

    generator = Generator(self.depth, self.weight)
    discriminator = Discriminator(
        batch_size=self.batch_size, img_size=self.img_size[0])

    if self.draw_model:
        print('-' * 50)
        print('Drawing model graph to tensorboard, you can check it with:http://127.0.0.1:6006 in tensorboard '
              '--logdir={}'.format(os.path.join(self.save_path, 'tensorboard')))
        log.add_graph(generator, torch.randn(
            self.batch_size, 1, self.img_size[0], self.img_size[1]))
        print('Drawing doe!')
        print('-' * 50)
    print('Generator model info: \n')
    g_params, g_macs = model_structure(
        generator, img_size=(1, self.img_size[0], self.img_size[1]))
    print('Discriminator model info: \n')
    d_params, d_macs = model_structure(
        discriminator, img_size=(2, self.img_size[0], self.img_size[1]))
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    # 打印配置
    with open(path + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
        f.writelines('\n' + 'The parameters of generator: {:.2f} M'.format(g_params) + '\n' + 'The Gflops of '
                                                                                              'generator: {:.2f}'
                                                                                              ' G'.format(g_macs))
        f.writelines('\n' + 'The parameters of discriminator: {:.2f} M'.format(d_params) + '\n' + 'The Gflops of '
                                                                                                  ' discriminator: {:.2f}'
                                                                                                  ' G'.format(d_macs))
        f.writelines('\n' + '-------------------------------------------')
    print('train models at the %s device' % device)
    os.makedirs(path, exist_ok=True)

    # 加载数据集
    train_data = MyDataset(self.data, img_size=self.img_size)

    train_loader = DataLoader(train_data,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              shuffle=True,
                              drop_last=True)
    assert len(train_loader) != 0, 'no data loaded'

    d_optimizer, g_optimizer = get_opt(self, generator, discriminator)

    # 学习率退火

    LR_D, LR_G = learning_rate_scheduler(self, d_optimizer, g_optimizer)

    loss = get_loss(self)
    loss = loss.to(device)

    # 此处开始训练
    # 使用cuDNN加速训练
    if self.cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True

    # 开始训练
    discriminator.train()
    generator.train()
    epoch = 0
    while epoch < self.epochs:
        # 参数储存
        Ssim = [0.]
        source_g = [0.]
        fake_tensor = torch.zeros(
            (self.batch_size, 3, self.img_size[0], self.img_size[1]))
        d_g_z2 = 0.
        d_output = 0
        g_output = 0
        # 储存loss 判断模型好坏
        loss_all = [99.]
        gen_loss = []
        dis_loss = []
        # 断点训练参数设置
        if self.resume != ['']:
            g_path_checkpoint = self.resume[0]
            g_checkpoint = torch.load(g_path_checkpoint)  # 加载断点
            generator.load_state_dict(g_checkpoint['net'])
            g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            g_epoch = g_checkpoint['epoch']  # 设置开始的epoch
            loss.load_state_dict = g_checkpoint['loss']
            epoch = g_epoch
            print('继续第：{}轮训练'.format(epoch + 1))
            self.resume = ['']  # 跳出循环
        print('第{}轮训练'.format(epoch + 1))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}', colour='#8762A5')
        for data in pbar:
            target, (img, _) = data
            # 对输入图像进行处理
            img_lab = PSrgb2lab(img)
            gray, a, b = torch.split(img_lab, 1, 1)
            color = torch.cat([a, b], dim=1)
            # lamb = color.abs().max()  # 取绝对值最大值，避免负数超出索引
            lamb = 128.
            gray = gray.to(device)
            color = color.to(device)

            with autocast(enabled=self.amp):
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()

                '''---------------训练判别模型---------------'''
                fake = generator(gray)
                fake_inputs = discriminator(fake.detach())
                real_outputs = discriminator(color / lamb)

                # 图像拼接还原
                fake_tensor = torch.zeros_like(img, dtype=torch.float32)
                fake_tensor[:, 0, :, :] = gray[:, 0, :, :]  # 主要切片位置
                fake_tensor[:, 1:, :, :] = lamb * fake

                fake_img = PSlab2rgb(fake_tensor)
                psn = peak_signal_noise_ratio(fake_img, img, data_range=255.)

                real_lable = torch.ones_like(fake_inputs, requires_grad=False)
                fake_lable = torch.zeros_like(fake_inputs, requires_grad=False)
                # D 希望 real_loss 为 1
                d_real_output = loss(real_outputs, real_lable)
                d_real_output.backward()
                d_x = real_outputs.mean().item()
                # D希望 fake_loss 为 0
                d_fake_output = loss(fake_inputs, fake_lable)
                d_fake_output.backward()
                d_g_z1 = fake_inputs.mean().item()
                d_output = (d_real_output.item() + d_fake_output.item()) / 2.
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10)
                d_optimizer.step()

                '''--------------- 训练生成器 ----------------'''
                fake_inputs = discriminator(fake)
                # G 希望 fake 为 1 加上 psn及 ssim相似损失
                g_output = (loss(fake_inputs, real_lable) + loss((100 - psn) * 0.01, torch.ones_like(psn))) * 0.5
                g_output.backward()
                d_g_z2 = fake_inputs.mean().item()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 20)
                g_optimizer.step()

            gen_loss.append(g_output.item())
            dis_loss.append(d_output)

            source_g.append(d_g_z2)
            pbar.set_description('||Epoch: [%d/%d]|--|--|Batch: [%d/%d]|--|--|Loss_D: %.4f|--|--|Loss_G: '
                                 '%.4f|--|--|D(x): %.4f|--|--|D(G(z)): %.4f / %.4f|'
                                 % (epoch + 1, self.epochs, target + 1, len(train_loader),
                                    d_output, g_output.item(), d_x, d_g_z1, d_g_z2))

            g_checkpoint = {
                'net': generator.state_dict(),
                'optimizer': g_optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss.state_dict() if loss is not None else None
            }
            d_checkpoint = {
                'net': discriminator.state_dict(),
                'optimizer': d_optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss.state_dict() if loss is not None else None
            }
            torch.save(g_checkpoint, path + '/generator/last.pt')
            torch.save(d_checkpoint, path + '/discriminator/last.pt')
        # eval model
        if (epoch+1) % 50 == 0 & (epoch+1) >= 50:
            print("Evaling the generator model")
            generator.eval()
            ssim_source = ssim(fake_tensor, img)
            if ssim_source > max(Ssim):
                torch.save(g_checkpoint, path + '/generator/best.pt')
                torch.save(d_checkpoint, path + '/discriminator/best.pt')
            Ssim.append(ssim_source)
            print("Model SSIM : %.4f", ssim_source)
            generator.train()

        # 判断模型是否提前终止
        if torch.eq(fake_tensor, torch.zeros_like(fake_tensor)).all():
            print('fake tensor is zero!')
            break
        if d_g_z2 <= 1e-5:
            break

        # 学习率退火
        if self.lr_deduce == 'no':
            pass
        elif self.lr_deduce == 'reduceLR':
            assert LR_D is not None, 'no such lr deduce'
            assert LR_G is not None, 'no such lr deduce'
            LR_D.step(d_output)
            LR_G.step(g_output)
        elif self.lr_deduce == 'coslr' or 'lamb':
            assert LR_D is not None, 'no such lr deduce'
            assert LR_G is not None, 'no such lr deduce'
            LR_D.step()
            LR_G.step()


        # 写入日志文件
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch + 1,
                                                  gloss_str=" ".join(
                                                      ["{:4f}".format(np.mean(gen_loss))]),
                                                  dloss_str=" ".join(
                                                      ["{:4f}".format(np.mean(dis_loss))]),
                                                  Dx_str=" ".join(
                                                      ["{:4f}".format(d_x)]),
                                                  Dgz0_str=" ".join(
                                                      ["{:4f}".format(d_g_z1)]),
                                                  Dgz1_str=" ".join(
                                                      ["{:4f}".format(d_g_z2)]),
                                                  PSN_str=" ".join(["{:4f}".format(np.mean(Ssim))]))
        with open(train_log, "a") as f:
            f.write(to_write)
        # 可视化训练结果

        log.add_scalar('generation loss', np.mean(gen_loss), epoch + 1)
        log.add_scalar('discrimination loss', np.mean(dis_loss), epoch + 1)
        log.add_scalar('PSN', np.mean(Ssim), epoch + 1)
        log.add_scalar('learning rate', g_optimizer.state_dict()
        ['param_groups'][0]['lr'], epoch + 1)

        log.add_images('real', img, epoch + 1)
        log.add_images('fake', fake_img, epoch + 1)
        epoch += 1
        pbar.close()
    log.close()
