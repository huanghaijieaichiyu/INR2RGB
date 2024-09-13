import os

import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics.functional import peak_signal_noise_ratio
from torchvision import transforms
from utils.misic import set_random_seed, get_opt, get_loss, ssim, model_structure, save_path
from torch.backends import cudnn
from torch.cuda.amp import autocast

from torch.utils import tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print

from datasets.data_set import MyDataset
from models.base_mode import Generator, Discriminator
from utils.color_trans import PSlab2rgb, PSrgb2lab


def train(args):
    # 避免同名覆盖
    if args.resume != '':
        path = os.path.dirname(os.path.dirname(args.resume))
    else:
        path = save_path(args.save_path)
        os.makedirs(os.path.join(path, 'generator'))
        os.makedirs(os.path.join(path, 'discriminator'))
    # 创建训练日志文件
    train_log = path + '/log.txt'
    train_log_txt_formatter = (
        '{time_str} \t [Epoch] \t {epoch:03d} \t [gLoss] \t {gloss_str} \t [dLoss] \t {dloss_str} \t {Dx_str} \t ['
        'Dgz0] \t {Dgz0_str} \t [Dgz1] \t {Dgz1_str}\n')

    args_dict = args.__dict__
    print(args_dict)

    # 训练前数据准备
    device = torch.device('cpu')
    if args.device == 'cuda':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    log = tensorboard.SummaryWriter(log_dir=os.path.join(args.save_path, 'tensorboard'),
                                    filename_suffix=str(args.epochs),
                                    flush_secs=180)
    set_random_seed(args.seed, deterministic=args.deterministic,
                    benchmark=args.benchmark)

    # 选择模型参数

    generator = Generator(args.depth, args.weight)
    discriminator = Discriminator(
        batch_size=args.batch_size, img_size=args.img_size[0])

    if args.draw_model:
        print('-' * 50)
        print('Drawing model graph to tensorboard, you can check it with:http://127.0.0.1:6006 in tensorboard '
              '--logdir={}'.format(os.path.join(args.save_path, 'tensorboard')))
        log.add_graph(generator, torch.randn(
            args.batch_size, 1, args.img_size[0], args.img_size[1]))
        print('Drawing doe!')
        print('-' * 50)
    print('Generator model info: \n')
    g_params, g_macs = model_structure(
        generator, img_size=(1, args.img_size[0], args.img_size[1]))
    print('Discriminator model info: \n')
    d_params, d_macs = model_structure(
        discriminator, img_size=(2, args.img_size[0], args.img_size[1]))
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
    train_data = MyDataset(args.data, img_size=args.img_size)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=True)
    assert len(train_loader) != 0, 'no data loaded'

    d_optimizer, g_optimizer = get_opt(args, generator, discriminator)

    g_loss = get_loss(args.loss)
    g_loss = g_loss.to(device)
    d_loss = nn.SmoothL1Loss()
    d_loss.to(device)

    # 此处开始训练
    # 使用cuDNN加速训练
    if args.cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True

    # 开始训练
    discriminator.train()
    generator.train()
    epoch = 0
    while epoch < args.epochs:
        # 参数储存
        Ssim = [0.]
        source_g = [0.]
        fake_tensor = torch.zeros(
            (args.batch_size, 3, args.img_size[0], args.img_size[1]))
        d_g_z2 = 0.
        d_output = 0
        g_output = 0
        # 储存loss 判断模型好坏
        gen_loss = []
        dis_loss = []
        # 断点训练参数设置
        if args.resume != '':
            # Loading the generator's checkpoints
            g_path_checkpoint = os.path.join(args.resume, 'generator')
            g_checkpoint = torch.load(g_path_checkpoint)  # 加载断点
            generator.load_state_dict(g_checkpoint['net'])
            g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            g_epoch = g_checkpoint['epoch']  # 设置开始的epoch
            g_loss.load_state_dict = g_checkpoint['loss']

            d_path_checkpoint = os.path.join(args.resume, 'discriminator')
            d_checkpoint = torch.load(d_path_checkpoint)  # 加载断点
            discriminator.load_state_dict(d_checkpoint['net'])
            d_optimizer.load_state_dict(d_checkpoint['optimizer'])
            d_loss.load_state_dict = d_checkpoint['loss']

            epoch = g_epoch
            print('继续第：{}轮训练'.format(epoch + 1))
            args.resume = ''  # 跳出循环
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

            with autocast(enabled=args.amp):
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()

                '''---------------训练判别模型---------------'''
                fake = generator(gray)
                fake_inputs = discriminator(fake.detach())
                real_inputs = discriminator(color / lamb)

                # 图像拼接还原
                fake_tensor = torch.zeros_like(img, dtype=torch.float32)
                fake_tensor[:, 0, :, :] = gray[:, 0, :, :]  # 主要切片位置
                fake_tensor[:, 1:, :, :] = lamb * fake

                fake_img = PSlab2rgb(fake_tensor)
                psn = peak_signal_noise_ratio(fake_img, img, data_range=255.)

                real_lable = torch.ones_like(fake_inputs, requires_grad=False)
                fake_lable = torch.zeros_like(fake_inputs, requires_grad=False)
                # D 希望 real_loss 为 1
                d_real_output = d_loss(real_inputs, real_lable)
                d_real_output.backward()
                d_x = real_inputs.mean().item()
                # D希望 fake_loss 为 0
                d_fake_output = d_loss(fake_inputs, fake_lable)
                d_fake_output.backward()
                d_g_z1 = fake_inputs.mean().item()
                d_output = (d_real_output.item() + d_fake_output.item()) / 2.
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10)
                d_optimizer.step()

                '''--------------- 训练生成器 ----------------'''
                fake_inputs = discriminator(fake)
                # G 希望 fake 为 1 加上 psn及 ssim相似损失
                g_output = (g_loss(fake_inputs, real_lable) +
                            g_loss((100 - psn) * 0.01, torch.ones_like(psn))) * 0.5
                g_output.backward()
                d_g_z2 = fake_inputs.mean().item()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
                g_optimizer.step()

            gen_loss.append(g_output.item())
            dis_loss.append(d_output)

            source_g.append(d_g_z2)
            pbar.set_description('||Epoch: [%d/%d]|--|--|Batch: [%d/%d]|--|--|Loss_D: %.4f|--|--|Loss_G: '
                                 '%.4f|--|--|D(x): %.4f|--|--|D(G(z)): %.4f / %.4f|'
                                 % (epoch + 1, args.epochs, target + 1, len(train_loader),
                                    d_output, g_output.item(), d_x, d_g_z1, d_g_z2))

            g_checkpoint = {
                'net': generator.state_dict(),
                'optimizer': g_optimizer.state_dict(),
                'epoch': epoch,
                'loss': g_loss.state_dict()
            }
            d_checkpoint = {
                'net': discriminator.state_dict(),
                'optimizer': d_optimizer.state_dict(),
                'epoch': epoch,
                'loss': d_loss.state_dict()
            }
            torch.save(g_checkpoint, path + '/generator/last.pt')
            torch.save(d_checkpoint, path + '/discriminator/last.pt')
        # eval model
        if (epoch + 1) % 10 == 0 and (epoch + 1) >= 10:
            print("Evaluating the generator model")
            ssim_source = ssim(fake_tensor, img)
            if ssim_source.item() > max(Ssim):
                torch.save(g_checkpoint, path + '/generator/best.pt')
                torch.save(d_checkpoint, path + '/discriminator/best.pt')
            Ssim.append(ssim_source.item())
            print("Model SSIM : ", ssim_source.item())

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
                                                      ["{:4f}".format(d_g_z2)])
                                                  )
        with open(train_log, "a") as f:
            f.write(to_write)
        # 可视化训练结果
        log.add_scalar('generation loss', np.mean(gen_loss), epoch + 1)
        log.add_scalar('discrimination loss', np.mean(dis_loss), epoch + 1)
        log.add_scalar('learning rate', g_optimizer.state_dict()
                       ['param_groups'][0]['lr'], epoch + 1)
        log.add_scalar('SSIM', np.mean(Ssim), epoch + 1)
        log.add_images('real', img, epoch + 1)
        log.add_images('fake', fake_img, epoch + 1)
        epoch += 1
    pbar.close()
    log.close()


def predict(self):
    # 防止同名覆盖
    path = save_path(self.save_path, model='predict')
    # 数据准备
    if self.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = Generator()
    model_structure(model, (1, self.img_size[0], self.img_size[1]))
    checkpoint = torch.load(self.model)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    test_data = MyDataset(self.data, self.img_size)
    img_pil = transforms.ToPILImage()
    test_loader = DataLoader(test_data,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             drop_last=True)
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), bar_format='{l_bar}{bar:10}| {n_fmt}/{'
                                                                           'total_fmt} {elapsed}')
    model.eval()
    torch.no_grad()
    i = 0
    if not os.path.exists(os.path.join(path, 'predictions')):
        os.makedirs(os.path.join(path, 'predictions'))
    for data in pbar:
        target, (img, label) = data

        img_lab = PSrgb2lab(img)
        gray, _, _ = torch.split(img_lab, [1, 1, 1], 1)

        lamb = 128.  # 取绝对值最大值，避免负数超出索引
        gray = gray.to(device)

        fake = model(gray)
        fake_tensor = torch.zeros(
            (self.batch_size, 3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
        fake_tensor[:, 0, :, :] = gray[:, 0, :, :]  # 主要切片位置
        fake_tensor[:, 1:, :, :] = lamb * fake
        for j in range(self.batch_size):

            fake_img = np.array(
                img_pil(PSlab2rgb(fake_tensor)[j]), dtype=np.float32)

            if i > 10 and i % 10 == 0:  # 图片太多，十轮保存一次
                img_save_path = os.path.join(
                    path, 'predictions', str(i) + '.jpg')
                cv2.imwrite(img_save_path, fake_img)
            i = i + 1
        pbar.set_description('Processed %d images' % i)
    pbar.close()


def predict_live(self):
    if self.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = Generator(1, 1)
    model_structure(model, (1, self.img_size[0], self.img_size[1]))
    checkpoint = torch.load(self.model)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    cap = cv2.VideoCapture(2)  # 读取图像
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    write = cv2.VideoWriter()
    write.open(self.save_path + '/fake.mp4', fourcc=fourcc, fps=cap.get(cv2.CAP_PROP_FPS), isColor=True,
               frameSize=[640, 480])

    model.eval()
    torch.no_grad()
    if not os.path.exists(os.path.join(self.save_path, 'predictions')):
        os.makedirs(os.path.join(self.save_path, 'predictions'))
    while cap.isOpened():
        _, frame = cap.read()
        _, frame = cap.read()
        frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 是否需要resize取决于新图片格式与训练时的是否一致
        frame_pil = cv2.resize(frame_pil, self.img_size)
        # 是否需要resize取决于新图片格式与训练时的是否一致
        frame_pil = cv2.resize(frame_pil, self.img_size)

        frame_pil = torch.tensor(np.array(
            frame_pil, np.float32) / 255., dtype=torch.float32).to(device)  # 转为tensor
        frame_pil = torch.unsqueeze(frame_pil, 0).permute(
            0, 3, 1, 2)  # 提升维度--转换维度
        frame_lab = PSrgb2lab(frame_pil)  # 转为LAB
        gray, _, _ = torch.split(frame_lab, [1, 1, 1], 1)

        fake_ab = model(gray)
        fake_ab = fake_ab.permute(0, 2, 3, 1).detach().cpu().numpy()[
            0].astype(np.float32)
        fake = np.zeros(
            (self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        fake[:, :, 0] = gray.permute(
            0, 2, 3, 1).detach().cpu().numpy()[0][:, :, 0]
        fake[:, :, 1:] = fake_ab * 128
        fake = cv2.cvtColor(fake, cv2.COLOR_Lab2BGR)
        # fake *= 255.
        fake = cv2.resize(fake, (640, 480))  # 维度还没降下来
        cv2.imshow('fake', fake)

        cv2.imshow('gray', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # 写入文件
        # float转uint8 fake[0,1]转[0,255]
        fake = np.array(255 * fake, dtype=np.uint8)
        write.write(fake)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cap.release()
    write.release()
