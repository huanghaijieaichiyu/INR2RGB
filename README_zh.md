# INR2RGB：基于深度学习的灰度图像着色

本项目实现了一种用于灰度图像着色的创新深度学习方法，基于改进的 DCGAN 架构，并引入了 Vision Transformer (ViT) 结构。该模型在 LAB 颜色空间进行训练，能够生成高质量的彩色图像。

## 核心特性

本项目具有以下突出特点：

1.  **Deep-Transformer-GAN:** 将 Vision Transformer (ViT) 架构引入 GAN 网络中，增强了模型对全局信息的捕捉能力。
2.  **训练方式改进:** 采用梯度同步更新策略，显著提高了训练的准确性和鲁棒性。
3.  **创新性神经网络结构设计:** 结合了 RepViT 块、SPPELAN、PSA 和其他自定义层，实现了高效的特征提取和图像生成。

## 安装

1.  克隆存储库：

    ```bash
    git clone <repository_url>
    cd INR2RGB
    ```

2.  安装所需的依赖项：

    ```bash
    pip install -r requirements.txt
    ```

    `requirements.txt` 文件包含以下软件包：

    ```
    torch
    opencv-python<=4.9.0.80
    numpy<=1.26.3
    timm<=0.9.16
    torchvision
    tqdm<=4.65.0
    ptflops<= 0.7.2.2
    torcheval<=0.70
    rich~=13.7.1
    cython
    PyYAML~=6.0.1
    ```

## 用法

### 训练

使用 `train.py` 脚本训练模型：

```bash
python train.py --data <数据集路径> --epochs <训练轮数> --batch_size <批次大小>
```

*   `--data`: 数据集目录的路径 (例如, `../datasets/LOLdataset`).
*   `--epochs`: 训练轮数 (默认: 500).
*   `--batch_size`: 批次大小 (默认: 16).
*   `--img_size`: 输入图像的大小 (默认: (256, 256)).
*   `--optimizer`: 优化器 (默认: Adam).
*   `--num_workers`: 数据加载器的工作线程数 (默认: 0). Windows 中必须设置为 0.
*   `--seed`: 随机种子 (默认: 随机整数).
*   `--resume`: 恢复训练的检查点路径.
*   `--amp`: 是否使用自动混合精度 (AMP).
*   `--cuDNN`: 是否使用 cuDNN 加速.
*   `--loss`: 损失函数 (默认: bce).
*   `--lr`: 学习率 (默认: 3.5e-4).
*   `--momentum`: SGD 优化器的动量 (默认: 0.5).
*   `--depth`: 生成器的深度 (默认: 1).
*   `--weight`: 生成器的权重 (默认: 1).
*   `--device`: 训练设备 (默认: cuda).
*   `--save_path`: 保存训练结果的目录 (默认: runs/).
*   `--benchmark`: 是否使用 `torch.benchmark` 加速.
*   `--deterministic`: 是否使用确定性初始化.
*   `--draw_model`: 是否绘制模型图到 TensorBoard.

示例：

```bash
python train.py --data ../datasets/LOLdataset --epochs 200 --batch_size 32
```

### 预测

使用 `predict.py` 脚本对图像进行着色：

```bash
python predict.py --data <图像或目录路径> --model <生成器检查点路径>
```

*   `--data`: 要着色的图像或目录路径。使用 `0` 打开相机进行实时着色.
*   `--model`: 生成器检查点路径 (例如, `runs/train(3)/generator/last.pt`).
*   `--batch_size`: 批次大小 (默认: 16).
*   `--img_size`: 输入图像大小 (默认: (256, 256)).
*   `--num_workers`: 数据加载器的工作线程数 (默认: 0).
*   `--device`: 预测设备 (默认: cuda).
*   `--save_path`: 保存着色图像的目录 (默认: runs/).

示例：

```bash
python predict.py --data test_image.png --model runs/train(3)/generator/last.pt
```

实时着色：

```bash
python predict.py --data 0 --model runs/train(3)/generator/last.pt
```

### 模型评估
目前没有直接的评估脚本，评估依赖训练过程中的指标。后续可以添加一个单独的`evaluate.py`脚本，用于计算PSNR, SSIM等指标。

## 着色效果

您可以通过比较 `fake.png` (模型生成的图像) 和 `real.png` (真实图像) 来查看模型的着色效果。这两张图片位于项目的根目录下。

## 模型

着色模型的核心是 `models/base_mode.py` 中定义的生成器。它结合了 RepViT 块、SPPELAN、PSA 和其他自定义层，从灰度输入生成彩色图像。判别器（或 WGAN 中的评论家）用于区分真实图像和生成图像，指导生成器产生更逼真的结果。

## 数据集

本项目使用 LOLdataset，包含低光照和正常光照图像对。`datasets/data_set.py` 中的 `LowLightDataset` 类负责加载和预处理数据。

数据集目录结构：

```
LOLdataset/
    our485/
        high/
            *.png
        low/
            *.png
    eval15/
        high/
            *.png
        low/
            *.png
```

## 算法对比

| 算法       | 特点                                                         | PSNR  | SSIM  | 备注                                                         |
| ---------- | ------------------------------------------------------------ | ----- | ----- | ------------------------------------------------------------ |
| DCGAN      | 传统的深度卷积生成对抗网络                                   |  -    |  -    | 基准模型，性能通常较低                                       |
| INR2RGB (本项目) | 引入 ViT, 梯度同步更新, 创新网络结构                         |  -    |  -    | 性能预计显著优于 DCGAN。具体数值需通过实验测定。 |
| Pix2Pix    | 基于条件 GAN 的图像翻译模型                                 |  -    |  -    |  一种常用的图像着色方法。                                    |
| CycleGAN   | 无需配对数据进行图像翻译                                     |  -    |  -    |  适用于无配对数据集的着色。                                  |

**注意:**  PSNR 和 SSIM 值目前为空，需要在训练和评估后填入实际数值。

## 使用 WGAN 进行训练

`engine.py` 包含 `train_WGAN` 函数，支持 WGAN-GP 训练。您需要修改 `train.py` 脚本以调用 `train_WGAN`。

## 许可证

[查看 `LICENSE` 文件]
