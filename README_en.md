# INR2RGB: Grayscale Image Colorization

This project implements a deep learning approach to colorize grayscale images. It uses a modified DCGAN (Deep Convolutional Generative Adversarial Network) architecture with self-attention, cross-layer connections, and residual blocks. The model is trained on RGB images converted to the LAB color space, using the A and B channels as color information labels.

## Overview

The project consists of the following key components:

*   **`engine.py`**: Contains the core training and prediction logic.
*   **`train.py`**: Implements the training script with argument parsing.
*   **`predict.py`**: Implements the prediction script for image colorization and live video colorization.
*   **`models/`**: Contains the model definitions:
    *   `base_mode.py`: Defines the Generator, Discriminator, and Critic models.
    *   `Repvit.py`: Defines the RepViTBlock.
    *   `common.py`: Defines common layers and blocks.
*   **`datasets/`**: Contains the dataset loading and preprocessing logic:
    *   `data_set.py`: Implements the `LowLightDataset` class for loading the LOLdataset.
*   **`utils/`**: Contains utility functions:
    *   `color_trans.py`: Contains color space conversion functions (RGB to LAB and LAB to RGB).
    *   `loss.py`: Defines custom loss functions.
    *   `misic.py`: Contains helper functions for training.

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd INR2RGB
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file includes the following packages:

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

## Usage

### Training

To train the model, run the `train.py` script:

```bash
python train.py --data <path_to_dataset> --epochs <number_of_epochs> --batch_size <batch_size>
```

*   `--data`: Path to the dataset directory (e.g., `../datasets/LOLdataset`).
*   `--epochs`: Number of training epochs (default: 500).
*   `--batch_size`: Batch size (default: 16).
*   `--img_size`: Size of the input images (default: (256, 256)).
*   `--optimizer`: Optimizer to use (default: Adam).
*   `--num_workers`: Number of data loading workers (default: 0). Set to 0 in Windows.
*   `--seed`: Random seed for reproducibility (default: random integer).
*   `--resume`: Path to a checkpoint for resuming training.
*   `--amp`: Whether to use Automatic Mixed Precision (AMP) for training.
*   `--cuDNN`: Whether to use cuDNN for accelerated training.
*   `--loss`: Loss function to use (default: bce).
*   `--lr`: Learning rate (default: 3.5e-4).
*   `--momentum`: Momentum for SGD optimizer (default: 0.5).
*   `--depth`: Depth of the generator (default: 1).
*   `--weight`: Weight of the generator (default: 1).
*   `--device`: Device to use for training (default: cuda).
*   `--save_path`: Directory to save training results (default: runs/).
*   `--benchmark`: Whether to use `torch.benchmark` to accelerate training.
*   `--deterministic`: Whether to use deterministic initialization.
*   `--draw_model`: Whether to draw the model graph to TensorBoard.

Example:

```bash
python train.py --data ../datasets/LOLdataset --epochs 200 --batch_size 32
```

### Prediction

To colorize images using a trained model, run the `predict.py` script:

```bash
python predict.py --data <path_to_image_or_directory> --model <path_to_generator_checkpoint>
```

*   `--data`: Path to the image or directory containing images to colorize. Use `0` to open your camera for live colorization.
*   `--model`: Path to the generator checkpoint file (e.g., `runs/train(3)/generator/last.pt`).
*   `--batch_size`: Batch size (default: 16).
*   `--img_size`: Size of the input images (default: (256, 256)).
*   `--num_workers`: Number of data loading workers (default: 0).
*   `--device`: Device to use for prediction (default: cuda).
*   `--save_path`: Directory to save the colorized images (default: runs/).

Example:

```bash
python predict.py --data test_image.png --model runs/train(3)/generator/last.pt
```

To use your camera for live colorization:

```bash
python predict.py --data 0 --model runs/train(3)/generator/last.pt
```

### Dataset

The project uses the LOLdataset, which contains pairs of low-light and normal-light images. The `LowLightDataset` class in `datasets/data_set.py` handles loading and preprocessing the data.

The dataset directory should have the following structure:

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

### Models

The core of the colorization process is the generator model, defined in `models/base_mode.py`. It uses RepViT blocks, SPPELAN, PSA, and other custom layers to generate colorized images from grayscale inputs. The discriminator (or critic in the WGAN version) is used to distinguish between real and generated images, guiding the generator to produce more realistic results.

## Training with WGAN

The `engine.py` file also includes a `train_WGAN` function for training the model using the WGAN-GP (Wasserstein GAN with Gradient Penalty) approach. This can potentially lead to more stable training and better results. To use this, you would need to modify the `train.py` script to call `train_WGAN` instead of `train`.

## License

[Check for a `LICENSE` file]
