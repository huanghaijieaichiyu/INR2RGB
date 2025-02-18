import os


# 批量重命名文件
def rename_files(data_dir):
    """
    批量重命名文件。
    Args:
        data_dir (str): 数据集目录。
        img_format (str): 图像格式。
    """
    # 获取数据集中的图片后缀名
    sample_image_path = next(os.path.join(root, file) for root, _, files in os.walk(
        data_dir) for file in files if file.endswith(('png', 'jpg', 'jpeg', 'bmp')))
    img_format = sample_image_path.split('.')[-1]
    for root, _, files in os.walk(data_dir):
        for i, file in enumerate(files):
            os.rename(os.path.join(root, file), os.path.join(
                root, f"{i}.{img_format}"))


if __name__ == '__main__':
    # 1. 设置数据集目录
    data_dir = "../datasets/kitti_LOL"  # 替换成你的数据集路径
    # 2. 重命名文件
    rename_files(data_dir)
    print("重命名完成！")
    # ...
