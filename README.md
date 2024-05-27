# 这是一个实现由INR图像生成RGB图像的神经网络，思路来源于知乎大佬的ai上色示例，链接: https://zhuanlan.zhihu.com/p/30493746
# 大佬的训练思路是由RGB图像转为LAB图像，通过A，B通道的色彩信息作为标签，来对L通道的明度图像进行训练，最后融合训练后的色彩信息与明度，实现对图像上色
这个思路恰好可以用来实现由红外图：INR ，来转化成普通色彩图像：RGB 的目的：本人研究方向为自动驾驶目标检测，此模块作为图像增强部分使用，实现自动驾驶汽车夜视识别功能
# 训练脚本如下：
!python train.py --data_dir='your data directory' --save_dir='your path to save the trained model' --epochs='number of epochs' --optimizer='your optimizer for instance:SGD AdamW'
# 其他训练参数可输入 opt--h 来查询
#本人在Windows下训练，相应脚本如下:
!python train.py --data="D:/code/datasets/bddn/images" --epochs=100 --loss=mse --batch_size=8
# tensorboard 可视化命令：
!bash tensorboard --log_dir='runs/tensorboard'
#使用默认摄像头进行预测
!python predict.py --data='0' --model=bast.pt
# coco数据集下载
!mkdir ../datasets/coco
!cd ../datasets/coco
!mkdir images
!cd images

!wget -c http://images.cocodataset.org/zips/train2017.zip
!wget -c http://images.cocodataset.org/zips/val2017.zip
!wget -c http://images.cocodataset.org/zips/test2017.zip

!unzip train2017.zip
!unzip val2017.zip
!unzip test2017.zip

!rm train2017.zip
!rm val2017.zip
!rm test2017.zip

!cd ../
!wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

!unzip annotations_trainval2017.zip

!rm annotations_trainval2017.zip

