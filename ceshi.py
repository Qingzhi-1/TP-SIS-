import cv2
import numpy as np

# ⚠️ 替换成你 2017 训练集里的一张 label 图片路径
path = "/root/datasets/endovis2017/train/annotations/seq_1_frame000.bmp" 
# (或者随便找一张存在的 bmp)

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
print(f"包含的像素值: {np.unique(img)}")