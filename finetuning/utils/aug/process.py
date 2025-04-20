
from json.tool import main
from random import random

from .ops import blur,fliph,flipv,noise,rotate
import cv2
from skimage import transform
import torch
import numpy as np

import torchvision.transforms as transforms

class DataAug():
    def __init__(self):
        self.process_blur=blur.Blur(2.0) #高斯模糊
        self.process_FlipH=fliph.FlipH() #水平翻转
        self.process_FlipV=flipv.FlipV() #垂直翻转
        self.process_Noise=noise.Noise(0.02) #添加噪声
        self.process_Rotate=rotate.Rotate(-90) # 旋转图像

        self.process_color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # 颜色抖动
        self.process_random_affine = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)  # 随机仿射变换

        
    def preprocess(self,img,gt=None):
        # if random()<0.1:
        #     img=self.process_blur.process(img)
        # if random()<0.1:
        #     img=self.process_Noise.process(img)
        # if random()>0.5:
        #     img=self.process_Rotate.process(img)
        #     gt=self.process_Rotate.process(gt)
            
            
        if random()<0.5:
            img=self.process_FlipH.process(img)
            gt=self.process_FlipH.process(gt)
        if random()<0.5:
            img=self.process_FlipV.process(img)
            gt=self.process_FlipV.process(gt)
        
            
        # if img[img>1].any()==True:
        #     return img,gt
        # else:
        #     return img*255,gt
        gt[gt==1]=255
        gt=np.array(gt,dtype=np.uint8)
        # print(len(gt[gt>127]))
        # print(np.max(gt))
        img=np.ascontiguousarray(img)
        gt=np.ascontiguousarray(gt)
        return img,gt
