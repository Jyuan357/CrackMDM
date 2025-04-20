import os
import os.path
import torch
from torchvision import transforms
import numpy as np
import scipy.misc as m
import glob
import torch.utils.data as data
import cv2
from torch.utils import data
# from .aug.process import DataAug



class Crackloader(data.Dataset):

    def __init__(self, txt_path, size, normalize=True):
        self.txt_path = txt_path
        self.size = size

        if normalize:
            self.img_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.img_transforms = transforms.ToTensor()

        self.train_set_path = self.make_dataset(txt_path)
        # self.Aug=DataAug()

    def __len__(self):
        return len(self.train_set_path)

    def __getitem__(self, index):
        img_path, lbl_path = self.train_set_path[index]

        img = cv2.imread(img_path)
        lbl = cv2.imread(lbl_path,0) 
        # print(lbl)

        img, lbl = self.preprocess(img, lbl)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.uint8)

        # img,lbl=self.Aug.preprocess(img,lbl)
        img = self.img_transforms(img)
        img=img.type(torch.FloatTensor)


        _, binary = cv2.threshold(lbl,127, 1, cv2.THRESH_BINARY)

        return img, binary

    def make_dataset(self, txt_path):
        dataset = []
        index=0
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                # print(index,line)
                index+=1
                line = ''.join(line).strip()
                line_list = line.split(' ')
                dataset.append([line_list[0], line_list[1]])
        return dataset
    
    def preprocess(self, img, lbl):
        H, W, _ = img.shape

        target_width = self.size
        target_height = self.size

        if target_height > H or target_height > W:
            img=cv2.resize(img,(self.size, self.size),cv2.INTER_NEAREST )
            lbl=cv2.resize(lbl,(self.size, self.size),cv2.INTER_NEAREST )
        else:
            sw = (W - target_width) // 2
            sh = (H - target_height) // 2

            img = img[sh:sh+target_height, sw:sw+target_width, :]
            lbl = lbl[sh:sh+target_height, sw:sw+target_width]

        return img, lbl

from PIL import Image
from torchvision.transforms import functional as F
import random
class Crackloader_Aug(data.Dataset):
    def __init__(self, txt_path, size, augmentation=True):
        self.train_set_path = self.make_dataset(txt_path)
        self.image_size = size
        self.augmentation = augmentation

    def __len__(self):
        return len(self.train_set_path)

    def make_dataset(self, txt_path):
        dataset = []
        index=0
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                # print(index,line)
                index+=1
                line = ''.join(line).strip()
                line_list = line.split(' ')
                dataset.append([line_list[0], line_list[1]])
        return dataset

    def augmentate(self, image, mask):

        image = F.resize(image, size=[self.image_size, self.image_size])
        mask = F.resize(mask, size=[self.image_size, self.image_size])

        image = F.adjust_gamma(image, gamma=random.uniform(0.8, 1.2))
        image = F.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
        image = F.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
        image = F.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))
        image = F.adjust_hue(image, hue_factor=random.uniform(-0.2, 0.2))

        image = F.to_tensor(image)  


        image_mask = torch.cat([image, mask], dim=0)

        if random.uniform(0, 1) > 0.5:
            image_mask = F.hflip(image_mask)
        if random.uniform(0, 1) > 0.5:
            image_mask = F.vflip(image_mask)
        if random.uniform(0, 1) > 0.5:
            angles = [19, 23, 90]
            angle = random.choice(angles)   
            image_mask = F.rotate(image_mask, angle=angle)


        image = image_mask[0:3, ...]
        mask = image_mask[3:4, ...]
        # image = image / 255
        # mask = mask / 255

        return image, mask.squeeze(0)

    def __getitem__(self, index):
        image_path, mask_path = self.train_set_path[index]

        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')    
        mask = Image.open(mask_path)

        if self.augmentation:
            image, mask = self.augmentate(image, mask)
        return image, mask


from torchvision.utils import save_image
if __name__ == '__main__':
    db_train = Crackloader_Aug(txt_path="datasets/new_dataset/train/CFD.txt", size=448)
    
    # trainloader = data.DataLoader(db_train, batch_size= 4, shuffle=True, num_workers=8, pin_memory=True)

    for i,(image, mask) in enumerate(db_train):
        print(image.shape, mask.shape)
        print(torch.max(image), torch.min(image))
        print(torch.max(mask), torch.min(mask))

        image = np.array(image.numpy() * 255.0 , dtype=np.uint8)
        mask = np.array(mask.numpy() * 255.0 , dtype=np.uint8)

        cv2.imwrite(os.path.join('datasets/pics/image', str(i)+'.png'), image.transpose((1, 2, 0)))
        cv2.imwrite(os.path.join('datasets/pics/label', str(i)+'.png'), mask.transpose((1, 2, 0)))

