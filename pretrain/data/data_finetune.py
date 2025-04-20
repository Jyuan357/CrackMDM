import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from torch.utils import data
from PIL import Image
import numpy as np
import cv2
class Crackloader(data.Dataset):

    def __init__(self, txt_paths, is_train, size):
        self.istrain = is_train
        self.size = size

        self.train_set_path = self.make_dataset(txt_paths)

        self.tensorTrans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.train_set_path)

    def __getitem__(self, index):
        img_path, lbl_path = self.train_set_path[index]

        img = cv2.imread(img_path) 
        lbl = cv2.imread(lbl_path,0)

        img, lbl = self.preprocess(img, lbl)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.uint8)

        # cv2.imwrite("img.jpg", img)
        # cv2.imwrite("lbl.png", lbl)


        img = self.tensorTrans(img)
        img=img.type(torch.FloatTensor)
        if(self.istrain):
            _, binary = cv2.threshold(lbl,127, 1, cv2.THRESH_BINARY) 
        else:
            _, binary = cv2.threshold(lbl,80, 255, cv2.THRESH_BINARY) 
        

        return img, binary 

    def make_dataset(self, txt_paths):
        dataset = []
        index=0

        for txt_path in txt_paths:
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
        
        if H / W == 448 / 448:
            img_resized = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            lbl_resized = cv2.resize(lbl, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        else:
            target_ratio = 512 / 512  
            current_ratio = H / W  
            
            if current_ratio < target_ratio:
         
                target_height = int(W * target_ratio)
                img_cropped = img[:target_height, :, :]
                lbl_cropped = lbl[:target_height, :, :]
            else:

                target_width = int(H / target_ratio)
                img_cropped = img[:, :target_width, :]
                lbl_cropped = lbl[:, :target_width, :]

            img_resized = cv2.resize(img_cropped, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            lbl_resized = cv2.resize(lbl_cropped, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return img_resized, lbl_resized


def build_loader_finetune(config, logger):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, logger=logger)
    config.freeze()
    dataset_val, _ = build_dataset(is_train=False, config=config, logger=logger)
    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    # sampler_train = DistributedSampler(
    #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    # sampler_val = DistributedSampler(
    #     dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    # )

    data_loader_train = DataLoader(
        dataset_train, #sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, #sampler=sampler_val,
        batch_size=1, # config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config, logger):
    
    if config.DATA.DATASET == 'imagenet':
        transform = build_transform(is_train, config)
        logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')

        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'crack':
        path_root = "datasets/new_dataset/"
        path_root += 'train' if is_train else 'test'
        path_list = [os.path.join(path_root, txt) for txt in config.DATA.DATA_PATH]
        dataset = Crackloader(txt_paths=path_list, is_train = is_train, size=config.DATA.IMG_SIZE)
        nb_classes = len(config.DATA.DATA_PATH)
    else:
        raise NotImplementedError("Error dataset name.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
