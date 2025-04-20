import torch
import torch.nn as nn
import math

class HOGLayer(nn.Module):
    def __init__(self, nbins, pool, bias=False, max_angle=math.pi, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins

        self.conv = nn.Conv2d(1, 2, 3, stride=stride, padding=padding, dilation=dilation, padding_mode='reflect', bias=bias)
        mat = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.conv.weight.data = mat[:, None, :, :]

        self.max_angle = max_angle
        # self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)
        self.pooler = nn.Conv2d(9, 9, kernel_size=1, stride=pool, padding=0)


    @ torch.no_grad()
    def forward(self, x):  # [B, 1, 224, 224]
        gxy = self.conv(x)

        # 2. Mag/ Phase
        mag = gxy.norm(dim=1)
        norm = mag[:, None, :, :]
        phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])

        # 3. Binning Mag with linear interpolation
        phase_int = phase/self.max_angle*self.nbins
        phase_int = phase_int[:, None, :, :]

        n, c, h, w = gxy.shape
        out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
        out.scatter_(1, phase_int.floor().long() % self.nbins, norm)

        hog = self.pooler(out)
        hog = nn.functional.normalize(hog, p=2, dim=1) 
        return hog

class LocalLoss2(nn.Module):

    def __init__(self,
                 
                 hog_bias=False):
        super(LocalLoss2, self).__init__()
        
        # target
        self.hog_enc = nn.ModuleList([HOGLayer(nbins=9, pool=k, bias=hog_bias) for k in [16, 8, 4, 2, 1]])
        for hog_enc in self.hog_enc:
            for param in hog_enc.parameters():
                param.requires_grad = False

    def HOG(self, imgs, k):  # [B, 3, 224, 224]
        """
        imgs: (N, 3, H, W)
        x: (N, L, d)
        """
        hog_R = self.hog_enc[k](imgs[:, :1, :, :])  # [B, nb, h, w]
        hog_G = self.hog_enc[k](imgs[:, 1:2, :, :])  # [B, nb, h, w]
        hog_B = self.hog_enc[k](imgs[:, 2:, :, :])  # [B, nb, h, w]
        hog_feat = torch.cat([hog_R, hog_G, hog_B], 1)  # [B, 3*nb, h, w]
        # hog_feat = hog_feat.flatten(2, 3).transpose(1, 2)
        return hog_feat
    
    def recal_mask(self, mask, k):
        scale = [16, 8, 4, 2, 1]
        B, L, s = mask.size(0), mask.size(1), scale[k]
        H, W = mask.size(2), mask.size(3)
        # if s >= 1:
        #     mask = mask.squeeze(1).unsqueeze(3).unsqueeze(2).repeat(1, 1, s, 1, s).reshape(B, -1)
        # else:
        #     s = int(1/s)
        mask = mask.float().reshape(B, H//s, s, W//s, s).transpose(2, 3).mean((-2, -1))

        return mask
    
    def forward(self, pred, imgs, mask):

        target = [self.HOG(imgs, k) for k in range(len(self.hog_enc))]

        # visualize_hog_features(target)

        loss = 0
        for k in range(len(target)):
            M = self.recal_mask(mask, k)
            loss += (((pred[k]-target[k])**2).mean(dim=1)*M).sum()/M.sum()

        return loss

import matplotlib.pyplot as plt
import numpy as np
def visualize_hog_features(hog_features):
    """
    Visualize HOG features as images.
    Args:
        hog_features (Tensor): The HOG features with shape [B, nb, H, W]
    """
    # Convert tensor to numpy array for visualization
    hog_features = hog_features[0].cpu().detach().numpy()

    # Assuming hog_features is of shape [B, nb, H, W]
    B, nb, H, W = hog_features.shape

    # Save each feature map as a separate image
    for b in range(B):  # For each batch sample
        for i in range(nb):  # For each feature map
            feature_map = hog_features[b, i, :, :]
            file_path = f"batch{b}_feature{i}.png"
            plt.imsave(file_path, feature_map, cmap='gray')
            print(f"Saved {file_path}")



class LocalLoss1(nn.Module):

    def __init__(self,
                 
                 hog_bias=False):
        super(LocalLoss1, self).__init__()
        
        # target
        self.hog_enc = HOGLayer(nbins=9, pool=1, bias=hog_bias)
        for param in self.hog_enc.parameters():
            param.requires_grad = False

    def HOG(self, imgs, k=1):  # [B, 3, 224, 224]
        """
        imgs: (N, 3, H, W)
        x: (N, L, d)
        """
        hog_R = self.hog_enc(imgs[:, :1, :, :])  # [B, nb, h, w]
        hog_G = self.hog_enc(imgs[:, 1:2, :, :])  # [B, nb, h, w]
        hog_B = self.hog_enc(imgs[:, 2:, :, :])  # [B, nb, h, w]
        hog_feat = torch.cat([hog_R, hog_G, hog_B], 1)  # [B, 3*nb, h, w]
        # hog_feat = hog_feat.flatten(2, 3).transpose(1, 2)
        return hog_feat
    

    def forward(self, pred, imgs):

        target = self.HOG(imgs)
        B, C, H, W = target.shape


        loss_matrix = torch.empty(B, len(target), C, H, W, dtype=torch.float32, device=imgs.device)
        
        loss_matrix = ((pred-target)**2)

        return loss_matrix
