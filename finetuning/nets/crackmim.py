from tkinter import W
from tkinter.tix import MAIN
from turtle import mainloop
from unicodedata import name
from pip import main
from torch_dct import dct, idct

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from einops import rearrange
import complexPyTorch.complexLayers as CPL
from thop import profile
from torchinfo import summary


SIZE=448
# SIZE=512 

def complex_gelu(input):
    return F.gelu(input.real).type(torch.complex64)+1j*F.gelu(input.imag).type(torch.complex64)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, H, W,stage):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            PatchMerging(in_channels, out_channels),
            FDConv_Block(out_channels, out_channels, (H,W)),
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class BAM(nn.Module):
    """
    中间平均池化、max池化 的结构，提取全局
    """
    def __init__(self, in_channels, W, H, freq_sel_method = 'top16'):
        super(BAM, self).__init__()
        self.in_channels = in_channels

        # local channel
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.tw = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0,
                            groups=self.in_channels)
        self.twln = nn.LayerNorm([self.in_channels, 1, 1])
        self.sigmoid = nn.Sigmoid()
        self.register_parameter('wdct', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))
        self.register_parameter('wmax', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))


    def forward(self, x):
        N, C, H, W = x.shape  # global
        # self and local
        x_s = (self.wmax * self.maxpool(x).squeeze(-1)) + self.wdct * (self.gap(x).squeeze(-1))
        # attention weights
        x_s = x_s.unsqueeze(-1)
   
        att_c =self.sigmoid(self.twln(self.tw(x_s)))
        return att_c

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Linear(dim, dim * 4, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B,C,H,W
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        x = self.expand(x)
        x = x.view(B, H, W, -1)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, H * 2, W * 2, C // 4)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm): # dim=out-dim=64
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H,W
        """
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C] 0::2 从0开始每隔2个选一个，1::2 从1开始每隔2个选一个
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C] 在x3的维度拼接
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
        x = self.norm(x)
        x = self.reduction(x).view(B, H // 2, W // 2, -1)  # [B, H/2*W/2, 2*C]
        x = x.permute(0, 3, 1, 2)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768, group_num=4):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim // group_num)

    def forward(self, x):
        x = self.dwconv(x)
        return x
def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):

    return torch.nn.Conv2d(in_, out, 3, padding=1)




class Mlp(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = out_features // 4
        self.fc1 = Conv1X1(in_features, hidden_features)
        self.gn1=nn.GroupNorm(hidden_features//4,hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gn2 = nn.GroupNorm(hidden_features // 4, hidden_features)
        self.act = act_layer()
        self.fc2 = Conv1X1(hidden_features, out_features)
        self.gn3=nn.GroupNorm(out_features//4,out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x=self.gn1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x=self.gn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x=self.gn3(x)
        x = self.drop(x)
        return x
    
# from fightingcv_attention.attention.SKAttention import SKAttention
from fightingcv_attention.attention.CBAM import CBAMBlock
# from fightingcv_attention.attention.DANet import DAModule
# from fightingcv_attention.attention.EMSA import EMSA
# from fightingcv_attention.attention.OutlookAttention import OutlookAttention
# from fightingcv_attention.attention.S2Attention import S2Attention
# from fightingcv_attention.attention.MobileViTv2Attention import MobileViTv2Attention
# from fightingcv_attention.attention.DAT import DAT, DAttentionBaseline
from fightingcv_attention.attention.CrissCrossAttention import CrissCrossAttention
# from fightingcv_attention.attention.Axial_attention import AxialImageTransformer
# from .Attention import GLAM, SegNeXtAtt, FocalAtt, CloAttention, EMA, ECA, CSWinBlock


class TFBlock(nn.Module):

    def __init__(self, in_chnnels, out_chnnels, input_resolution, mlp_ratio=2., drop=0.3, 
                 drop_path=0., act_layer=nn.GELU, linear=False):
        super(TFBlock, self).__init__()
        self.in_chnnels = in_chnnels
        self.out_chnnels = out_chnnels
        
        self.sp = SP(in_chnnels)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.mlp = Mlp(in_features=in_chnnels, out_features=out_chnnels, act_layer=act_layer, drop=drop, linear=linear)

        self.norm1 = nn.LayerNorm([in_chnnels, input_resolution[0],input_resolution[1]])
        self.norm2 = nn.LayerNorm([in_chnnels, input_resolution[0],input_resolution[1]])
        # self.norm1 = nn.BatchNorm2d(in_chnnels)
        # self.norm2 = nn.BatchNorm2d(in_chnnels)
        
        # self.norm3 = nn.BatchNorm2d(in_chnnels)

        self.linear = nn.Linear(in_chnnels, out_chnnels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        
        
        # x = x + self.drop_path(self.attn(x))
        # x1 = self.linear(x.permute(0, 2, 3, 1)).view(B, C, H, W)

        # x = x + self.drop_path(self.sp(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x + self.drop_path(self.sp(x))
        x = x + self.drop_path(self.mlp(x))

        return x

class LocalAtt(nn.Module):
    def __init__(self, dim):
        super(LocalAtt, self).__init__()
        self.dim = dim
        # self.proj_1 = ComplexConv2d(dim, dim, 1)
        
        self.dw_conv = nn.Sequential(
            # nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            # nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
            nn.Conv2d(dim, dim // 4, 3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            # nn.GELU(),
        )

        self.pw_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            # nn.Conv2d(dim, dim // 2, 3, padding=1),
            # nn.BatchNorm2d(dim // 2),
            # nn.GELU(),
            # nn.Conv2d(dim // 2, dim // 2, 3, padding=1),
            # nn.BatchNorm2d(dim // 2),
            # nn.GELU(),
        )
        self.proj_2 = nn.Conv2d(dim*2, dim, 1)
        self.act = nn.GELU()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input):
        s_attn = self.dw_conv(input)

        c_attn = self.pw_conv(input)
        attn = self.act(s_attn * c_attn)
        # attn = self.proj_2(attn)

        return attn

class GLAtt(nn.Module):
    def __init__(self, dim):
        super(GLAtt, self).__init__()
        self.dim = dim
        # self.proj_1 = ComplexConv2d(dim, dim, 1)
        
        self.globalAtt = nn.Sequential(
            CrissCrossAttention(dim)
        )

        self.localAtt = nn.Sequential(
            LocalAtt(dim)
        )
        self.proj_2 = nn.Conv2d(dim*2, dim, 1)
        self.act = nn.GELU()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input):
        s_attn = self.globalAtt(input)

        c_attn = self.localAtt(input)
        attn = self.act(s_attn + c_attn)
        # attn = self.proj_2(attn)

        return attn

class TFBlock1(nn.Module):

    def __init__(self, in_chnnels, out_chnnels, input_resolution, mlp_ratio=2., drop=0.3,
                 drop_path=0., act_layer=nn.GELU, linear=False):
        super(TFBlock1, self).__init__()
        self.in_chnnels = in_chnnels
        self.out_chnnels = out_chnnels
        # self.attn = LambdaAttention(
        #     in_channels=in_chnnels, out_channels=out_chnnels
        # )
        # self.attn = TDAtt(in_chnnels)
        self.sp = SP(in_chnnels)
        # self.sp = SP1(in_chnnels)
        # self.sp = SP_TD(in_chnnels)

        self.attn = LocalAtt(in_chnnels)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=in_chnnels, out_features=out_chnnels, act_layer=act_layer, drop=drop, linear=linear)
        # self.mlp = SwinTransformerBlock(dim=in_chnnels, input_resolution=input_resolution, num_heads=2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.sp(x))
        x = x + self.drop_path(self.mlp(x))

        return x

class FDConv_Block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, input_resolution):

        super().__init__()
        self.double_conv = nn.Sequential(
            TFBlock(in_channels, in_channels, input_resolution),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
          
        )

    def forward(self, x):
        return self.double_conv(x)





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out





class decoder(nn.Module):
    def __init__(self, in_channels, out_channels,H,W,stage):
        super(decoder, self).__init__()
        # ConvTranspose
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.FDBlock = nn.Sequential(
                        TFBlock1(in_channels, in_channels, (H,W)), 
                        nn.BatchNorm2d(in_channels),
                        nn.GELU())
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
       

    def forward(self, x_copy, x, interpolate=True):
        out = self.up(x)
        if interpolate:
            # Iterative filling, in order to obtain better results
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True
                                )
        else:
            # Different filling volume
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        # Splicing
        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.FDBlock(out)
        out_conv = self.conv(out_conv)
        return out_conv

class ResDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, H, W,stage):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            Bottleneck1(in_channels, out_channels, stride=2),
            # FDConv_Block(out_channels, in_channels, (H,W))
            TFBlock1(out_channels, out_channels, (H,W)),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.maxpool_conv(x)




class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, H, W, stage,bilinear=True, ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = FDConv_Block(in_channels, in_channels, (H,W))
        else:
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up = PatchExpand(in_channels)
            self.conv = nn.Sequential(FDConv_Block(in_channels, in_channels, (H,W)))
            # self.conv = FDConv_Block(in_channels, in_channels, (H,W))
        self.convv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x2, x1):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return self.convv(x)


class Bottleneck1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, inplanes * 2,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(inplanes * 2))
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out

class Modulate(nn.Module):
    def __init__(self, dim):
        super(Modulate, self).__init__()
        self.dim = dim
        # self.proj_1 = ComplexConv2d(dim, dim, 1)
        self.dw_conv = nn.Sequential(
            # ComplexConv2d(dim, dim, 5, padding=2, groups=dim),
            # ComplexConv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim // 2, 3, stride=1, padding=1, groups=dim // 2, dilation=1)
            # ComplexSigmoid()
        )

        self.pw_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1, bias=True)
        )
        # self.proj_2 = ComplexConv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input):
        s_attn = self.dw_conv(input)
        c_attn = self.pw_conv(input)
        attn = self.act(s_attn + c_attn)
        return attn

class SP1(nn.Module):
    def __init__(self, dim):
        super(SP1, self).__init__()
        self.dim = dim
        # self.proj_1 = ComplexConv2d(dim, dim, 1)
        
        self.dw_conv = nn.Sequential(
            Modulate(dim)
        )

        self.pw_conv = nn.Sequential(
            Modulate(dim)
        )
        self.proj_2 = nn.Conv2d(dim*2, dim, 1)
        self.act = nn.GELU()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input):
        s_attn = dct(input)
        s_attn = self.dw_conv(s_attn)
        s_attn = idct(s_attn)

        c_attn = self.pw_conv(input)
        attn = self.act(torch.cat((s_attn, c_attn), dim=1))
        # attn = self.proj_2(attn)

        return attn

class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        return x



class SKAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        # self.d = channel // 8
        
        self.dw = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel),
            # nn.BatchNorm2d(channel),
            # nn.ReLU(inplace=False),
            nn.Conv2d(channel, channel, kernel_size=1)
            )
        # self.conv = nn.Conv2d(channel, channel//2, kernel_size=1)
        self.pw = nn.Sequential(
            # nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel),
            nn.Conv2d(channel, channel, kernel_size=1),
            # nn.BatchNorm2d(channel),
            # nn.ReLU(inplace=False)
            )
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1)
            )

        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        conv_outs.append(self.dw(x))
        conv_outs.append(self.pw(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = self.conv(x)  # bs,c,h,w
        # U = sum(conv_outs)  # bs,c,h,w
        # U = conv_outs[0] * conv_outs[1]

        ### reduction channel
        S1 = U.mean(-1).mean(-1)  # bs,c
        S2 = U.max(-1)[0].max(-1)[0]  # bs, c
        S = S1 + S2
        # S = self.conv(S)
        # S1 = torch.mean(U, dim=1, keepdim=True)  #bs,1,h,w

        Z = self.fc(S)  # bs,d
        # Z = self.relu(Z)

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        # V = (attention_weughts)

        return V


# class SKAttention(nn.Module):

#     def __init__(self, channel=512, reduction=16, L=32):
#         super().__init__()
#         self.d = max(L, channel // reduction)
#         # self.d = channel // 8
        
#         self.dw = nn.Sequential(
#             nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel),
#             # nn.BatchNorm2d(channel),
#             # nn.ReLU(inplace=False),
#             nn.Conv2d(channel, channel, kernel_size=1)
#             )
#         # self.conv = nn.Conv2d(channel, channel//2, kernel_size=1)
#         self.pw = nn.Sequential(
#             # nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel),
#             nn.Conv2d(channel, channel, kernel_size=1),
#             # nn.BatchNorm2d(channel),
#             # nn.ReLU(inplace=False)
#             )
#         self.conv = nn.Sequential(
#             nn.Conv2d(channel, channel, kernel_size=1)
#             )

#         # self.fc = nn.Linear(channel, self.d)
#         # self.fcs = nn.ModuleList([])
#         # for i in range(2):
#         #     self.fcs.append(nn.Linear(self.d, channel))
#         # self.softmax = nn.Softmax(dim=0)


#         self.convv = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
#         self.sigmoid = nn.Sigmoid()


#     def forward(self, x):
#         bs, c, _, _ = x.size()
#         conv_outs = []
#         ### split
#         conv_outs.append(self.dw(x))
#         conv_outs.append(self.pw(x))
#         feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

#         ### fuse
#         U = self.conv(x)  # bs,c,h,w
#         # U = sum(conv_outs)  # bs,c,h,w
#         # U = conv_outs[0] * conv_outs[1]

#         ### reduction channel
#         S1 = U.mean(-1).mean(-1)  # bs,c
#         S2 = U.max(-1)[0].max(-1)[0]  # bs, c
#         S = S1 + S2
#         # S = self.conv(S)
#         # S1 = torch.mean(U, dim=1, keepdim=True)  #bs,1,h,w

#         Z = S.unsqueeze(-1).transpose(-1, -2)
#         Z = self.convv(Z).transpose(-1, -2).unsqueeze(-1)
#         weight = self.sigmoid(Z)

#         ### fuse
#         V = (weight * feats[0]) + (weight * feats[1])

#         return V
    
class SP(nn.Module):
    def __init__(self, dim):
        super(SP, self).__init__()
        self.dim = dim
        # self.proj_1 = ComplexConv2d(dim, dim, 1)
        
        self.sk1 = SKAttention(channel=dim//2, reduction=16, L=32)
        self.sk2 = SKAttention(channel=dim//2, reduction=16, L=32)
        # self.sk2 = nn.Sequential(
        #     nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, groups=dim//2),
        #     nn.Conv2d(dim//2, dim//2, kernel_size=1))

        self.act = nn.GELU()
        # self.apply(self._init_weights)

        self.conv = nn.Conv2d(dim, dim, kernel_size=1)


    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input):
        # input = self.se(input)
        # input = self.conv(input)
        input1, input2 = torch.chunk(input, 2, dim=1)

        s_attn = dct(input1)
        s_attn = self.sk1(s_attn)
        s_attn = idct(s_attn)

        c_attn = self.sk2(input2)
        attn = self.act(torch.cat((s_attn, c_attn), dim=1))
        attn = self.conv(attn)


        return attn


class SP_TD(nn.Module):
    def __init__(self, dim):
        super(SP_TD, self).__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 15), padding=(0, 7), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (15, 1), padding=(7, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

        self.se = SE_Block(dim)

        self.act = nn.GELU()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input):
        attn = self.conv0(input)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        return attn
    
class CrackMiM(nn.Module):
    def __init__(self, num_classes=1, bilinear=False):
        self.H = self.W = SIZE 
        self.inplanes = 128
        self.bilinear = bilinear
        super(CrackMiM, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        
        )

    
        factor = 2 if bilinear else 1
       

        self.down1 = Down(64, 128, self.H // 2, self.W // 2,stage=0)
        self.down2 = Down(128, 256, self.H // 4, self.W // 4,stage=0)
        self.layer1 = Down(256, 512, self.W //8, self.H //8,stage=2)
        self.layer2 = Down(512, 1024 // factor, self.W // 16, self.H // 16,stage=2)

        self.up0 = Up(1024, 512,self.H//8, self.W//8,stage = 2, bilinear=bilinear)
        self.up1 = Up(512, 256, self.H//4, self.W//4,stage = 3, bilinear=bilinear)
        self.up2 = Up(256, 128 // factor, self.H // 2, self.W // 2,stage=4, bilinear=bilinear)
        self.up3 = Up(128, 64, self.H, self.W, stage=5,bilinear=bilinear)
   
        #Full connection
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initalize_weights()

    def _initalize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


    def forward(self, x):
 
        x0 = self.conv0(x) #x0 [H W 64]

        x1 = self.down1(x0)# x1 [H/2,W/2,128] 2c

        x2 = self.down2(x1)#  256 4c

        x3 = self.layer1(x2)# x3 [H/8,W/8,512] layer1

        x4 = self.layer2(x3)# x4 [H/16,W/16,1024] layer2


        x = self.up0(x3, x4)
        x = self.up1(x2, x)
        x = self.up2(x1, x)
        x = self.up3(x0, x)

        final = self.final_conv(x)

        return final



