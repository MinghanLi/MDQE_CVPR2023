# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class MaskHead(nn.Module):
    def __init__(self, hidden_dim, fpn_dims, num_frames, num_classes):
        super().__init__()
        self.num_frames = num_frames
        self.num_gen_params = hidden_dim // 8
        self.num_classes = num_classes

        # 256 resnet => 32 per group; 96/192 swin transformer => 12/24 per group
        self.lay1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, hidden_dim)
        self.lay2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, hidden_dim)
        self.lay3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn3 = nn.GroupNorm(8, hidden_dim)
        self.out_lay1 = DepthwiseSeparableConv2d(hidden_dim, hidden_dim, 5, padding=2,
                                                 activation=F.relu)
        self.out_uplay = nn.ConvTranspose2d(hidden_dim, hidden_dim, 1, stride=2,
                                            output_padding=1, groups=hidden_dim)
        self.out_lay2 = DepthwiseSeparableConv2d(hidden_dim, self.num_gen_params, 5, padding=2,
                                                 activation=F.relu)

        self.adapter1 = nn.Conv2d(fpn_dims[0], hidden_dim, 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], hidden_dim, 1)
        self.act_layer = nn.GELU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, fpns: List[Tensor], is_training=False):
        x = self.lay1(x)
        x = self.gn1(x)
        x = self.act_layer(x)

        cur_fpn = self.adapter1(fpns[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)
        x = self.gn2(x)
        x = self.act_layer(x)

        cur_fpn = self.adapter2(fpns[1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = self.act_layer(x)

        proto = self.out_lay2(self.out_uplay(self.out_lay1(x)))  # BTxCxHxW
        B = proto.shape[0] // self.num_frames if is_training else 1
        proto = rearrange(proto, '(B T) M H W -> B M T H W', B=B)  # BxTxMxHxW

        return proto


class DepthwiseSeparableConv2d(nn.Module):
    """
    A kxk depthwise convolution + a 1x1 convolution.

    In :paper:`xception`, norm & activation are applied on the second conv.
    :paper:`mobilenet` uses norm & activation on both convs.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
        activation=None
    ):
        """
        Args:
            norm1, norm2 (str or callable): normalization for the two conv layers.
            activation1, activation2 (callable(Tensor) -> Tensor): activation
                function for the two conv layers.
        """
        super().__init__()
        self.activation = activation
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1)
        gn_dim = 32 if out_channels % 32 == 0 else 24
        self.gn = nn.GroupNorm(gn_dim, out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.gn(self.pointwise(self.depthwise(x)))
        return self.activation(x) if self.activation is not None else x
