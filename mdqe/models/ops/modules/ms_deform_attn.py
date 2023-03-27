# ------------------------------------------------------------------------
# SeqFormer Deformable Attention.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
from torch.cuda.amp import autocast

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from einops import rearrange, repeat

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, n_frames=1, pred_offsets=True, mode='spatial'):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head "
                          "a power of 2, which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.mode = mode
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.pred_offsets = pred_offsets
        self.scale = 8.

        self.n_frames = n_frames
        if mode == 'spatial':
            self.lvl = self.n_levels
            lvl_spatial_scales = torch.arange(1, self.lvl + 1)
        else:
            self.lvl = self.n_frames
            lvl_spatial_scales = torch.as_tensor([2]).repeat(self.lvl)
        self.register_buffer("lvl_spatial_scales", lvl_spatial_scales)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.attention_weights = nn.Linear(d_model, n_heads * self.lvl * n_points)
        if self.pred_offsets:
            self.sampling_offsets = nn.Linear(d_model, n_heads * self.lvl * n_points * 2)
        else:
            self.sampling_grid_offsets = nn.Linear(d_model, n_heads * self.lvl * n_points * 2)

        self._reset_parameters()

    def _reset_parameters(self):
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = repeat(grid_init, 'H C -> H L K C', L=self.lvl, K=self.n_points).clone()
        for k in range(self.n_points):
            grid_init[:, :, k, :] *= k + 1
        grid_init = grid_init / self.n_points * self.scale
        if self.pred_offsets:
            constant_(self.sampling_offsets.weight.data, 0.)
            grid_init = grid_init * 0.05 * self.lvl_spatial_scales.reshape(1, -1, 1, 1)
            with torch.no_grad():
                self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        else:
            # TODO: align sampling points with pred boxes in decoder layers
            sampling_offsets = rearrange(grid_init, 'H L K C -> 1 1 H L K C')
            self.register_buffer("sampling_offsets", sampling_offsets)
            constant_(self.sampling_grid_offsets.weight.data, 0.)
            constant_(self.sampling_grid_offsets.bias.data, 0.)

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask=None):
        if self.mode == 'spatial':
            return self.spatial_forward(query, reference_points, input_flatten, input_spatial_shapes,
                                        input_padding_mask)
        elif self.mode == 'temporal':
            return self.temporal_clip_forward(query, reference_points, input_flatten, input_spatial_shapes,
                                              input_padding_mask)
        else:
            NotImplementedError

    @autocast(enabled=False)
    def spatial_forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask=None):
        """
                query: BxQxC
           ref_points: BxQx4, [cx, cy, w, h], reference boxes
        input_flatten: BxNxC, N=sum_l H_l*W_l
        B: batch size
        T: number of frames
        Q: number of queries
        N: HxW
        K: number of selected points in per feature level
        H: number of heads
        L: number of feature levels
        """

        B, N, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == N

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = rearrange(value, 'B N (H D) -> B N H D', H=self.n_heads)

        reference_points = rearrange(reference_points, 'B Q C -> B Q 1 1 1 C')
        if self.pred_offsets:
            sampling_offsets = rearrange(self.sampling_offsets(query), 'B Q (H L K C) -> B Q H L K C',
                                         H=self.n_heads, L=self.n_levels, K=self.n_points)
        else:
            sampling_offsets = self.sampling_offsets * 0.5 * reference_points[..., 2:]
            sampling_grid_offsets = rearrange(self.sampling_grid_offsets(query), 'B Q (H L K C) -> B Q H L K C',
                                              H=self.n_heads, L=self.lvl, K=self.n_points).to(reference_points)
            sampling_grid_offsets = torch.where(sampling_grid_offsets > -reference_points[..., 2:] * self.scale,
                                                sampling_grid_offsets, -reference_points[..., 2:] * self.scale)
            sampling_grid_offsets = torch.where(sampling_grid_offsets < reference_points[..., 2:] * self.scale,
                                                sampling_grid_offsets, reference_points[..., 2:] * self.scale)
            sampling_offsets = sampling_offsets + sampling_grid_offsets

        sampling_locations = reference_points[..., :2] + sampling_offsets / self.scale

        attention_weights = rearrange(self.attention_weights(query), 'B Q (H O) -> B Q H O',
                                      H=self.n_heads, O=self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = rearrange(attention_weights, 'B Q H (L K) -> B Q H L K',
                                      L=self.n_levels, K=self.n_points)

        input_level_start_index = torch.cat([input_spatial_shapes.new_zeros(1),
                                             input_spatial_shapes.prod(-1).cumsum(0)[:-1]]).long()
        results = MSDeformAttnFunction.apply(value,
                                             input_spatial_shapes,
                                             input_level_start_index,
                                             sampling_locations,
                                             attention_weights,
                                             self.im2col_step)
        output = self.output_proj(results)

        return output

    @autocast(enabled=False)
    def temporal_clip_forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                              input_padding_mask=None):
        """
                query: BxQxC
        input_flatten: BxTxNxC
           ref_points: BxQxLx2
        B: batch size
        T: number of frames
        Q: number of queries
        L: number of feature levels, ==> number of frames
        N: sum{HixWi}, i = 1,...,L
        K: number of points
        H: number of heads
        f: number of frames used, if f=3, [t-1, t, t+1]
        """
        input_spatial_nums = input_spatial_shapes.prod(-1)

        value = self.value_proj(input_flatten)   # BxTxNxC
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = rearrange(value, 'B T N (H D) -> B T N H D', H=self.n_heads)

        reference_points = rearrange(reference_points, 'B Q C -> B Q 1 1 1 C')
        if self.pred_offsets:
            sampling_offsets = rearrange(self.sampling_offsets(query), 'B Q (H L K C) -> B Q H L K C',
                                         H=self.n_heads, L=self.lvl, K=self.n_points)
        else:
            sampling_offsets = self.sampling_offsets * 0.5 * reference_points[..., 2:]
            sampling_grid_offsets = rearrange(self.sampling_grid_offsets(query), 'B Q (H L K C) -> B Q H L K C',
                                              H=self.n_heads, L=self.lvl, K=self.n_points).to(reference_points)
            sampling_grid_offsets = torch.where(sampling_grid_offsets > -reference_points[..., 2:] * self.scale,
                                                sampling_grid_offsets, -reference_points[..., 2:] * self.scale)
            sampling_grid_offsets = torch.where(sampling_grid_offsets < reference_points[..., 2:] * self.scale,
                                                sampling_grid_offsets, reference_points[..., 2:] * self.scale)
            sampling_offsets = sampling_offsets + sampling_grid_offsets

        sampling_locations = reference_points[..., :2] + sampling_offsets / self.scale

        attention_weights = rearrange(self.attention_weights(query), 'B Q (H O) -> B Q H O',
                                      H=self.n_heads, O=self.lvl*self.n_points)
        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = rearrange(attention_weights, 'B Q H (L K) -> B Q H L K', K=self.n_points)

        value_list = value.split([H_*W_ for (H_, W_) in input_spatial_shapes], dim=2)  # [BxTxH_lW_lxC]

        results_list = []
        for l, input_spatial_shape in enumerate(input_spatial_shapes):
            value_l = value_list[l].contiguous().flatten(1, 2)       # BxTH_lW_lxHxD
            input_spatial_shapes_l = repeat(input_spatial_shapes[l], 'C -> T C', T=self.n_frames).contiguous()
            input_level_start_index_l = torch.cat([t*input_spatial_nums[l].reshape(1) for t in range(self.n_frames)])

            output_samp_l = MSDeformAttnFunction.apply(value_l,
                                                       input_spatial_shapes_l,
                                                       input_level_start_index_l,
                                                       sampling_locations.contiguous(),
                                                       attention_weights.contiguous(),
                                                       self.im2col_step)
            results_list.append(output_samp_l)       # BxQxC

        results = torch.stack(results_list).mean(0)  # LxBxQxC -> BxQxC
        output = self.output_proj(results)           # BxQxC

        return output