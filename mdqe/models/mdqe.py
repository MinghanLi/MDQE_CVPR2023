# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
IFC model and criterion classes.
Modified by the authors of Video Instance Segmentation using Inter-Frame Communication Transformer.
"""
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list)


class mdqe(nn.Module):
    """ This is the IFC module that performs object detection """
    def __init__(self, backbone, transformer_enc, transformer_dec, n_frames, num_feature_levels=1, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer_enc architecture. See transformer_enc.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         IFC can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.transformer_enc = transformer_enc
        self.transformer_dec = transformer_dec
        hidden_dim = transformer_enc.dim

        num_backbone_outs = len(backbone.feature_strides)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = backbone.num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)

        self.backbone = backbone
        self.n_frames = n_frames
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size*T x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size*T x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features, pos = self.forward_pre_backbone(samples)

        # Encoder
        encoded_srcs, encoded_masks, spatial_shapes = self.forward_deformable_enc(features, pos)

        # Decoder
        out = self.transformer_dec(encoded_srcs, encoded_masks, spatial_shapes, targets)

        return out

    def forward_pre_backbone(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        return features, pos

    def forward_deformable_enc(self, features, pos, is_training=True):
        # Step 1: Prepare multi-scale features to feed deformable transformer_enc
        spatial_shapes = []
        srcs, masks, poses = [], [], []
        for lf in range(self.transformer_enc.n_feature_levels):
            if lf < len(features):
                # src: BTx_CxHixWi, mask: BTxHixWi,
                src_l, mask_l = features[lf].decompose()
                src_l = self.input_proj[lf](src_l)  # BTxCxHixWi
                pos_l = pos[lf]
            else:
                src_l = features[-1].tensors if lf == len(features) else srcs[-1]
                src_l = self.input_proj[lf](src_l)
                mask_l = F.interpolate(features[-1].mask[None].float(), size=src_l.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src_l, mask_l)).to(src_l.dtype)

            srcs.append(src_l)
            masks.append(mask_l)
            poses.append(pos_l)
            spatial_shapes += [src_l.size()[-2:]]

        # Step 2: Deformable transformer, LxBTxHWxC
        encoded_srcs = self.transformer_enc(srcs, masks, poses, is_training)                  # list[BTxN]
        encoded_masks = torch.cat([rearrange(m, 'B H W -> B (H W)') for m in masks], dim=-1)  # BTxN
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=encoded_srcs.device)

        return encoded_srcs, encoded_masks, spatial_shapes

    def forward_mask_head_inference(self, encoded_srcs, spatial_shapes):
        lvl_start_index = torch.cat([spatial_shapes.new_zeros(1),
                                     spatial_shapes.prod(-1).cumsum(0)]).long()

        # Mask features Head processes a long clip for fast speed of near-online inference
        encoded_srcs = [rearrange(encoded_srcs[:, lvl_start_index[lf]:lvl_start_index[lf + 1]],
                                  'B (H W) C -> B C H W', H=spatial_shapes[lf, 0])
                        for lf in range(spatial_shapes.shape[0])]
        mask_feats = self.transformer_dec.mask_head(encoded_srcs[2], [encoded_srcs[1], encoded_srcs[0]])

        return mask_feats
