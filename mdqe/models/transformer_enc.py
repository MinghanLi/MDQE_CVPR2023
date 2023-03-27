import copy
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.cuda.amp import autocast

from .ops.modules.ms_deform_attn import MSDeformAttn as MSAttnBlock
from .misc import make_reference_points


class Transformer_Enc(nn.Module):
    def __init__(self, dim, n_heads=8, n_feature_levels=4, n_enc_points=3, n_enc_layers=3,
                 n_frames=1):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.n_feature_levels = n_feature_levels
        self.n_enc_layers = n_enc_layers

        # encoder
        encoder_layer = EncoderLayer(dim, n_heads, n_feature_levels, n_enc_points, n_frames=n_frames,
                                     pred_offsets=True)
        encoder_norm = nn.LayerNorm(dim)
        self.encoder = Encoder(encoder_layer, n_enc_layers, encoder_norm, return_intermediate=False)

        self.level_embed = nn.Parameter(torch.Tensor(n_feature_levels, dim))
        nn.init.normal_(self.level_embed)

    def forward(self, srcs, masks, pos_embeds, is_training=True):
        BT = srcs[0].shape[0]

        # srcs: num_feature_layers * [NxTxCxHixWi], query_embed: QxC
        src_flatten, mask_flatten, lvl_pos_embed_flatten = [], [], []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            spatial_shapes += [src.size()[-2:]]

            src_flatten += [rearrange(src, 'B C H W -> B (H W) C')]
            mask_flatten += [rearrange(mask, 'B H W -> B (H W)')]
            pos_embed = rearrange(pos_embed, 'B C H W -> B (H W) C')
            lvl_pos_embed_flatten += [pos_embed + self.level_embed[lvl].view(1, 1, -1)]

        src_flatten = torch.cat(src_flatten, dim=-2)    # BTx\sum(HW)xC
        mask_flatten = torch.cat(mask_flatten, dim=-1)  # BTx\sum(HW)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=-2)  # BTx\sum(HW)xC
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        reference_points = torch.cat([make_reference_points(shape, device=src_flatten.device)
                                      for shape in spatial_shapes])       # Nx2

        # Encoder
        encoded_features = self.encoder(src_flatten,
                                        repeat(reference_points, 'N C -> B N C', B=BT),
                                        spatial_shapes,
                                        lvl_pos_embed_flatten,
                                        mask_flatten,
                                        is_training)  # LxBTxNxC

        return encoded_features


class EncoderLayer(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        n_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, n_heads, n_feature_levels, n_points=3, mlp_ratio=4.,
                 drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_frames=1, pred_offsets=True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio

        self.self_attn = MSAttnBlock(dim, n_points=n_points, n_heads=n_heads,
                                     n_levels=n_feature_levels, pred_offsets=pred_offsets, mode='spatial')
        self.dropout1 = nn.Dropout(drop)
        self.norm1 = norm_layer(dim)

        self.n_frames = n_frames

        d_ffn = int(dim * mlp_ratio)
        self.linear1 = nn.Linear(dim, d_ffn)
        self.activation = act_layer()
        self.dropout2 = nn.Dropout(drop)
        self.linear2 = nn.Linear(d_ffn, dim)
        self.dropout3 = nn.Dropout(drop)
        self.norm2 = norm_layer(dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_pos, reference_boxes, spatial_shapes, padding_mask=None, is_training=True):
        BT, Q, C = x.shape
        x2 = self.self_attn(self.with_pos_embed(x, x_pos), reference_boxes, x, spatial_shapes, padding_mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm, return_intermediate):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    @autocast(enabled=False)
    def forward(self, src, reference_points, spatial_shapes, pos=None, padding_mask=None, is_training=True):
        output = src
        reference_boxes = torch.cat([reference_points, torch.ones_like(reference_points) * 0.1], dim=-1)  # Nx4

        intermediate = []
        for layer in self.layers:
            output = layer(output, pos, reference_boxes, spatial_shapes, padding_mask, is_training)
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.return_intermediate:
            return torch.stack(intermediate)
    
        return self.norm(output)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


