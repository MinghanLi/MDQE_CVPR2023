import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.cuda.amp import autocast

from .segmentation import MaskHead as MaskHead_yolact
from .ops.modules.ms_deform_attn import MSDeformAttn as MSAttnBlock
from .misc import MLP, make_reference_points
from ..util.misc import inverse_sigmoid
from ..util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class Transformer_Dec(nn.Module):
    def __init__(self, num_classes, dim, n_heads=8, n_feature_levels=4, n_frames=1,
                 n_dec_points=3, n_dec_layers=3, mlp_ratio=4, n_query=196,
                 fpn_dims=None, dec_temporal=False, clip_peak_matcher=None,
                 rpn_level=0, window_inter_frame_asso=5, query_embed_dim=64,
                 is_coco=False, mask_on=True):
        super().__init__()

        self.num_classes = num_classes
        self.dim = dim
        self.n_heads = n_heads
        self.n_feature_levels = n_feature_levels
        self.n_frames = n_frames
        self.n_query = n_query
        self.clip_peak_matcher = clip_peak_matcher
        self.rpn_level = rpn_level
        self.is_coco = is_coco

        self.query_embed_dim = query_embed_dim
        self.mask_on = mask_on

        self.decoder_norm = nn.LayerNorm(dim)
        decoder_layer = DecoderDefAttnLayer(dim, n_heads, n_feature_levels, n_frames, n_dec_points,
                                            mlp_ratio=mlp_ratio, pred_offsets=False, use_tca=dec_temporal)

        self.bbox_embed = MLP(dim, dim, 4, 3)
        self.point2pos_proj = nn.Linear(2, dim)
        self.decoder = DecoderDefAttn(decoder_layer, n_dec_layers, self.bbox_embed,
                                      self.decoder_norm, n_frames,
                                      point2pos_proj=self.point2pos_proj,
                                      return_intermediate=True)

        # FFN heads
        self.rpn_cls_embed = MLP(dim, dim, num_classes, 3)
        self.cls_embed = MLP(dim, dim, num_classes, 3)
        self.track_embed = MLP(dim, dim, self.query_embed_dim, 3)

        if self.mask_on:
            # mask features head and mask parameters
            self.mask_head = MaskHead_yolact(dim, fpn_dims, n_frames, num_classes)
            self.mask_embed = MLP(dim, dim, self.mask_head.num_gen_params, 3)

        # Query initialization: taking the w*w grid as the matched window (temporal consistency in a short clip)
        self.n_query_bins = int(math.sqrt(self.n_query))
        self.window_inter_frame_asso = window_inter_frame_asso
        i, j = torch.meshgrid(torch.arange(self.n_query_bins), torch.arange(self.n_query_bins))
        indices = torch.stack([j, i], dim=-1).view(-1, 2)
        query_relpos_grid = (indices[:, None] - indices[None]).abs()  # 100x100x2
        self.register_buffer("query_relpos_grid", query_relpos_grid)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSAttnBlock):
                m._reset_parameters()

        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0.)
        bias_value = math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.cls_embed.layers[-1].bias.data, -bias_value)
        nn.init.constant_(self.rpn_cls_embed.layers[-1].bias.data, -bias_value)

    def grid_guided_query_selection(self, sem_cls_conf):
        """
        enc_feat_cls_conf: BTxHxWxK
        split images into 14*14 grid (196 object queries), and select the peaks of class scores in each grid
        as the initialized positional and content embeddings of object queries before Decoder.
        Args:
            sem_cls_conf: BTxHWxK, semantic class confidences of encoder features
        """
        BT, H, W, K = sem_cls_conf.shape
        max_score = sem_cls_conf.float().sigmoid().max(dim=-1)[0].unsqueeze(1)
        H_up = (torch.div(2*H, self.n_query_bins, rounding_mode='floor') + 1) * self.n_query_bins
        W_up = (torch.div(2*W, self.n_query_bins, rounding_mode='floor') + 1) * self.n_query_bins

        # upsample the class-aware activation map into the shape, which can be divided by the grid
        max_score = F.interpolate(max_score, size=(H_up, W_up), mode='bilinear')
        max_score_cut = rearrange(max_score, 'B 1 (h r) (w t) -> (B h w) (r t)',
                                  h=self.n_query_bins, w=self.n_query_bins)
        selected_idx = max_score_cut.argmax(dim=-1)  # BThw

        # find the pixel indexes of selected queries
        query_selection_idx_cut = torch.arange(H_up*W_up, device=sem_cls_conf.device).reshape(H_up, W_up)
        query_selection_idx_cut = repeat(query_selection_idx_cut, '(h r) (w t) -> (B h w) (r t)',
                                         B=BT, h=self.n_query_bins, w=self.n_query_bins)
        query_idx = query_selection_idx_cut[torch.arange(len(selected_idx)), selected_idx].reshape(BT, -1)  # BTxQ
        query_x = torch.fmod(query_idx, W_up) / W_up
        query_y = torch.div(query_idx, W_up) / H_up
        query_coords_init = torch.stack([query_x, query_y], dim=-1)  # BTxQx2
        
        return query_coords_init

    def inter_frame_query_association(self, query_init, query_init_coords, query_init_embed):
        """
        Associate object queries across frames in a clip, thereby maintaining the temporal consistency of queries.
        Args:
                   query_init: BTxQxC
            query_init_coords: BTxQx2
            query_init_embed: BTxQxC_track
        """
        BT, Q, C = query_init.shape
        n_frames = self.n_frames if self.training else BT

        if n_frames == 1:
            return query_init, query_init_coords
        else:
            B = int(BT / n_frames)
            ct = int((n_frames - 1) / 2)

            # view the central frame as key frame
            query_init_embed = rearrange(query_init_embed, '(B T) Q C -> B T Q C', T=n_frames)

            sim_masked = []
            # only consider the wxw grid as the matched window (temporal consistency in a short clip)
            w = self.window_inter_frame_asso if self.training else self.window_inter_frame_asso / 2
            cos_sim = torch.einsum('btqc,bkc -> btqk', query_init_embed,
                                   query_init_embed[:, ct])  # BxTxQxK
            for t in range(n_frames):
                itv = max(t - ct, ct - t)
                mask_t = repeat((self.query_relpos_grid > w*itv).any(dim=-1), 'q k -> b q k', b=B)
                sim_masked.append(cos_sim[:, t].masked_fill_(mask_t, float('-inf')).softmax(dim=-2))

            aligned_idx = torch.stack(sim_masked, dim=1).flatten(0, 1).argmax(dim=-2)  # BTxQxK -> BTxK
            aligned_query_init = torch.stack([queries[idx] for queries, idx in zip(query_init, aligned_idx)])
            aligned_query_init_coords = torch.stack([points[idx] for points, idx in zip(query_init_coords, aligned_idx)])

            return aligned_query_init, aligned_query_init_coords

    def query_initialization(self, encoded_feat, targets, spatial_shapes, lvl_start_index, is_training=True):
        """
        Embedding initialization for object queries includes two steps:
            1) grid-guided query selection,
            2) inter-frame query association
        """
        reference_points = torch.cat([make_reference_points(shape, device=encoded_feat.device)
                                      for shape in spatial_shapes])  # Nx2

        # -------------------------- object query initialization -------------------------------
        # Feed the top level of encoded features into MLP to output the score map
        H, W = spatial_shapes[self.rpn_level]
        start_idx, end_idx = lvl_start_index[self.rpn_level], lvl_start_index[self.rpn_level + 1]
        rpn_reference_points = reference_points[start_idx:end_idx]  # NxC
        rpn_encoded_feat = encoded_feat[:, start_idx:end_idx]  # BTxHWxC
        rpn_cls_conf = rearrange(self.rpn_cls_embed(rpn_encoded_feat),
                                 'B (H W) K -> B H W K', H=H, W=W)  # BTxHxWxK

        # a) Grid-guided query selection for the initial coordinates and features
        query_init_coords = self.grid_guided_query_selection(rpn_cls_conf)  # BTxQx2
        query_init_coords_grid = rearrange(query_init_coords, 'B (h w) k -> B h w k', h=self.n_query_bins)
        # Extract features of the selected point coordinates. NOTE that the grid should be in the range of [-1, 1],
        # where [-1, -1] is the left-top point, [1, 1] is the right-bottom point.
        query_init_coords_grid = 2 * query_init_coords_grid - 1
        query_init = []
        for l, (H_l, W_l) in enumerate(spatial_shapes):
            query_init.append(F.grid_sample(rearrange(encoded_feat[:, lvl_start_index[l]:lvl_start_index[l+1]],
                                                      'B (H W) C -> B C H W', H=H_l),
                                            query_init_coords_grid,
                                            mode='bilinear',
                                            padding_mode="border",
                                            align_corners=False))  # BTxCxQ^.5xQ^.5
        query_init = rearrange(torch.stack(query_init).mean(0), 'B C h w -> B (h w) C')  # BTxQxC
        
        # b) Inter-frame query association: mapping query_init in the embedding space to associate queries across frames
        query_init_embed = self.track_embed(query_init)  # BTxQxC_track
        query_init, query_init_coords = self.inter_frame_query_association(query_init,
                                                                           query_init_coords,
                                                                           query_init_embed)

        if self.training:
            # During training, select the positive samples that belong to objects for initialization loss:
            # tgt_aligned_output => (tgt_cls_labels, tgt_cls_dist, tgt_ids)
            tgt_aligned_output = self.clip_peak_matcher(targets, rpn_reference_points, (H, W))
            query_init_id = F.grid_sample(rearrange(tgt_aligned_output[2], 'B T (H W) -> (B T) 1 H W', H=H).float(),
                                          query_init_coords_grid,
                                          mode='nearest', padding_mode="border",
                                          align_corners=False)  # BTx1xhxw

            output_init = {
                'rpn_sem_cls': rearrange(rpn_cls_conf, '(B T) H W K -> B T H W K', T=self.n_frames),
                'query_init_embed': rearrange(query_init_embed, '(B T) Q C -> B T Q C', T=self.n_frames),
                'query_init_inst_id': query_init_id.reshape(-1, self.n_frames, self.n_query).long()  # BxTxQ
                }

        else:
            output_init = None
            tgt_aligned_output = None

        return query_init, query_init_coords, output_init, tgt_aligned_output

    def forward(self, encoded_feat, encoded_feat_padding_masks, spatial_shapes, targets=None):
        # encoded_feat: BTxNxC, N = \sum{HW}
        # encoded_feat_padding_masks: BTxN
        # padding_masks: BxTxN

        lvl_start_index = torch.cat([spatial_shapes.new_zeros(1),
                                     spatial_shapes.prod(-1).cumsum(0)]).long()

        # -------------------------------- Query initialization ---------------------------
        #        query: BTxQxC
        # query_coords: BTxQx2
        query, query_coords, output_init, tgt_aligned_output = self.query_initialization(
            encoded_feat, targets, spatial_shapes, lvl_start_index, self.training)

        # --------------------------------------- Decoder ---------------------------------
        # Fed the initialized object queries into Decoder to output the instance query (denoted query_clip)
        query, clip_query, boxes = self.decoder(query, query_coords, encoded_feat, spatial_shapes,
                                                encoded_feat_padding_masks)

        n_frames = self.n_frames if self.training else boxes.shape[1]
        boxes = box_cxcywh_to_xyxy(rearrange(boxes, 'L (B T) Q C -> L B Q T C', T=n_frames))

        if self.training:
            output = {
                'cls': self.cls_embed(self.decoder_norm(clip_query)),
                'boxes': boxes,
                'mask_coeff': self.mask_embed(self.decoder_norm(clip_query)).tanh(),
            }  # LxBxQxK

            encoded_srcs = [rearrange(encoded_feat[:, lvl_start_index[lf]:lvl_start_index[lf + 1]],
                                      'B (H W) C -> B C H W', H=spatial_shapes[lf, 0])
                            for lf in range(spatial_shapes.shape[0])]
            output['proto'] = self.mask_head(encoded_srcs[2], [encoded_srcs[1], encoded_srcs[0]], self.training)

            outputs_all = {'query_init': output_init, 'query': output}

            return outputs_all, tgt_aligned_output
        else:

            if self.is_coco:
                output = {'cls': self.cls_embed(self.decoder_norm(clip_query[-1])).sigmoid()}

                encoded_srcs = [rearrange(encoded_feat[:, lvl_start_index[lf]:lvl_start_index[lf + 1]],
                                          'B (H W) C -> B C H W', H=spatial_shapes[lf, 0])
                                for lf in range(spatial_shapes.shape[0])]
                mask_feats = self.mask_head(encoded_srcs[2], [encoded_srcs[1], encoded_srcs[0]])
                mask_coeff = self.mask_embed(self.decoder_norm(clip_query[-1])).tanh()  # BxQxM
                output['masks'] = torch.einsum('bqm, bmthw -> bqthw', mask_coeff, mask_feats)  # BxQxTxHxW

            else:

                output = {
                    'cls': self.cls_embed(self.decoder_norm(clip_query[-1])).sigmoid(),
                    'mask_coeff': self.mask_embed(self.decoder_norm(clip_query[-1])).tanh(),
                    'query_embed': clip_query[-1],    # BxQxC before norm
                    }

            return output


class DecoderDefAttnLayer(nn.Module):
    r""" Decoder layer with self-attn, spatial-cross-attn, temporal-cross-attn and FFN
    by multiscale deformable attention module.
    This part is built on the decoder architecture of SeqFormer (https://arxiv.org/abs/2112.08275).
    Args:
        dim (int): Number of input channels.
        n_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, n_heads, fpn_levels=3, n_frames=1, n_points=3, mlp_ratio=4., drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pred_offsets=True, use_tca=False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.n_frames = n_frames
        self.use_tca = use_tca

        # ------------------------------ box-level --------------------------------------
        # self-attn
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.dropout1 = nn.Dropout(drop)
        self.norm1 = norm_layer(dim)

        # cross-attn
        self.cross_attn = MSAttnBlock(dim, n_points=n_points, n_heads=n_heads,
                                      n_levels=fpn_levels, n_frames=n_frames,
                                      pred_offsets=pred_offsets, mode='spatial')
        self.dropout2 = nn.Dropout(drop)
        self.norm2 = norm_layer(dim)

        d_ffn = int(dim * mlp_ratio)
        self.linear1 = nn.Linear(dim, d_ffn)
        self.activation = act_layer()
        self.dropout3 = nn.Dropout(drop)
        self.linear2 = nn.Linear(d_ffn, dim)
        self.dropout4 = nn.Dropout(drop)
        self.norm3 = norm_layer(dim)

        # ------------------------------ instance-level -------------------------------
        # temporal fusion
        self.time_weights = nn.Linear(dim, 1)

        # inst-level self-attn
        self.self_attn_inst = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.dropout1_inst = nn.Dropout(drop)
        self.norm1_inst = norm_layer(dim)

        # inst-level cross-attn (temporal cross-attention in our paper)
        if self.use_tca:
            self.temp_attn_inst = MSAttnBlock(dim, n_points=n_points, n_heads=n_heads,
                                              n_levels=fpn_levels, n_frames=n_frames,
                                              pred_offsets=pred_offsets, mode='temporal')
        self.dropout2_inst = nn.Dropout(drop)
        self.norm2_inst = norm_layer(dim)

        # inst-level ffn
        self.linear1_inst = nn.Linear(dim, d_ffn)
        self.activation_inst = act_layer()
        self.dropout3_inst = nn.Dropout(drop)
        self.linear2_inst = nn.Linear(d_ffn, dim)
        self.dropout4_inst = nn.Dropout(drop)
        self.norm3_inst = norm_layer(dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ca_box(self, x, x_pos, x_ref_boxes, input_flatten, input_spatial_shapes, padding_masks=None):
        # spatial cross-attn: BTxQxC
        x2 = self.cross_attn(self.with_pos_embed(x, x_pos), x_ref_boxes,
                             input_flatten, input_spatial_shapes, padding_masks)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

    def forward_sa_box(self, x, x_pos):
        q = k = x + x_pos
        x2 = self.self_attn(q, k, value=x)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)  # BTxQxC
        return x

    def forward_ffn_box(self, x):
        x2 = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        x = x + self.dropout4(x2)
        x = self.norm3(x)
        return x

    def forward_ca_inst(self, shortcut_w, shortcut_x, x_inst, x_inst_pos, x_inst_ref_boxes,
                        input_flatten, input_spatial_shapes, padding_masks=None):
        # merge clip-level query
        #       shortcut_w: BTxQxC
        #       shortcut_x: BTxQxC
        #    input_flatten: BTxNxC
        # x_inst_ref_boxes: BTxQx4
        n_frames = self.n_frames if self.training else shortcut_w.shape[0]
        ct = int((n_frames - 1) / 2)
        t_idx_start = max(ct - int((self.n_frames - 1) / 2), 0)
        t_idx_end = ct + self.n_frames

        time_weights = self.time_weights(rearrange(shortcut_w, '(B T) Q C -> B T Q C', T=n_frames))
        shortcut_x = rearrange(shortcut_x, '(B T) Q C -> B T Q C', T=n_frames)
        x_inst2 = (F.softmax(time_weights, dim=1) * shortcut_x).sum(dim=1)  # BxTx(Q+P)xC -> Bx(Q+P)xC
        x_inst = x_inst2 if x_inst is None else x_inst

        padding_masks = rearrange(padding_masks, '(B T) N -> B T N', T=n_frames)[:, t_idx_start:t_idx_end] \
            if padding_masks is not None else None
        input_flatten = rearrange(input_flatten, '(B T) N C -> B T N C', T=n_frames)[:, t_idx_start:t_idx_end]
        if input_flatten.shape[1] < self.n_frames:
            pad_frames = self.n_frames-input_flatten.shape[1]
            if padding_masks is not None:
                padding_masks = torch.cat([padding_masks, padding_masks[:, -1:].repeat(1, pad_frames, 1)], dim=1)
            input_flatten = torch.cat([input_flatten, input_flatten[:, -1:].repeat(1, pad_frames, 1, 1)], dim=1)

        # temporal cross-attn
        if self.use_tca:
            x_inst2 = self.temp_attn_inst(self.with_pos_embed(x_inst2, x_inst_pos), x_inst_ref_boxes,
                                          input_flatten, input_spatial_shapes, padding_masks)

        x_inst = x_inst + self.dropout2_inst(x_inst2)
        x_inst = self.norm2_inst(x_inst)
        return x_inst

    def forward_sa_inst(self, x_inst, x_inst_pos):
        q_inst = k_inst = self.with_pos_embed(x_inst, x_inst_pos)
        x_inst2 = self.self_attn_inst(q_inst, k_inst, value=x_inst)[0]
        x_inst = x_inst + self.dropout1_inst(x_inst2)
        x_inst = self.norm1_inst(x_inst)
        return x_inst

    def forward_ffn_inst(self, x_inst):
        # inst-level ffn
        x_inst2 = self.linear2_inst(self.dropout3_inst(self.activation_inst(self.linear1_inst(x_inst))))
        x_inst = x_inst + self.dropout4_inst(x_inst2)
        x_inst = self.norm3_inst(x_inst)
        return x_inst

    def forward(self, x, x_pos, x_ref_boxes, x_inst, x_inst_pos, x_inst_ref_boxes,
                input_flatten, input_spatial_shapes, input_padding_masks=None):
        BT, Q, C = x.shape

        # ------------------------ box-level ---------------------------------
        # First cross-attention and then self-attention
        x = self.forward_ca_box(x, x_pos, x_ref_boxes, input_flatten, input_spatial_shapes, input_padding_masks)
        shortcut_x = x

        x = self.forward_sa_box(x, x_pos)
        x = self.forward_ffn_box(x)
        shortcut_w = x

        # --------------------- instance-level -------------------------------
        # First cross-attention and then self-attention
        x_inst = self.forward_ca_inst(shortcut_w, shortcut_x, x_inst, x_inst_pos, x_inst_ref_boxes,
                                      input_flatten, input_spatial_shapes, input_padding_masks)
        x_inst = self.forward_sa_inst(x_inst, x_inst_pos)
        x_inst = self.forward_ffn_inst(x_inst)

        return x, x_inst


class DecoderDefAttn(nn.Module):
    def __init__(self, decoder_layer, num_layers=3, bbox_embed=None, norm=None, n_frames=1,
                 point2pos_proj=None, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.point2pos_proj = point2pos_proj
        self.n_frames = n_frames
        self.bbox_embed = bbox_embed
        self.norm = norm
        self.return_intermediate = return_intermediate

    @autocast(enabled=False)
    def forward(self, x, x_ref_points, srcs, srcs_spatial_shapes, srcs_padding_mask=None):
        """
        from object queries to output instance-level query
                     x: BTxQxC
          x_ref_points: BTxQx2 or x_ref_embed BTxQxC
                  srcs: encoder output with shape BxTxNxC
                   tgt: BTxQxC
        tgt_ref_points: BTxQx2
        L: number of multiscale feature layers
        """
        BT, Q, _ = x.shape
        n_frames = self.n_frames if self.training else BT
        ct = int((n_frames-1)/2)

        x_ref_boxes = torch.cat([x_ref_points, torch.ones_like(x_ref_points) * 0.1], dim=-1)  # BTxQx4
        x_inst = rearrange(x, '(B T) Q C -> B T Q C', T=n_frames)[:, ct]
        intermediate, intermediate_boxes, intermediate_inst = [], [], []

        # Warmup layer (namely the 0-th layer)
        x_boxes_offset = self.bbox_embed(self.norm(x))  # BTxQx4
        x_boxes = (x_boxes_offset + inverse_sigmoid(x_ref_boxes)).sigmoid()
        x_ref_boxes = x_boxes.clone().detach()
        x_pos = self.point2pos_proj(x_boxes[..., :2])

        # Circumscribed boxes for an instance in the whole clip,
        # which teaches the decoder to 'see' objects in a larger receptive field
        t_idx_start = max(ct - int((self.n_frames-1)/2), 0)
        t_idx_end = ct + self.n_frames
        x_inst_ref_boxes = rearrange(x_ref_boxes, '(B T) Q C -> B Q T C', T=n_frames)[:, :, t_idx_start:t_idx_end]
        x_inst_ref_boxes = box_cxcywh_to_xyxy(x_inst_ref_boxes).clamp(min=0, max=1)
        x_inst_ref_boxes = torch.cat([x_inst_ref_boxes[..., :2].min(dim=-2)[0],
                                      x_inst_ref_boxes[..., 2:].max(dim=-2)[0]], dim=-1)  # BxQx4
        x_inst_ref_boxes = box_xyxy_to_cxcywh(x_inst_ref_boxes)
        x_inst_pos = self.point2pos_proj(x_inst_ref_boxes[..., :2])  # BxQx2

        if self.return_intermediate:
            intermediate.append(x)
            intermediate_inst.append(x_inst)
            intermediate_boxes.append(x_boxes)

        for layer in self.layers:
            x, x_inst = layer(x, x_pos, x_ref_boxes,
                              x_inst, x_inst_pos, x_inst_ref_boxes,
                              srcs, srcs_spatial_shapes, srcs_padding_mask)

            x_boxes_offset = self.bbox_embed(self.norm(x))  # BTxQx4
            x_boxes = (x_boxes_offset + inverse_sigmoid(x_ref_boxes)).sigmoid()
            x_ref_boxes = x_boxes.clone().detach()
            x_pos = self.point2pos_proj(x_boxes[..., :2])

            # Circumscribed boxes for an instance in the whole clip
            x_inst_ref_boxes = rearrange(x_ref_boxes, '(B T) Q C -> B Q T C', T=n_frames)[:, :, t_idx_start:t_idx_end]
            x_inst_ref_boxes = box_cxcywh_to_xyxy(x_inst_ref_boxes).clamp(min=0, max=1)
            x_inst_ref_boxes = torch.cat([x_inst_ref_boxes[..., :2].min(dim=-2)[0],
                                          x_inst_ref_boxes[..., 2:].max(dim=-2)[0]], dim=-1)  # BxQx4
            x_inst_ref_boxes = box_xyxy_to_cxcywh(x_inst_ref_boxes)
            x_inst_pos = self.point2pos_proj(x_inst_ref_boxes[..., :2])

            if self.return_intermediate:
                intermediate.append(x)
                intermediate_inst.append(x_inst)
                intermediate_boxes.append(x_boxes)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_inst), torch.stack(intermediate_boxes)

        return x, x_inst, x_boxes


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


