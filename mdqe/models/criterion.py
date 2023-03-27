# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
IFC model and criterion classes.
Modified by the authors of Video Instance Segmentation using Inter-Frame Communication Transformer.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

from ..util.misc import get_world_size, is_dist_avail_and_initialized
from ..util.box_ops import matched_boxlist_giou, video_box_iou

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_on_grid,
)


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: number of masks
    """
    inputs = inputs.float()
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1).float()
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / max(num_masks, 1)


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def interinst_dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        targets_interinst: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        targets_interinst: the union of target masks that nearby the target, NxTxHxW
        num_masks:
    """
    # remove pixels that contained in target and target_interinst at the same time
    targets_interinst = targets_interinst.gt(0.5) & (1-targets).gt(0.5)

    inputs = inputs.float()
    inputs_fg = inputs.sigmoid().flatten(1)
    inputs_bg = (-inputs).sigmoid().flatten(1)
    targets = targets.flatten(1).float()
    targets_interinst = targets_interinst.flatten(1).float()

    numerator = 2 * (inputs_fg * targets).sum(1) + (inputs_bg * targets_interinst).sum(1)
    denominator = inputs_fg.sum(1) + targets.sum(1) + targets_interinst.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / max(num_masks, 1)

interinst_dice_loss_jit = torch.jit.script(
    interinst_dice_loss
)  # type: torch.jit.ScriptModule

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the number of masks

    Returns:
        Loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / max(num_masks, 1)


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def interinst_sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        targets_interinst: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        targets_interinst: inter-instance targets, including itself, shape: NxTxHpxWp
        num_masks: the number of masks

    Returns:
        Loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    targets_interinst = targets_interinst.flatten(1)

    # alpha, alpha_o = 2, other as 1
    weights = targets_interinst + 1
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = (loss * weights).sum(1) / weights.sum(1).clamp(min=1)

    return loss.sum() / max(num_masks, 1)


interinst_sigmoid_ce_loss_jit = torch.jit.script(
    interinst_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        no_obj_weight: float = 0.1,
        alpha: float = 0.25,
        gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        no_obj_weight: weight of no object queries
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    inputs = inputs.float()
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    no_obj = (targets == 0).all(dim=-1)
    is_obj = (targets > 0).any(dim=-1)
    weight = is_obj.float() + no_obj_weight * no_obj.float()

    return (loss.sum(dim=-1) * weight).sum() / weight.sum().clamp(min=1)


sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule


def weighted_sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        dist_weight: torch.Tensor,
        num_boxes: torch.Tensor,
        alpha: float = 2,
        gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape. (B, P, K)
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).  (B, P, K)
        dist_weight: weights that defined by the distance from the pixel to the object center  (B, P, K)
        num_boxes: number of matcher boxes in mini-batch  (B)
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    inputs = inputs.float()
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = (1 - prob) * targets + prob * (1 - targets)
    loss = ce_loss * (p_t ** alpha)

    # Set higher weights for pixels that are closer to the bounding box center
    gamma_t = dist_weight * targets + (1 - dist_weight) * (1 - targets)
    loss = loss * (gamma_t ** gamma)
    loss = loss.sum(dim=(-2, -1)) / num_boxes

    return loss.mean()


weighted_sigmoid_focal_loss_jit = torch.jit.script(
    weighted_sigmoid_focal_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        weight (Tensor): A tensor of shape (R, 1, ...) for weighted value
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def calculate_uncertainty_sigmoid(logits, gt_one_hot_labels):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, H, W, K) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        gt_one_hot_labels: (R, H, W, K)
    Returns:
        scores (Tensor): A tensor of shape (R, H, W) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    num_classes = logits.shape[-1]

    class_logits = logits.clone().float().sigmoid()
    uncertainty = num_classes * (1 - class_logits) * gt_one_hot_labels + class_logits * (1 - gt_one_hot_labels)

    return uncertainty.sum(dim=-1)


class SetCriterion(nn.Module):
    """ This class computes the loss for IFC.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth masks and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, box and mask)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, n_frames, n_queries,
                 window_inter_frame_asso=5, interinst_mask_loss_enabled=True, interinst_mask_threshold=0.1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object queries
            n_frames: the number of frames in each input clip
            n_queries: the number of queries, default as 196 (14 patches x 14 patches)
            window_inter_frame_asso: window size of inter-frame query association
            interinst_mask_loss_enabled: enable the proposed inter-instance mask repulsion loss
            interinst_mask_threshold: a threshold controls the number of the nearby non-target instances.
            With the threshold increases, the number will decrease.
        """
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.n_frames = n_frames
        self.n_queries = n_queries
        self.window_inter_frame_asso = window_inter_frame_asso
        self.interinst_mask_loss_enabled = interinst_mask_loss_enabled
        self.interinst_mask_threshold = interinst_mask_threshold

        # used in semantic segmentation loss to encourage that the pixels of objects have a higher response
        self.num_points = 12544

        # only consider the w*w grid as matched area (temporal consistency in a short clip)
        self.n_query_bins = int(math.sqrt(self.n_queries))
        i, j = torch.meshgrid(torch.arange(self.n_query_bins), torch.arange(self.n_query_bins))
        indices = torch.stack([j, i], dim=-1).view(-1, 2)
        query_relpos_grid = (indices[:, None] - indices[None]).abs()  # QxQx2
        self.register_buffer("query_relpos_grid", query_relpos_grid)

    def forward(self, outputs_all, targets):
        outputs, tgt_matched = outputs_all

        # Two loss items of query initialization
        losses = self.forward_query_initialization(outputs['query_init'], tgt_matched)

        # Class, box and mask losses with hungarian matching
        losses_query = self.forward_hungarian_loss(outputs['query'], targets)
        losses.update(losses_query)

        for k, v in losses.items():
            if k in self.weight_dict:
                losses[k] = self.weight_dict[k] * v
            elif k[:-2] in self.weight_dict:
                losses[k] = self.weight_dict[k[:-2]] * v
            else:
                losses[k] = 0.5 * v

        return losses

    def forward_hungarian_loss(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        device = outputs['cls'].device

        # Retrieve the matching between the outputs of the last layer and the targets
        outputs_without_aux = {k: v[-1] if k not in {'proto'} else v for k, v in outputs.items()}
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target masks across all nodes, for normalization purposes
        num_masks = sum([len(i) for (_, i) in indices])
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = self.get_loss(outputs_without_aux, targets, indices, num_masks)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        for l in range(outputs['cls'].shape[0] - 1):
            aux_outputs = {k: v[l] if k not in {'proto'} else v for k, v in outputs.items()}
            indices_l = self.matcher(aux_outputs, targets)

            # Compute the average number of target masks across all nodes, for normalization purposes
            num_masks = sum([len(i) for (_, i) in indices_l])
            num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_masks)
            num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

            l_dict = self.get_loss(aux_outputs, targets, indices_l, num_masks)
            l_dict = {k + f'_{l}': v for k, v in l_dict.items()}
            losses.update(l_dict)

        return losses

    def get_loss(self, outputs, targets, indices, num_masks, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }

        losses = {}
        for k, loss_function in loss_map.items():
            losses.update(loss_function(outputs, targets, indices, num_masks, **kwargs))
        return losses

    def loss_labels(self, outputs, targets, indices, num_masks):
        """
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_tgt_masks]
        """
        src_logits = outputs['cls']  # BxQxK

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = F.one_hot(target_classes_o, num_classes=self.num_classes).to(src_logits)

        loss_cls = sigmoid_focal_loss_jit(src_logits.flatten(0, 1),
                                          target_classes.flatten(0, 1),
                                          self.eos_coef)

        return {'loss_cls': loss_cls}

    def loss_boxes(self, outputs, targets, indices, num_masks):
        """
        shape: BCxQxTx4
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (x,y,x,y), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # NxTx4
        T = target_boxes.shape[1]
        valid = ((target_boxes[..., 2:] - target_boxes[..., :2]) > 0).all(dim=-1).flatten(0, 1)  # NT

        src_boxes = outputs['boxes'][idx]  # NxTx4
        src_boxes = src_boxes.flatten(0, 1)  # NTx4
        target_boxes = target_boxes.to(src_boxes).flatten(0, 1)  # NTx4

        return {'loss_bbox': F.l1_loss(src_boxes[valid], target_boxes[valid], reduction='sum') / (T * num_masks),
                'loss_giou': (1 - matched_boxlist_giou(src_boxes[valid], target_boxes[valid])).sum() / (T * num_masks)}

    def loss_masks(self, outputs, targets, indices, num_masks):
        """
        shape: mask coefficient: BCxQxM, mask features: BCxMxTxHxW
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_tgt_masks, h, w]
        """
        idx = self._get_src_permutation_idx(indices)
        batch_idx, src_idx = idx

        src_masks = torch.einsum('bqm, bmthw -> bqthw', outputs["mask_coeff"], outputs['proto'])[idx]  # NxTxHpxWp
        tgt_masks = torch.cat([t['match_masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)  # NxTxHpxWp

        if len(src_idx) == 0:
            return {
                "loss_mask": sigmoid_ce_loss(src_masks, tgt_masks, num_masks),
                "loss_dice": dice_loss(src_masks, tgt_masks, num_masks),
            }

        if self.interinst_mask_loss_enabled:
            with torch.no_grad():
                # to obtain inter-instance ground-truth masks
                tgt_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)])  # NxTx4
                tgt_boxes_wh = (tgt_boxes[..., 2:] - tgt_boxes[..., :2]).clamp(min=0.05)
                tgt_boxes_xy = 0.5 * (tgt_boxes[..., 2:] + tgt_boxes[..., :2])
                tgt_boxes = torch.cat([tgt_boxes_xy - 0.5 * tgt_boxes_wh,
                                       tgt_boxes_xy + 0.5 * tgt_boxes_wh], dim=-1)
                box_iou = video_box_iou(tgt_boxes, tgt_boxes)[0].max(dim=-1)[0]  # NxNxT -> NxN
                is_same_clip = (batch_idx[:, None] == batch_idx[None]).to(src_masks)  # NxN
                box_iou = box_iou * is_same_clip  # NxN

                tgt_interinst_masks = torch.stack([tgt_masks[iou > self.interinst_mask_threshold].gt(0.5).any(dim=0)
                                                   for iou in box_iou]).float()  # NxTxHpxWp

            return {
                "loss_mask": interinst_sigmoid_ce_loss_jit(src_masks, tgt_masks, tgt_interinst_masks, num_masks),
                "loss_dice": interinst_dice_loss_jit(src_masks, tgt_masks, tgt_interinst_masks, num_masks),
            }
        else:

            return {
                "loss_mask": sigmoid_ce_loss_jit(src_masks, tgt_masks, num_masks),
                "loss_dice": dice_loss_jit(src_masks, tgt_masks, num_masks),
            }
        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward_query_initialization(self, outputs, tgt_matched):
        # two loss items of query initialization: semantic segmentation loss and contrastive learning loss
        rpn_cls_tgt_labels, rpn_cls_tgt_dist, _ = tgt_matched

        # semantic segmentation loss
        sem_class_loss = self.loss_labels_query_init(outputs['rpn_sem_cls'], rpn_cls_tgt_labels, rpn_cls_tgt_dist)

        # contrastive learning loss
        ctt_loss, aux_loss = self.loss_reid_query_init(outputs['query_init_embed'], outputs['query_init_inst_id'])

        return {'loss_sem_cls_query_init': sem_class_loss,
                'loss_reid_query_init': ctt_loss, 'loss_reid_query_init_aux': aux_loss}

    def loss_labels_query_init(self, rpn_cls_out_logits, rpn_cls_tgt_labels, rpn_cls_tgt_dist):
        B, T, H, W, K = rpn_cls_out_logits.shape
        rpn_cls_tgt_labels = rpn_cls_tgt_labels.reshape(-1)  # BTHW
        rpn_pos_idxs = torch.nonzero(rpn_cls_tgt_labels >= 0, as_tuple=False)

        with torch.no_grad():
            # prepare one_hot gt labels
            rpn_cls_tgt = torch.zeros_like(rpn_cls_out_logits.flatten(0, -2))  # BTHWxK
            rpn_cls_tgt[rpn_pos_idxs, rpn_cls_tgt_labels[rpn_pos_idxs]] = 1.
            rpn_cls_tgt = rpn_cls_tgt.reshape(B*T, H, W, K)

        rpn_cls_tgt_dist = rpn_cls_tgt_dist.reshape(B*T, H, W, K)
        rpn_cls_out_logits = rpn_cls_out_logits.flatten(0, 1)  # BxTxHxWxK -> BTxHxWxK

        uncertainty_map = calculate_uncertainty_sigmoid(rpn_cls_out_logits, rpn_cls_tgt)  # BTxHxW

        with torch.no_grad():
            # sample point_indices: BT x 12544, point_coords: BT x 12544 x 2
            point_indices, point_coords = get_uncertain_point_coords_on_grid(
                uncertainty_map.unsqueeze(1),
                self.num_points
            )
            # get gt labels
            batch_idx = torch.arange(point_indices.shape[0]).reshape(-1, 1).repeat(1, point_indices.shape[1])
            point_labels = rpn_cls_tgt.reshape(B*T, H*W, K)[batch_idx, point_indices]  # BTx12544xK
            point_dist = rpn_cls_tgt_dist.reshape(B*T, H*W, K)[batch_idx, point_indices]  # BTx12544xK

        point_logits = rpn_cls_out_logits.reshape(B*T, H*W, K)[batch_idx, point_indices]  # BTx12544xK

        num_boxes = point_labels.any(dim=-1).sum(dim=1).clamp(min=1)  # BT
        rpn_class_loss = weighted_sigmoid_focal_loss_jit(
            point_logits,
            point_labels,
            point_dist,
            num_boxes
        )

        return rpn_class_loss

    @autocast(enabled=False)
    def loss_reid_query_init(self, query_embeds, query_inst_IDs):
        aux_loss, ctt_loss = [], []
        B, T, Q, E = query_embeds.shape
        B, T, Q = query_inst_IDs.shape

        for bs, (init_embeds, inst_IDs) in enumerate(zip(query_embeds, query_inst_IDs)):
            # inst_IDs: -1 background, >= 0 ground-truth instance ids
            inst_IDs_unique = inst_IDs[inst_IDs >= 0].unique()
            inst_idxs_unique = ((inst_IDs_unique[:, None] - inst_IDs.flatten()[None]) == 0).float().argmax(dim=-1)  # U

            for inst_idx, inst_ID in zip(inst_idxs_unique, inst_IDs_unique):
                inst_IDs = inst_IDs.flatten()  # TQ
                init_embeds = init_embeds.reshape(T*Q, -1)  # TQ x E

                frame_idx = torch.div(inst_idx, Q, rounding_mode='trunc')
                w = max(self.window_inter_frame_asso, 2)
                fired_area = torch.stack([
                    (self.query_relpos_grid[:, inst_idx % Q] <= (w * (torch.abs(t - frame_idx) + 1))).all(dim=-1)
                    for t in range(T)
                ]).flatten()  # TQ

                if (inst_IDs[fired_area] == inst_ID).all():
                    fired_area = torch.ones_like(inst_IDs)

                target_embeds = init_embeds[inst_idx]  # E
                fired_inst_IDs = inst_IDs[fired_area]  # N
                fired_init_embeds = init_embeds[fired_area]  # N x E

                same_objs = fired_inst_IDs == inst_ID  # N
                diff_objs = fired_inst_IDs != inst_ID  # N
                same_embeds = fired_init_embeds[same_objs]  # N_pos x E
                diff_embeds = fired_init_embeds[diff_objs]  # N_neg x E

                n_dynk_neg = min(diff_objs.sum(), 50 * T)

                # select pos samples with dynamic topk
                n_dynk_pos = max(int(n_dynk_neg / 5), 2)
                _dynamic_k_pos_idx = torch.randperm(same_objs.sum())[:n_dynk_pos]
                _same_embeds_pos = same_embeds[_dynamic_k_pos_idx]
                _pos_embed = torch.einsum('ke,e->k', _same_embeds_pos, target_embeds)
                # select neg samples
                _dynamic_k_neg_idx = torch.randperm(diff_objs.sum())[:n_dynk_neg]
                _diff_embeds_neg = diff_embeds[_dynamic_k_neg_idx]
                _neg_embed = torch.einsum('qe,e->q', _diff_embeds_neg, target_embeds)

                _negpos_embed = _neg_embed[:, None] - _pos_embed[None]  # N_neg x N_pos
                ctt_loss.append(torch.log(1 + _negpos_embed.exp().sum(dim=0).clamp(max=1e3)).mean().reshape(-1))

                labels = torch.cat([_pos_embed.new_ones(len(_dynamic_k_pos_idx)),
                                    _neg_embed.new_zeros(len(_dynamic_k_neg_idx))])
                track_embeds = torch.cat([_same_embeds_pos, _diff_embeds_neg])
                random = torch.randperm(track_embeds.shape[0])
                aux_cosine = torch.einsum('e,ke->k',
                                          F.normalize(target_embeds, dim=-1),
                                          F.normalize(track_embeds, dim=-1))
                aux_loss.append((torch.abs(aux_cosine[random] - labels[random]) ** 2).mean().reshape(-1))

        if len(ctt_loss) == 0:
            loss = torch.tensor(0.).to(query_embeds)
            return loss, loss
        else:
            return sum(ctt_loss) / len(ctt_loss), sum(aux_loss) / len(ctt_loss)