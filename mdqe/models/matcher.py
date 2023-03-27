import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..util.box_ops import video_generalized_box_iou, box_xyxy_to_cxcywh


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.float()
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    thw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + \
           torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / thw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def get_in_boxes_info(boxes, gt_boxes, expanded_strides=32):
    #    boxes: QxTx4, xyxy
    # gt_boxes: NxTx4, xyxy

    gt_boxes_cxcy = box_xyxy_to_cxcywh(gt_boxes)
    boxes_cxcy = box_xyxy_to_cxcywh(boxes)
    anchor_center_x = boxes_cxcy[..., 0].unsqueeze(1)  # Qx1xT
    anchor_center_y = boxes_cxcy[..., 1].unsqueeze(1)

    # whether the center of each anchor is inside a gt box
    b_l = anchor_center_x > gt_boxes[..., 0].unsqueeze(0)
    b_r = anchor_center_x < gt_boxes[..., 2].unsqueeze(0)
    b_t = anchor_center_y > gt_boxes[..., 1].unsqueeze(0)
    b_b = anchor_center_y < gt_boxes[..., 3].unsqueeze(0)
    is_in_boxes = torch.stack([b_l, b_r, b_t, b_b], dim=-1).all(dim=-1)  # QxNxT
    is_in_boxes_all = is_in_boxes.any(dim=1)  # QxT
    # in fixed center
    center_radius = 2.5
    b_l = anchor_center_x > (gt_boxes_cxcy[..., 0] - (1 * center_radius / expanded_strides)).unsqueeze(0)
    b_r = anchor_center_x < (gt_boxes_cxcy[..., 0] + (1 * center_radius / expanded_strides)).unsqueeze(0)
    b_t = anchor_center_y > (gt_boxes_cxcy[..., 1] - (1 * center_radius / expanded_strides)).unsqueeze(0)
    b_b = anchor_center_y < (gt_boxes_cxcy[..., 1] + (1 * center_radius / expanded_strides)).unsqueeze(0)
    is_in_centers = torch.stack([b_l, b_r, b_t, b_b], dim=-1).all(dim=-1)  # QxNxT
    is_in_centers_all = is_in_centers.any(dim=1)  # QxT

    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
    is_in_boxes_and_center = (is_in_boxes & is_in_centers)

    return is_in_boxes_anchor, is_in_boxes_and_center

def dynamic_k_matching_idol(cost, pair_wise_ious, n_candidate_k=10):
    pair_wise_ious = pair_wise_ious.clamp(min=0)

    num_gt = cost.shape[1]
    if num_gt == 0:
        return torch.tensor([], device=cost.device, dtype=torch.int64), \
            torch.tensor([], device=cost.device, dtype=torch.int64), \
            torch.tensor([], device=cost.device, dtype=torch.int64)

    matching_matrix = torch.zeros_like(cost)  # Qxnum_gt

    # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(dim=0).int(), min=2)
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
        matching_matrix[pos_idx, gt_idx] = 1.0

    # a query should be matched with a single gt instance
    anchor_matching_gt = matching_matrix.sum(dim=1)  # Q
    if (anchor_matching_gt > 1).sum() > 0:
        # find gt for these queries with minimal cost
        _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
        matching_matrix[anchor_matching_gt > 1] *= 0
        matching_matrix[anchor_matching_gt > 1, cost_argmin] = 1

    # a gt instance needs to be matched with one query at least
    while (matching_matrix.sum(0) == 0).any() and (matching_matrix.sum(1) == 0).any() :
        matched_query_id = matching_matrix.sum(1) > 0
        cost[matched_query_id] += 100000.0
        unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
        for gt_idx in unmatch_id:
            pos_idx = torch.argmin(cost[:, gt_idx])
            matching_matrix[pos_idx, gt_idx] = 1.0
        anchor_matching_gt = matching_matrix.sum(1)  # Q
        if (anchor_matching_gt > 1).sum() > 0:  # If a query matches more than one gt
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
            matching_matrix[anchor_matching_gt > 1, cost_argmin] = 1  # keep gt with minimal cost

    # assert not (matching_matrix.sum(0) == 0).any()
    selected_query = torch.nonzero(matching_matrix.sum(dim=1) > 0).reshape(-1)
    gt_indices = matching_matrix[selected_query].max(dim=1)[1]
    assert len(selected_query) == len(gt_indices)

    return selected_query, gt_indices


class HungarianMatcher(nn.Module):
    """This class computes a one-to-many label assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_box: float = 3,
        cost_dice: float = 1,
        num_classes: int = 80,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the sigmoid_focal error of the masks in the matching cost
            cost_dice: This is the relative weight of the dice loss of the masks in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        assert cost_class != 0 or cost_dice != 0, "all costs cannot be 0"

        self.num_classes = num_classes
        self.num_cum_classes = [0] + np.cumsum(np.array(num_classes) + 1).tolist()

    @torch.no_grad()
    def forward(self, outputs, targets, return_hungarian_matcher=False):
        # We flatten to compute the cost matrices in a batch
        out_probs = outputs["cls"].float().sigmoid()   # BxQxK
        out_masks = torch.einsum('bqm, bmthw -> bqthw', outputs["mask_coeff"], outputs['proto'])  # BxQxTxHpxWp
        out_boxes = outputs["boxes"]  # BxQxTx4, xyxy

        indices = []
        for target, out_prob, out_box, out_mask in zip(targets, out_probs, out_boxes, out_masks):
            b_tgt_labels = target["labels"]
            cost_class = -out_prob[:, b_tgt_labels]

            if b_tgt_labels.nelement() == 0:
                indices.append((torch.tensor([], device=out_prob.device, dtype=torch.int64),
                                torch.tensor([], device=out_prob.device, dtype=torch.int64)))
                continue

            tgt_mask = target["match_masks"].to(out_mask)  # NxTxHpxWp
            with autocast(enabled=False):
                cost_bce = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask).to(cost_class)
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask).to(cost_class)

            cost_mask = cost_bce + cost_dice

            is_in_boxes = None
            if (~torch.isnan(out_box)).all():
                # Compute the L1 cost between boxes
                gt_box = target["boxes"]  # NxTx4, xyxy
                gt_box_wh = box_xyxy_to_cxcywh(gt_box)[..., 2:]  # NxTx2

                valid_box = (gt_box_wh > 0).all(dim=-1)
                cost_bbox_sm = torch.cdist(out_box.flatten(1), gt_box.flatten(1), p=1).to(cost_class)
                cost_bbox_giou = video_generalized_box_iou(out_box, gt_box, valid_box)
                cost_bbox = cost_bbox_sm + (1 - cost_bbox_giou)

                # Final cost matrix
                is_in_boxes, is_in_boxes_and_center = get_in_boxes_info(out_box, gt_box)
                C = self.cost_class * cost_class + self.cost_box * cost_bbox + self.cost_dice * cost_mask
            else:
                C = self.cost_class * cost_class + self.cost_dice * cost_mask

            if (torch.isnan(C) | torch.isinf(C)).any():
                C[torch.isnan(C) | torch.isinf(C)] = 1000.

            if return_hungarian_matcher or is_in_boxes is None:
                out_ind, tgt_ind = linear_sum_assignment(C.cpu())  # minimum
                out_ind = torch.as_tensor(out_ind, device=out_prob.device, dtype=torch.int64)
                tgt_ind = torch.as_tensor(tgt_ind, device=out_prob.device, dtype=torch.int64)
                indices.append((out_ind, tgt_ind))

            else:
                # opt matcher: one-to-many matching
                Cost = C + 100.0 * (~is_in_boxes).sum(dim=-1)[..., None]
                out_ind, tgt_ind = dynamic_k_matching_idol(Cost, cost_bbox_giou)  # minimize
                indices.append((out_ind, tgt_ind))


        return indices


class ClipPeakMatcher(nn.Module):
    def __init__(self, num_frames, num_classes, mask_on):
        super().__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.mask_on = mask_on

    @torch.no_grad()
    def forward(self, targets, ref_points, spatial_shapes):
        """
        frame-level matcher: Match each prior box with the ground truth box of the highest jaccard
        overlap, encode the bounding boxes, then return the matched indices corresponding to both
        confidence and location predictions.
        Args:
            targets:
                gt_boxes: (tensor) Ground truth boxes, Shape: [num_obj, T, 4].  [x1, y1, x2, y2]
                gt_labels: (tensor) All the class gt_labels for the image, Shape: [num_obj].
                gt_masks: [num_obj, T, h ,w]
            ref_points: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4] [cx,cy,w,h] or [cx, cy]
            spatial_shapes: [h, w]
        Return:
            The matched indices corresponding to 1) location and 2) confidence preds.
        """

        matched_label_list, matched_dist_list, matched_gt_id_list = [], [], []
        for target in targets:
            outs = self._match_clip(target, ref_points, spatial_shapes)
            matched_label_list.append(outs[0])
            matched_dist_list.append(outs[1])
            matched_gt_id_list.append(outs[2])

        return torch.stack(matched_label_list), torch.stack(matched_dist_list), torch.stack(matched_gt_id_list)

    @torch.no_grad()
    def _match_clip(self, target, ref_points, spatial_shapes):
        """
         N: number of ground-truth instances
         T: number of frames
         P: number of pixels HxW

        """
        gt_labels, gt_boxes, gt_ids = target['labels'], target['boxes'], target['ids']
        if self.mask_on:
            gt_masks = F.interpolate(target['masks'].float(), spatial_shapes, mode='bilinear',
                                     align_corners=False).gt(0.5)
        P = ref_points.shape[0]
        device = ref_points.device
        matched_labels = torch.ones(self.num_frames, P, device=device).long() * -1 # TxP
        matched_dist = torch.zeros(self.num_frames, P, self.num_classes, device=device)
        matched_gt_ids = torch.ones(self.num_frames, P, device=device).long() * -1  # TxP

        # Sort boxes by the area of boxes
        gt_boxes_area, area_sorted_idx = box_xyxy_to_cxcywh(gt_boxes)[..., 2:].prod(-1).mean(-1).sort()
        gt_boxes_c = box_xyxy_to_cxcywh(gt_boxes[area_sorted_idx])  # NxTx4
        gt_labels = gt_labels[area_sorted_idx].reshape(-1).long()
        gt_ids = gt_ids[area_sorted_idx]  # NxT
        if self.mask_on:
            gt_masks = gt_masks[area_sorted_idx].flatten(-2)  # NxTxHW

        # Remove objects with some error annotations with width <= 0 or height <= 0 in all frames of the input clip
        valid = gt_boxes_c[..., 2:].gt(0.).all(dim=-1).any(-1) & (gt_labels >= 0)  # N
        gt_boxes_c = gt_boxes_c[valid]  # NxTx4
        gt_labels = gt_labels[valid]    # N
        gt_ids = gt_ids[valid]          # NxT
        N_inst = gt_boxes_c.shape[0]
        if self.mask_on:
            gt_masks = gt_masks[valid]  # NxTxHW
            assert P == gt_masks.shape[-1], 'Masks should have same resolution wih selected fpn layer for matcher!'

        if valid.sum() == 0:
            return (matched_labels, matched_dist, matched_gt_ids)

        dist_normed = torch.div(gt_boxes_c[..., None, :2] - ref_points[None, None],
                                torch.clamp(gt_boxes_c[..., None, 2:], min=0.05))  # NxTxPx2
        dist = torch.sum(dist_normed ** 2, dim=-1)   # NxTxP
        for t in range(self.num_frames):
            for n in range(N_inst):
                if gt_ids[n, t] == -1:
                    continue

                sorted_dist, sorted_dist_idx = dist[n, t].sort(dim=0)   # P
                if self.mask_on:
                    inner_points = gt_masks[n, t, sorted_dist_idx]
                else:
                    # dist = 0.5**2 + 0.0**2 = 0.25 ==> inner circle of bounding boxes
                    # dist = 0.5**2 + 0.5**2 = 0.5 ==> circumscribed circle of bounding boxes
                    inner_points = sorted_dist < 0.5

                if inner_points.sum() == 0:
                    pos_topk_idx = sorted_dist_idx[:1]
                else:
                    pos_topk_idx = sorted_dist_idx[inner_points]

                matched_labels[t, pos_topk_idx] = gt_labels[n]
                matched_dist[t, pos_topk_idx, gt_labels[n]] = 1. - 2 * dist[n, t, pos_topk_idx].clamp(min=0, max=0.5)
                matched_gt_ids[t, pos_topk_idx] = gt_ids[n, t]

                dist[:, t, pos_topk_idx] = 1e9

        return (matched_labels, matched_dist, matched_gt_ids)
