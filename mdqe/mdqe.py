# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .models.backbone import Joiner
from .models.position_encoding import PositionEmbeddingSine
from .models import mdqe, SetCriterion, Transformer_Enc, Transformer_Dec, HungarianMatcher, ClipPeakMatcher
from .tracking import Clips, OverTracker
from .util.misc import NestedTensor, interpolate, aligned_bilinear
from .util.box_ops import box_iou, box_xyxy_to_cxcywh

__all__ = ["MDQE"]


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = [backbone_shape[f].channels for f in backbone_shape.keys()]

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class MDQE(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.size_divisibility = 32

        self.n_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.is_coco = cfg.DATASETS.TEST[0].startswith("coco")
        self.is_ovis = 'ovis' in cfg.DATASETS.TEST[0]
        self.num_classes = cfg.MODEL.MDQE.NUM_CLASSES
        self.mask_stride = cfg.MODEL.MDQE.MASK_STRIDE
        self.match_stride = cfg.MODEL.MDQE.MATCH_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON
        self.hidden_dim = cfg.MODEL.MDQE.HIDDEN_DIM
        num_queries = cfg.MODEL.MDQE.NUM_OBJECT_QUERIES
        num_queries = int(math.sqrt(num_queries)) ** 2

        # grid-guided query selection and inter-frame query association
        # control the window size across frames, default as 5
        self.window_inter_frame_asso = cfg.MODEL.MDQE.WINDOW_INTER_FRAME_ASSOCIATION
        self.query_embed_dim = cfg.MODEL.MDQE.QUERY_EMBED_DIM
        self.interinst_mask_threshold = cfg.MODEL.MDQE.INTERINST_MASK_THRESHOLD
        self.interinst_mask_loss_enabled = cfg.MODEL.MDQE.INTERINST_MASK_LOSS_ENABLED

        # Transformer parameters:
        nheads = cfg.MODEL.MDQE.NHEADS
        dropout = cfg.MODEL.MDQE.DROPOUT
        self.enc_layers = cfg.MODEL.MDQE.ENC_LAYERS
        self.dec_layers = cfg.MODEL.MDQE.DEC_LAYERS
        num_feature_levels = cfg.MODEL.MDQE.NUM_FEATURE_LEVELS
        self.dec_n_points = cfg.MODEL.MDQE.DEC_NUM_POINTS
        self.enc_n_points = cfg.MODEL.MDQE.ENC_NUM_POINTS
        self.dec_temporal = cfg.MODEL.MDQE.DEC_TEMPORAL
        self.mlp_ratio = cfg.MODEL.MDQE.MLP_RATIO

        # Loss parameters:
        box_weight = cfg.MODEL.MDQE.BOX_WEIGHT
        mask_weight = cfg.MODEL.MDQE.MASK_WEIGHT
        dice_weight = cfg.MODEL.MDQE.DICE_WEIGHT
        deep_supervision = cfg.MODEL.MDQE.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MDQE.NO_OBJECT_WEIGHT

        N_steps = int(self.hidden_dim / 2)
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels
        backbone.feature_strides = d2_backbone.feature_strides

        # building peak matcher for semantic loss of query initialization
        clip_peak_matcher = ClipPeakMatcher(self.n_frames, self.num_classes, mask_on=self.mask_on)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=1,
            cost_box=box_weight,
            cost_dice=dice_weight,
            num_classes=self.num_classes
        )

        fpn_dims = [self.hidden_dim, self.hidden_dim]
        transformer_enc = Transformer_Enc(
            dim=self.hidden_dim,
            n_heads=nheads,
            n_feature_levels=num_feature_levels,
            n_enc_points=self.enc_n_points,
            n_enc_layers=self.enc_layers,
            n_frames=self.n_frames,
        )

        transformer_dec = Transformer_Dec(
            self.num_classes,
            dim=self.hidden_dim,
            n_heads=nheads,
            n_feature_levels=num_feature_levels,
            n_frames=self.n_frames,
            n_dec_points=self.dec_n_points,
            n_dec_layers=self.dec_layers,
            mlp_ratio=self.mlp_ratio,
            dec_temporal=self.dec_temporal,
            n_query=num_queries,
            fpn_dims=fpn_dims,
            clip_peak_matcher=clip_peak_matcher,
            window_inter_frame_asso=self.window_inter_frame_asso,
            query_embed_dim=self.query_embed_dim,
            is_coco=self.is_coco,
            mask_on=self.mask_on,
        )

        self.detr = mdqe(
            backbone, transformer_enc, transformer_dec, n_frames=self.n_frames,
            num_feature_levels=num_feature_levels, aux_loss=deep_supervision,
        )
        self.detr.to(self.device)

        self.mask_dim = transformer_dec.mask_head.num_gen_params

        # building criterion
        weight_dict = {"loss_sem_cls_query_init": 2,
                       "loss_cls": 2, "loss_bbox": box_weight, "loss_giou": box_weight,
                       "loss_mask": mask_weight, "loss_dice": dice_weight}
        self.criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            n_frames=self.n_frames,
            n_queries=num_queries,
            window_inter_frame_asso=self.window_inter_frame_asso,
            interinst_mask_loss_enabled=self.interinst_mask_loss_enabled,
            interinst_mask_threshold=self.interinst_mask_threshold,
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        # eval hyper-parameters
        self.clip_stride = cfg.MODEL.MDQE.CLIP_STRIDE
        self.merge_on_cpu = cfg.MODEL.MDQE.MERGE_ON_CPU
        self.merge_device = "cpu" if self.merge_on_cpu else self.device
        self.is_multi_cls = cfg.MODEL.MDQE.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.MDQE.APPLY_CLS_THRES
        self.detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.n_frames_test = cfg.MODEL.MDQE.SAMPLING_FRAME_NUM_TEST
        self.n_frames_window_test = cfg.MODEL.MDQE.WINDOW_FRAME_NUM_TEST
        self.n_max_inst = cfg.MODEL.MDQE.MAX_NUM_INSTANCES

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """

        if self.training or self.is_coco:
            images, _ = self.preprocess_image(batched_inputs)
            images = ImageList.from_tensors(images, self.size_divisibility)

            if self.training:
                gt_instances = []
                for video in batched_inputs:
                    for frame in video["instances"]:
                        gt_instances.append(frame.to(self.device))

                # print('new iter')
                targets = self.prepare_targets(batched_inputs, images)

                output = self.detr(images, targets)
                # print('model')

                loss_dict = self.criterion(output, targets)
                # print('criterion')

                return loss_dict
            else:

                output = self.detr(images)

                return self.inference_image(output, batched_inputs, images)

        else:
            
            # Youtube-VIS evaluation should be treated in a different manner.
            return self.inference_vis(batched_inputs)

    def prepare_targets(self, targets, images):
        BT, c, h_pad, w_pad = images.tensor.shape
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.n_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            gt_boxes_per_video = torch.zeros([_num_instance, self.n_frames, 4], device=self.device)

            gt_ids_per_video = []
            gt_classes_per_video = torch.full((_num_instance,), self.num_classes, device=self.device)
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                update_cls_idx = targets_per_frame.gt_classes != -1
                gt_classes_per_video[update_cls_idx] = targets_per_frame.gt_classes[update_cls_idx].long()
                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else:  # polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks
                update_box_idx = box_xyxy_to_cxcywh(targets_per_frame.gt_boxes.tensor)[..., 2:].gt(0).all(dim=-1)
                gt_boxes_per_video[update_box_idx, f_i] = targets_per_frame.gt_boxes.tensor[update_box_idx]  # xyxy

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1) & (gt_classes_per_video < self.num_classes)        # BxT
            gt_classes_per_video = gt_classes_per_video[valid_idx]  # N
            gt_ids_per_video = gt_ids_per_video[valid_idx]          # NxT

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})

            gt_masks_per_video = gt_masks_per_video[valid_idx].float()  # NXTxHxW
            m_h, m_w = math.ceil(h_pad/self.match_stride), math.ceil(w_pad/self.match_stride)
            gt_masks_for_match = interpolate(gt_masks_per_video, size=(m_h, m_w),
                                             mode="bilinear", align_corners=False)
            gt_instances[-1].update({"masks": gt_masks_per_video,
                                     "match_masks": gt_masks_for_match})

            gt_boxes_per_video = gt_boxes_per_video[valid_idx]
            x0, x1 = (gt_boxes_per_video[..., 0::2]/w_pad).unbind(-1)
            y0, y1 = (gt_boxes_per_video[..., 1::2]/h_pad).unbind(-1)
            gt_boxes_per_video = torch.stack([x0, y0, x1, y1], dim=-1).clamp(min=0, max=1)
            gt_instances[-1].update({"boxes": gt_boxes_per_video})

        return gt_instances

    def inference_vis(self, batched_inputs):
        # NOTE we assume that only a single video is taken as an input.
        # Each video is processing clip by clip, making it memory friendly.
        video_name = batched_inputs[0]['file_names'][0].split('/')[-2]
        video_frames, video_length = self.preprocess_image(batched_inputs)

        image_size = video_frames[0].shape[-2:]  # image size without padding after data augmentation
        ori_height = batched_inputs[0].get("height", image_size[0])
        ori_width = batched_inputs[0].get("width", image_size[1])

        # Split the whole video into several long clips for memory friendly
        window_start_idx, window_end_idx = 0, 0

        saved_clips = 0
        is_last_clip = False
        video_output = None
        pred_cls_clips, pred_masks_clips = [], []
        for clip_idx, start_idx in enumerate(range(0, video_length, self.clip_stride)):
            end_idx = start_idx + self.n_frames_test
            if end_idx > video_length:
                is_last_clip = True
                end_idx = video_length

            if end_idx > window_end_idx:
                # process backbone+Encoder in a window manner,
                # which is memory-friendly for super-long videos (>200 frames)
                window_start_idx = start_idx
                window_frames = ImageList.from_tensors(video_frames[window_start_idx:start_idx + self.n_frames_window_test],
                                                       self.size_divisibility)
                window_backbone_tensor, window_backbone_pos = self.detr.forward_pre_backbone(window_frames)
                window_encoder_srcs, window_encoder_masks, spatial_shapes = \
                    self.detr.forward_deformable_enc(window_backbone_tensor,
                                                     window_backbone_pos,
                                                     is_training=False)
                window_mask_feats = self.detr.forward_mask_head_inference(window_encoder_srcs, spatial_shapes)[0]  # MxTxHpxWp
                mask_feats_size = window_mask_feats.shape[-2:]

            window_frame_idx = list(range(start_idx-window_start_idx, end_idx-window_start_idx))
            clip_encoder_srcs = window_encoder_srcs[window_frame_idx]
            clip_encoder_masks = window_encoder_masks[window_frame_idx]
            clip_mask_feats = window_mask_feats[:, window_frame_idx]
            frame_idx = list(range(start_idx, end_idx))

            output = self.detr.transformer_dec(clip_encoder_srcs, clip_encoder_masks, spatial_shapes)

            _clip_results, _valid_idx = self.inference_clip(output, clip_mask_feats, mask_feats_size)
            clip_results = Clips(frame_idx, _clip_results.to(self.merge_device))

            if video_output is None:
                # OverTracker is memory-friendly, processing instance segmentation (long) clip by (long) clip
                # where the length of long clip is controlled by self.n_frames_window_test
                video_output = OverTracker(
                    self.n_max_inst, self.n_frames_test, self.n_frames_window_test, self.clip_stride,
                    self.num_classes, self.mask_dim, self.hidden_dim, mask_feats_size,
                    self.merge_device, self.apply_cls_thres
                )
            video_output.update(clip_results)

            # Save output results clip by clip, which is memory-friendly. After inference of the video,
            # the instance masks of all clips will be directly merged into the .json file (mdqe/data/ytvis_eval.py).
            is_output = start_idx + self.clip_stride >= self.n_frames_window_test * (saved_clips + 1)
            if is_last_clip or is_output:
                pred_cls, pred_masks = video_output.get_result(is_last_clip=is_last_clip)
                if self.merge_on_cpu:
                    pred_cls, pred_masks = pred_cls.cpu(), pred_masks.cpu()
                    
                pred_masks = aligned_bilinear(pred_masks, factor=self.match_stride).sigmoid()
                pred_masks = pred_masks[..., :image_size[0], :image_size[1]]
                pred_cls_clips.append(pred_cls)
                pred_masks_clips.append(pred_masks)
                saved_clips += 1

            if is_last_clip:
                break

        return self.inference_video((ori_height, ori_width), pred_cls_clips, pred_masks_clips)

    def inference_clip(self, output, mask_feats, image_size):
        mask_cls = output["cls"][0]              # QxK
        mask_params = output['mask_coeff'][0]    # QxM
        query_embeds = output['query_embed'][0]  # QxE

        sorted_scores, sorted_idxs = mask_cls.max(-1)[0].sort(descending=True)
        valid_idx = sorted_idxs[sorted_scores >= min(self.apply_cls_thres, sorted_scores[0])]
        if valid_idx.nelement() > 1:
            query_sim = torch.mm(F.normalize(query_embeds[valid_idx], dim=-1),
                                 F.normalize(query_embeds[valid_idx], dim=-1).t())
            max_query_sim = torch.triu(query_sim, diagonal=1).max(0)[0]
            valid_idx = valid_idx[max_query_sim < 0.99][:10 * self.detections_per_image]

        mask_cls = mask_cls[valid_idx]
        mask_params = mask_params[valid_idx]
        query_embeds = query_embeds[valid_idx]
        mask_pred = torch.einsum('qm,mthw->qthw', mask_params, mask_feats)

        # remove predicted blank mask
        nonblank = mask_pred.gt(0.).flatten(1).sum(1) > 0
        mask_cls = mask_cls[nonblank]
        mask_pred = mask_pred[nonblank]
        query_embeds = query_embeds[nonblank]

        if mask_cls.nelement() > 0:
            # Just avoid running out of memory
            mask_nms = mask_pred[:, ::2] if mask_pred.shape[1] >= 5 else mask_pred
            mask_soft = F.interpolate(mask_nms, scale_factor=0.5).flatten(1).sigmoid()
            mask_hard = mask_soft.gt(0.5).float()  # QxThw

            # mask siou
            numerator = torch.mm(mask_soft, mask_hard.t())  # QxQ
            denominator = mask_soft.sum(-1)[:, None] + mask_hard.sum(-1)[None] - numerator
            siou = numerator / (denominator + 1)  # QxQ
            max_iou = torch.triu(siou, diagonal=1).max(0)[0]
            mask_cls = mask_cls * (1 - max_iou[:, None])

            valid = max_iou < 0.5
            mask_cls = mask_cls[valid]
            mask_pred = mask_pred[valid]
            query_embeds = query_embeds[valid]

        # mask-aware confidence score
        mask_soft = mask_pred.sigmoid().flatten(1)
        mask_hard = mask_soft.gt(0.5).float()
        mask_scores = (mask_soft * mask_hard).sum(1) / (mask_hard.sum(1) + 1e-6)
        mask_cls = mask_cls * mask_scores[:, None]

        scores, labels = mask_cls.max(-1)
        sorted_idxs = scores.sort(descending=True)[1]
        n_topk = max((scores > self.apply_cls_thres).sum(), 1)
        topk_idxs = sorted_idxs[:n_topk]

        results = Instances(image_size)
        results.scores = scores[topk_idxs]
        results.pred_classes = labels[topk_idxs]
        results.cls_probs = mask_cls[topk_idxs]
        results.pred_masks = mask_pred[topk_idxs]
        results.query_embeds = query_embeds[topk_idxs]

        return results, valid_idx

    def inference_video(self, image_size, pred_cls_clips, pred_masks_clips):
        total_num_insts = pred_cls_clips[-1].shape[0]
        for l, pred_cls in enumerate(pred_cls_clips):
            num_miss_insts = total_num_insts - pred_cls.shape[0]
            pred_cls_clips[l] = torch.cat([
                pred_cls, torch.zeros(num_miss_insts, pred_cls.shape[1], device=pred_cls.device)
            ])
        pred_cls_clips = torch.stack(pred_cls_clips)
        out_cls = 0.75 * pred_cls_clips.mean(0) + 0.25 * pred_cls_clips.max(0)[0]

        out_masks_video = []
        for idx in range(total_num_insts):
            m_video = [m[idx].cpu() if idx < m.shape[0] else torch.zeros_like(m[0]).cpu() for m in pred_masks_clips]
            m_video = torch.cat(m_video, dim=0)
            out_masks_video.append(m_video)

        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(out_cls.shape[0], 1).flatten(0, 1)
        out_cls = out_cls.flatten().cpu()

        num_topk = max(out_cls.gt(0.05).sum(), 10)
        out_scores, topk_indices = out_cls.topk(num_topk, sorted=False)
        out_labels = labels[topk_indices].tolist()
        out_scores = out_scores.tolist()

        topk_indices = torch.div(topk_indices, self.num_classes, rounding_mode='floor')
        # Just avoid running out of memory for longer videos (> 200 frames).
        # we first store mask results clip by clip and then merge them together when saving results in .json file.
        out_masks_list = []
        for idx in topk_indices:
            m = retry_if_cuda_oom(interpolate)(
                out_masks_video[idx].unsqueeze(0), size=image_size, mode="nearest"
            ).squeeze(0)
            out_masks_list.append(retry_if_cuda_oom(lambda x: x > 0.5)(m))

        video_output = {
            'image_size': image_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            'pred_masks': out_masks_list,
        }

        return video_output

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for idx, video in enumerate(batched_inputs):
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))

        len_vid = len(images)

        return images, len_vid

    def inference_image(self, output, batched_inputs, images):
        """
                Arguments:
                    output (Dict): The Dict predicts the classification probability and instance masks for each query.
                    batched_inputs: original inputs
                    images (List[torch.Size]): the input image sizes

                Returns:
                    results (List[Instances]): a list of #images elements.
                """
        is_NMS = True  # opt as matcher (1-to-many matcher)
        mask_cls = output["cls"][0]  # QxK
        masks = output['masks'][0]   # QxTxHxW

        # For each mask we assign the best class or the second best if the best on is `no_object`.
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        ct = int((self.n_frames-1) / 2)
        mask = masks[:, ct]
        score = mask_cls.max(-1)[0]
        idx_out = torch.nonzero(score >= min(self.apply_cls_thres, score.max())).reshape(-1)

        mask_cls = mask_cls[idx_out]
        mask = mask[idx_out]
        mask = aligned_bilinear(mask.unsqueeze(1), factor=self.match_stride).squeeze(1)
        mask = mask[:, :image_size[0], :image_size[1]]

        # mask quality based scores = cls_score * mask_score
        interim_mask_soft = mask.sigmoid()
        interim_mask_hard = interim_mask_soft > 0.5
        numerator = (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)
        denominator = interim_mask_hard.flatten(1).sum(1)
        mask_score = (numerator / (denominator + 1e-6))
        mask_cls = mask_cls * mask_score[:, None]

        if is_NMS and len(idx_out) > 0:
            resorted_idx = mask_cls.max(-1)[0].sort(descending=True)[1]
            mask_cls = mask_cls[resorted_idx]
            mask = mask[resorted_idx]

            box_normalizer = torch.as_tensor([image_size[1], image_size[0],
                                              image_size[1], image_size[0]]).reshape(1, -1)
            # produce bounding boxes according to predicted masks
            mask_box_per_clip = BitMasks(mask.gt(0.)).get_bounding_boxes().tensor
            mask_box_per_clip_norm = (mask_box_per_clip / box_normalizer).to(mask.device)
            biou = box_iou(mask_box_per_clip_norm, mask_box_per_clip_norm)[0]
            max_biou = torch.triu(biou, diagonal=1).max(0)[0]
            mask_cls = mask_cls * (1 - max_biou)[:, None]

        if self.is_multi_cls:
            ls = torch.nonzero(mask_cls > self.apply_cls_thres)
            idxs, label = ls[..., 0], ls[..., 1]
            score = mask_cls[idxs, label]
            mask = mask[idxs]
        else:
            score, label = mask_cls.max(-1)

        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])

        # up-sample to the original image size
        mask = interpolate(
            mask.float().unsqueeze(1), size=[out_height, out_width], mode='bilinear'
        ).squeeze(1) > 0.

        result = Instances((out_height, out_width))
        result.scores = score
        result.pred_classes = label
        result.pred_masks = mask
        result.pred_boxes = BitMasks(mask).get_bounding_boxes()

        return [{"instances": result}]

