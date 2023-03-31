import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from typing import List

from detectron2.structures import Instances


class OverTracker:
    """
    This structure is to support instance tracking (long) clip by (long) clip, which is memory friendly for long videos.
     We only store the instance masks of a long clip, instead of all instance masks in the whole video.
    """

    def __init__(self, num_max_inst, num_frames, window_frames, clip_stride, num_classes, mask_dim, embed_dim,
                 image_size, device, apply_cls_thres):
        self.num_frames = num_frames
        self.window_frames = window_frames
        self.clip_stride = clip_stride
        self.num_classes = num_classes
        self.mask_dim = mask_dim
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.device = device
        self.apply_cls_thres = apply_cls_thres

        self.num_max_inst = num_max_inst
        self.num_inst = 0
        self.mem_length = window_frames + num_frames
        self.num_clips = window_frames // self.clip_stride + 2
        self.saved_idx_set = set()
        self._init_memory(is_first=True)
        
        # cost matrix params
        self.siou_match_threshold = 0.1
        self.ctt_match_threshold = 0.5
        self.beta_siou = 1
        self.beta_ctt = 1

        self.weighted_manner = True
        self.num_clip_mem_long = 15 // self.clip_stride
        self.num_clip_mem_short = max(self.num_frames, 5) // self.clip_stride
        self.weights_mem = torch.exp(torch.arange(self.num_clip_mem_long, device=self.device) * 0.25)
        self.saved_untracked_frames_mem = torch.zeros(self.num_max_inst,
                                                      dtype=torch.float, device=self.device)
        self.saved_query_embeds_mem = torch.zeros((self.num_max_inst, self.embed_dim),
                                                  dtype=torch.float, device=self.device)

    def _init_memory(self, is_first=False):
        self.num_clip = 0 if is_first else 1
        self.start_frame = 0 if is_first else self.start_frame + self.window_frames
        self.saved_idx_set.difference_update(range(self.start_frame))

        self.saved_logits = torch.zeros((self.num_clips, self.num_max_inst, self.mem_length, *self.image_size),
                                        dtype=torch.float, device=self.device)
        self.saved_valid = torch.zeros((self.num_clips, self.num_max_inst, self.mem_length),
                                       dtype=torch.bool, device=self.device)
        self.saved_cls = torch.zeros((self.num_clips, self.num_max_inst, self.num_classes),
                                     dtype=torch.float, device=self.device)
        self.saved_query_embeds = torch.zeros((self.num_clips, self.num_max_inst, self.embed_dim),
                                              dtype=torch.float, device=self.device)
        self.saved_frame_idx = range(self.start_frame, self.start_frame + self.mem_length)

    def _update_memory(self, n_clip, r_idx=None, c_idx=None, input_clip=None):
        saved_start_idx = max(min(input_clip.frame_idx) - self.start_frame, 0)
        saved_end_idx = max(input_clip.frame_idx) - self.start_frame
        start_idx = input_clip.frame_idx.index(self.saved_frame_idx[saved_start_idx])
        end_idx = input_clip.frame_idx.index(self.saved_frame_idx[saved_end_idx])

        assert len(r_idx) == len(c_idx)
        self.saved_logits[n_clip, r_idx, saved_start_idx:saved_end_idx + 1] = \
            input_clip.mask_logits[c_idx, start_idx:end_idx + 1].float()
        self.saved_valid[n_clip, r_idx, saved_start_idx:saved_end_idx+1] = True
        self.saved_cls[n_clip, r_idx] = input_clip.cls_probs[c_idx]
        self.saved_query_embeds[n_clip, r_idx] = input_clip.query_embeds[c_idx].float()

        # update mem pool
        self.saved_untracked_frames_mem += 1
        self.saved_untracked_frames_mem[r_idx] = 0
        if n_clip > 0 and self.weighted_manner:
            start_clip_idx = max(n_clip - 2, 0)
            query_embed_mem = self.saved_query_embeds[start_clip_idx:n_clip+1][:, r_idx]  # CxNxE
            w_mem = self.weights_mem[:query_embed_mem.shape[0]].reshape(-1, 1, 1)
            valid_mem = (query_embed_mem != 0).any(dim=-1)[..., None]  # CxNx1
            query_embed_mem_w = (query_embed_mem * w_mem).sum(dim=0)
            valid_mem_w = (valid_mem * w_mem).sum(dim=0).clamp(min=1)
            self.saved_query_embeds_mem[r_idx] = query_embed_mem_w / valid_mem_w  # NxE
        else:
            self.saved_query_embeds_mem[r_idx] = input_clip.query_embeds[c_idx].float()

    def _get_siou(self, saved_masks, input_masks):
        # input_masks : N_i, T, H, W
        # saved_masks : N_s, T, H, W
        # input_masks = F.interpolate(input_masks, scale_factor=0.5, mode='nearest')
        # saved_masks = F.interpolate(saved_masks, scale_factor=0.5, mode='nearest')

        input_masks = input_masks.flatten(1).gt(0.5).float()  # N_i, THW
        saved_masks = saved_masks.flatten(1).gt(0.5).float()  # N_s, THW

        input_masks = input_masks.unsqueeze(0)  #  1, N_i, THW
        saved_masks = saved_masks.unsqueeze(1)  # N_s, 1,  THW
        saved_valid = (saved_masks.any(dim=-1) & input_masks.any(dim=-1)).unsqueeze(-1)  # N_s, N_i, T, 1

        # N_s, N_i, THW
        numerator = saved_masks * input_masks
        denominator = saved_masks + input_masks - numerator

        numerator = (numerator * saved_valid).sum(-1)
        denominator = (denominator * saved_valid).sum(-1)
        siou = numerator / (denominator + 1e-6)  # N_s, N_i

        return siou

    def update(self, input_clip):
        siou_scores = None

        if self.num_inst == 0:
            matched_ID = matched_idx = list(range(input_clip.num_instance))
            self.num_inst += input_clip.num_instance
        else:

            # 1. Compute the score_mem of bi-softmax similarity: long matching + short matching
            query_embed_mem = self.saved_query_embeds_mem[:self.num_inst]
            still_appeared_long = (self.saved_untracked_frames_mem[:self.num_inst] < self.num_clip_mem_long).nonzero().reshape(-1).tolist()
            still_appeared_short = (self.saved_untracked_frames_mem[:self.num_inst] < self.num_clip_mem_short).nonzero().reshape(-1).tolist()

            scores_mem = torch.zeros(self.num_inst, input_clip.query_embeds.shape[0], device=self.device)
            scores_mem[still_appeared_long] = get_ctt_similarity(query_embed_mem[still_appeared_long],
                                                                 input_clip.query_embeds)

            scores_mem_short = get_ctt_similarity(query_embed_mem[still_appeared_short],
                                                  input_clip.query_embeds)
            scores_mem[still_appeared_short] = 0.5*(scores_mem[still_appeared_short] + scores_mem_short)

            # 2. Compute the mask iou on overlapping frames
            inter_input_idx, inter_saved_idx = [], []
            for o_i, f_i in enumerate(input_clip.frame_idx):
                if f_i in self.saved_idx_set and f_i >= self.start_frame:
                    inter_input_idx.append(o_i)
                    inter_saved_idx.append(self.saved_frame_idx.index(f_i))

            siou_scores = torch.zeros(query_embed_mem.shape[0], input_clip.query_embeds.shape[0], device=self.device)
            if len(inter_saved_idx) > 0:
                if self.beta_siou > 0:
                    i_masks = input_clip.mask_logits[:, inter_input_idx].float()
                    s_masks = self.saved_logits[:self.num_clip, :self.num_inst, inter_saved_idx]
                    s_valid = self.saved_valid[:self.num_clip, :self.num_inst].any(dim=-1)
                    s_masks = (s_masks.sum(0) / s_valid.sum(0).clamp(min=1).reshape(-1, 1, 1, 1))
                    siou_scores = self._get_siou(s_masks.sigmoid(), i_masks.sigmoid())  # N_s, N_i

            # 3. Combine score matrix
            scores = self.beta_siou * siou_scores + self.beta_ctt * scores_mem
            match_threshold = self.beta_siou * self.siou_match_threshold + \
                              self.beta_ctt * self.ctt_match_threshold
            above_thres = scores > match_threshold
            scores = scores * above_thres.float()

            row_idx, col_idx = linear_sum_assignment(scores.cpu(), maximize=True)

            matched_ID, matched_idx = [], []
            for is_above, r, c in zip(above_thres[row_idx, col_idx], row_idx, col_idx):
                if not is_above:
                    continue
                matched_idx.append(c)
                matched_ID.append(r)
                siou_scores[r, c] = -1
                scores_mem[r, c] = 0

        # Remove repeatedly-detected objects with high mask IoU
        unmatched_idx = [int(idx) for idx in range(input_clip.num_instance) if idx not in matched_idx]
        repeated_idx, repeated_siou = [], []
        for idx in unmatched_idx:
            max_matched_siou = siou_scores[:, idx].max(dim=0)[0]
            max_matched_ctt = scores_mem[:, idx].max(dim=0)[0]
            if max_matched_siou > 0.4 or max_matched_ctt > 0.6:
                repeated_idx.append(idx)
                repeated_siou.append([max_matched_siou, max_matched_ctt])

        unmatched_idx = [int(idx) for idx in range(input_clip.num_instance)
                         if idx not in matched_idx + repeated_idx and input_clip.scores[idx] > 2 * self.apply_cls_thres]

        new_assign_ID = list(range(self.num_inst, self.num_inst + len(unmatched_idx)))
        matched_ID = matched_ID + new_assign_ID
        matched_idx = matched_idx + unmatched_idx

        # Update memory
        self._update_memory(self.num_clip, matched_ID, matched_idx, input_clip)

        # Update status
        self.saved_idx_set.update(input_clip.frame_set)
        self.num_clip += 1
        self.num_inst += len(new_assign_ID)

    def get_result(self, is_last_clip=False):
        mask_logits = self.saved_logits[:self.num_clip, :self.num_inst]  # CxNxTxHxW
        valid = self.saved_valid[:self.num_clip, :self.num_inst]    # CxNxT
        cls = self.saved_cls[:self.num_clip, :self.num_inst]        # CxNxK
        query_embed = self.saved_query_embeds[:self.num_clip, :self.num_inst]  # CxNxE

        mask_logits = mask_logits.sum(dim=0) / valid.sum(dim=0).clamp(min=1)[..., None, None]  # NxTxHxW
        n_frames_valid = max(self.saved_idx_set) - self.start_frame + 1
        len_frames = self.window_frames if not is_last_clip else int(n_frames_valid)
        out_masks_logits = mask_logits[:, :len_frames]  # NxTxHxW

        valid_clip = valid.any(dim=-1)[..., None]                                      # CxNx1
        out_cls = (cls * valid_clip).sum(dim=0) / valid_clip.sum(dim=0).clamp(min=1)   # NxK
        
        # update query
        nc = min(max(3, (self.num_frames - 1) // self.clip_stride), self.num_clip)
        query_weighted = valid_clip[-nc:] * self.weights_mem[:nc].reshape(-1, 1, 1)
        query_embed = query_embed[-nc:] * query_weighted
        out_query_embed = query_embed.sum(0) / query_weighted.sum(0).clamp(min=1)

        # update memory pool for the next window
        if not is_last_clip:
            self._init_memory(is_first=False)
            self.saved_logits[0, :self.num_inst, :self.mem_length-self.window_frames] = \
                mask_logits[:self.num_inst, self.window_frames:]
            self.saved_valid[0, :self.num_inst, :self.mem_length-self.window_frames] = \
                valid[:, :self.num_inst, self.window_frames:].any(dim=0)
            self.saved_cls[0, :self.num_inst] = out_cls
            self.saved_query_embeds[0, :self.num_inst] = out_query_embed

        return out_cls, out_masks_logits


def get_ctt_similarity(saved_query_embeds, input_query_embeds):
    # input_query_embeds: N_i, E
    # saved_query_embeds: N_s, E
    feats = torch.einsum('nd,md->nm', saved_query_embeds, input_query_embeds)  # N_s, N_i
    Ns, Ni = feats.shape
    Ws = 1 if Ns > 1 else 0
    Wi = 1 if Ni > 1 else 0
    d2t_scores = feats.softmax(dim=0)
    t2d_scores = feats.softmax(dim=1)
    if Ns == 1 and Ni == 1:
        scores = 0.5 * (d2t_scores + t2d_scores)
    else:
        scores = (Ws * d2t_scores + Wi * t2d_scores) / max(Ws+Wi, 1)

    return scores


class Clips:
    def __init__(self, frame_idx: List[int], results: List[Instances]):
        self.frame_idx = frame_idx
        self.frame_set = set(frame_idx)

        self.classes = results.pred_classes
        self.scores = results.scores
        self.cls_probs = results.cls_probs
        self.mask_logits = results.pred_masks
        self.query_embeds = results.query_embeds  # NxC

        self.num_instance = len(self.scores)

