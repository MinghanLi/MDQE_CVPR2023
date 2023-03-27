"""
Plotting utilities to visualize training logs.
"""
import os
import math
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pathlib import Path, PurePath
from einops import rearrange

from detectron2.utils.colormap import _COLORS

from .box_ops import box_xyxy_to_cxcywh, box_iou


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


def plot_mask_features(mask_feats, vid_id, img_id=None, save_path=None):
    """
    visualization for mask features
    Args:
        mask_feats: TxMxHxW for masks
        vid_id: video id
        img_id: image id or image name

    Returns:

    """
    if save_path is None:
        save_path = 'output/visualization/mask_features'
    save_path = '/'.join([save_path, vid_id])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for t, mask_feat in enumerate(mask_feats):
        img_id = torch.randint(2000, (1,)).tolist()[0] if img_id is None else img_id
        save_dir = '/'.join([save_path, str(img_id)+'_'+str(t)])

        # We set the number of mask features is 32 as default,
        # if you use others, please adaptively change it.
        C, H, W = mask_feat.shape
        bin = 4
        ns = max(C // (10*bin), 1)
        nh = min(C, (10*bin)) // bin
        arr_n, arr_h, arr_w = (ns, nh, bin) if C > bin else (1, 1, C)

        # plot mask_feattypes
        for n in range(arr_n):
            arr_img = np.zeros([H * arr_h, W * arr_w])
            for y in range(arr_h):
                for x in range(arr_w):
                    i = arr_h*arr_w*n + arr_w * y + x
                    mask_feat_i = mask_feat[i, :, :].cpu().numpy()
                    mask_feat_i = (mask_feat_i - np.min(mask_feat_i)) / (np.max(mask_feat_i)+1e-5)
                    arr_img[y * H:(y + 1) * H, x * W:(x + 1) * W] = mask_feat_i

            plt.imshow(arr_img)
            plt.axis('off')
            plt.title(str(n))
            print('Mask features are visualized in:', ''.join([save_dir, str(n)+'.png']))
            plt.savefig(''.join([save_dir, str(n)+'.png']))
            plt.clf()


def plot_dec_attn(model, images, file_names, pixel_mean, pixel_std, out_dir=None):
    '''
         cls_logits: NxK
    bboxes_unscaled: Nx4
          enc_embed: HxWxC
    # use lists to store the outputs via up-values
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0]
    '''

    backbone_tensor, backbone_pos = model.forward_pre_backbone(images)
    output, _, output_features = model.forward_post_backbone_deformable(backbone_tensor,
                                                                        backbone_pos, is_train=False)
    batch_enc_embeds = F.normalize(output_features['enc'], dim=-1)
    _, h, w, c = batch_enc_embeds.shape
    scale_wh = torch.as_tensor([w, h], device=batch_enc_embeds.device)
    batch_cls_logits = output['cls']         # BTxQxK
    batch_bboxes_unscaled = rearrange(output['boxes'], 'B Q T C -> (B T) Q C')  # BTxQX4

    batch_enc_track_embeds = F.normalize(output_features['enc_track'], dim=-1)  # BTxHxWxC
    batch_query_track_embeds = F.normalize(output_features['query_track_init'], dim=-1)  # BTxQxC
    attn_weights_embed = torch.einsum('bqc,bhwc-> bqhw', batch_query_track_embeds, batch_enc_track_embeds)
    batch_points_scaled = output_features['query_points_init'] * scale_wh.reshape(1, 1, -1)
    batch_points_scaled = batch_points_scaled.floor().long().cpu().numpy()

    out_dir = 'output/visualization/enc_attn/' if out_dir is None else out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for n, (im, enc_embeds, cls_logits, bboxes_unscaled) \
            in enumerate(zip(images, batch_enc_embeds, batch_cls_logits, batch_bboxes_unscaled)):
        cls_scores, cls_labels = cls_logits.max(dim=-1)  # QxK
        sorted_score, sorted_idx = cls_scores.sort(descending=True)
        iou = box_iou(bboxes_unscaled[sorted_idx], bboxes_unscaled[sorted_idx])[0]
        max_iou = torch.triu(iou, diagonal=1).max(0)[0]
        idx_out = sorted_idx[(max_iou < 0.5)]

        bboxes_scaled = bboxes_unscaled * scale_wh.reshape(1, -1).repeat(1, 2)
        bboxes_center = box_xyxy_to_cxcywh(bboxes_scaled)[:, :2].long()
        box_center_embeds = enc_embeds[bboxes_center[:, 1], bboxes_center[:, 0]]
        attn_weights = torch.einsum('nc,hwc->nhw', box_center_embeds, enc_embeds)

        keep_idx = idx_out[:6]  # Q
        bboxes_scaled = bboxes_scaled[keep_idx].floor().long().cpu().numpy()
        if keep_idx.nelement() == 0:
            continue
        fig, axs = plt.subplots(ncols=keep_idx.nelement(), nrows=3, figsize=(22, 7))
        im = F.interpolate(im.unsqueeze(0), size=(h, w)).squeeze(0)
        im = (im * pixel_std + pixel_mean).permute(1, 2, 0) / 255.
        for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep_idx, axs.T, bboxes_scaled):
            ax = ax_i[0]
            ax.imshow(attn_weights[idx].view(h,w).cpu().numpy())
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
            ax = ax_i[1]
            ax.imshow(attn_weights_embed[n, idx].view(h, w).cpu().numpy())
            cx, cy = batch_points_scaled[n, idx]
            ax.add_patch(plt.Circle((cx, cy), radius=1.5, color='blue'))
            ax.axis('off')
            ax.set_title(CLASSES[cls_labels[idx]])
            ax = ax_i[2]
            ax.imshow(im.cpu().numpy())
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color='blue', linewidth=3))
            ax.axis('off')
        fig.tight_layout()

        out_path = ''.join([out_dir, file_names[n].split('/')[-1]])
        print('imshow attention weights:', out_path)
        plt.savefig(out_path)
        plt.clf()


def plot_clip_query_initialization(score_maps, ref_points_bef, ref_points_aft, img=None, out_dir=None):
    """
    Visualize the object queries after grid-guided query selection and inter-frame query association.
          score_maps: TxHxW
      ref_points_bef: TxQx2, [x, y]
      ref_points_aft: TxQx2, [x, y]
    """

    T, h, w = score_maps.shape
    if img is not None:
        img = [F.interpolate(im[None], size=(2*h, 2*w)).squeeze(0).permute(1,2,0) for im in img]
    d = int(math.sqrt(ref_points_bef.shape[1]))
    selected_idx_h = range(1, d, 3)
    selected_idx_w = range(1, d, 2)

    ref_points_bef = rearrange(ref_points_bef, 'T (d r) k -> T d r k', d=d)[:, selected_idx_h][:, :, selected_idx_w]
    ref_points_aft = rearrange(ref_points_aft, 'T (d r) k -> T d r k', d=d)[:, selected_idx_h][:, :, selected_idx_w]
    ref_points_bef = (ref_points_bef.flatten(1, 2) * torch.as_tensor([w, h], device=score_maps.device).reshape(1, 1, -1)).floor().long()
    ref_points_aft = (ref_points_aft.flatten(1, 2) * torch.as_tensor([w, h], device=score_maps.device).reshape(1, 1, -1)).floor().long()

    ct = T // 2 if T % 2 == 1 else (T - 1) // 2
    score, score_idx = score_maps[ct, ref_points_bef[ct, :, 1], ref_points_bef[ct, :, 0]].flatten().sort(descending=True)
    n_topk = score.gt(0.25).sum() + 1
    high_score_points = score_idx[:n_topk]

    colors = _COLORS
    fig, axs = plt.subplots(ncols=T, nrows=2, figsize=(7*T, 8))
    for t, ax_i in zip(range(T), axs.T):
        ax = ax_i[0]
        if img is None:
            ax.imshow(score_maps[t].cpu().numpy())
        else:
            ax.imshow(img[t].cpu().numpy())
        ax.axis('off')
        ax.set_title(f'frame id: {t}')
        for (x, y), c in zip(ref_points_bef[t].tolist(), colors):
            ax.add_patch(plt.Circle((x, y), radius=1.5, color=(1,1,1), edgecolor=(0,0,0)))

        ax = ax_i[1]
        if img is None:
            ax.imshow(score_maps[t].cpu().numpy())
        else:
            ax.imshow(img[t].cpu().numpy())
        for (x, y), c in zip(ref_points_aft[t, high_score_points].tolist(), colors):
            ax.add_patch(plt.Circle((x, y), radius=1.5, color=c, edgecolor=(0,0,0)))
        ax.axis('off')
    fig.tight_layout()

    out_dir = 'output/visualization/query_initialization/' if out_dir is None else out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = torch.randperm(1000)[0].item()
    print('Visualized queries are saved to:', ''.join([out_dir, str(n) + '.png']))
    plt.savefig(''.join([out_dir, str(n) + '.png']))
    plt.clf()



