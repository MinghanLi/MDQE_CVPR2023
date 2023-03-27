import os
import argparse
import torch
from einops import repeat


def parse_args():
    parser = argparse.ArgumentParser("D2 model converter")
    parser.add_argument("--convert_to_d2", default=False, type=bool,
                        help="Convert pretrained weights of Swin transformer to d2 format")
    parser.add_argument("--num_frames", default=2, type=int, help="Number of frames")
    parser.add_argument("--source_model", default="", type=str, help="Path or url to the  model to convert")
    return parser.parse_args()


def convert_to_d2(source_model):
    """
    Convert pretrained Swin Large model on ImageNet dataset to d2 format.
    """
    if os.path.splitext(source_model)[-1] != ".pth":
        raise ValueError("You should save weights as pth file")

    source_weights = torch.load(source_model, map_location=torch.device('cpu'))["model"]

    keys = list(source_weights.keys())
    converted_weights = {}
    prefix = 'detr.backbone.0.backbone.'
    for key in keys:
        print(key, source_weights[key].shape)
        converted_weights[prefix + key] = source_weights[key]

    output_model = source_model[:-4] + '_d2' + source_model[-4:]
    torch.save(converted_weights, output_model)
    print('Converted model for d2 format is used in ', output_model)

    exit()


def inflated_weights(num_frames, source_model, num_pretrain_frames=1, n_heads=8, n_points=4):
    """
    inflated weights in temporal direction from per-frame to per-clip weights.
    num_frames: the number of frames pre clip in MDQE of fine-tuning on VIS datasets
    source_model: model pretrined on COCO dataset with a single frame (or num_pretrain_frames)
    num_pretrain_frames: the number of frames used in pretraining on COCO dataset
    """
    if os.path.splitext(source_model)[-1] != ".pth":
        raise ValueError("You should save weights as pth file")

    T = max(num_frames+1 // num_pretrain_frames, 1)
    source_weights = torch.load(source_model, map_location=torch.device('cpu'))["model"]
    keys = list(source_weights.keys())
    for k in keys:
        if 'temp_attn' in k:
            if k.split('.')[-2] in {'attention_weights', 'sampling_grid_offsets'}:
                if k.split('.')[-1] == 'bias':
                    new_v = repeat(source_weights[k], '(H F K D) -> H (F T) K D',
                                   H=n_heads, F=num_pretrain_frames, T=T, K=n_points)
                    source_weights[k] = new_v[:, :num_frames].flatten()
                elif k.split('.')[-1] == 'weight':
                    new_v = repeat(source_weights[k], '(H F K D) C -> H (F T) K D C',
                                   H=n_heads, F=num_pretrain_frames, T=T, K=n_points)
                    source_weights[k] = new_v[:, :num_frames].flatten(0, -2)
                else:
                    continue
            elif k.split('.')[-1] == 'sampling_offsets':
                new_v = repeat(source_weights[k], 'H M K F D C -> H M K (F T) D C', T=T)
                source_weights[k] = new_v[:, :, :, :num_frames]
            elif k.split('.')[-1] == 'lvl_spatial_scales' and k.split('.')[-2] == 'temp_attn_inst':
                source_weights[k] = repeat(source_weights[k], 'F -> (F T)', T=T)[:num_frames]
            else:
                continue

        if 'temp_embed' in k:
            new_v = repeat(source_weights[k], 'F C -> (F T) C', F=num_pretrain_frames, T=T)
            source_weights[k] = new_v[:num_frames]

    output_model_path = source_model[:-4] + '_inflated_to_f' + str(num_frames) + '.pth'
    print('Mode with inflated weights is saved in:', output_model_path)
    torch.save(source_weights, output_model_path)

    return output_model_path


if __name__ == "__main__":
    args = parse_args()
    if args.convert_to_d2:
        convert_to_d2(args.source_model)
    else:
        inflated_weights(args.num_frames, args.source_model)