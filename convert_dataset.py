"""
The first part of converting coco2vis comes from VITA (https://github.com/sukjunhwang/VITA). Thanks (:
"""
import os
import json
import time

from mdqe.data.datasets.ytvis import (
    COCO_TO_YTVIS_2019,
    COCO_TO_YTVIS_2021,
    COCO_TO_OVIS
)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

# STEP1: Convert images in COCO as complementary VIS dataset
convert_list = [
    (COCO_TO_YTVIS_2019, 
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/annotations/coco2ytvis2019_train.json"), "COCO to YTVIS 2019:"),
    (COCO_TO_YTVIS_2019, 
        os.path.join(_root, "coco/annotations/instances_val2017.json"),
        os.path.join(_root, "coco/annotations/coco2ytvis2019_val.json"), "COCO val to YTVIS 2019:"),
    (COCO_TO_YTVIS_2021, 
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/annotations/coco2ytvis2021_train.json"), "COCO to YTVIS 2021:"),
    (COCO_TO_YTVIS_2021, 
        os.path.join(_root, "coco/annotations/instances_val2017.json"),
        os.path.join(_root, "coco/annotations/coco2ytvis2021_val.json"), "COCO val to YTVIS 2021:"),
    (COCO_TO_OVIS, 
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/annotations/coco2ovis_train.json"), "COCO to OVIS:"),
]

for convert_dict, src_path, out_path, msg in convert_list:
    src_f = open(src_path, "r")
    out_f = open(out_path, "w")
    src_json = json.load(src_f)

    out_json = {}
    for k, v in src_json.items():
        if k != 'annotations':
            out_json[k] = v

    converted_item_num = 0
    out_json['annotations'] = []
    for anno in src_json['annotations']:
        if anno["category_id"] not in convert_dict:
            continue

        out_json['annotations'].append(anno)
        converted_item_num += 1

    json.dump(out_json, out_f)
    print(msg, converted_item_num, "items converted.")


# STEP 2: Since ground-truth annotations in valid set are not published, we split the videos of training set
# in to train_sub set (90%) and valid_sub set (10%) for convenient model evaluation.
valid_precent = 0.1
annotation_files = ["ytvis_2019/train.json", "ytvis_2021/train.json", "ovis/train.json"]
for annotation_file in annotation_files:
   # split videos in training set into train_sub set and valid_sub set
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(annotation_file, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    n_videos_cate_valid_sub = [0] * len(dataset['categories'])

    videos = dataset['videos']
    annotations = dataset['annotations']
    n_videos_each_cate_valid_sub = 2 * int(len(videos) * valid_precent) // len(dataset['categories'])
    if annotation_file.split('/')[-2] == 'ovis':
        n_videos_each_cate_valid_sub *= 2

    dataset_train_sub = {'videos': [], 'annotations': []}
    for k, v in dataset.items():
        if k not in dataset_train_sub:
            dataset_train_sub[k] = v

    dataset_valid_sub = {'videos': [], 'annotations': []}
    for k, v in dataset.items():
        if k not in dataset_valid_sub:
            dataset_valid_sub[k] = v

    video_ids_train_sub, video_ids_valid_sub = [], []
    for anno in annotations:
        if anno['video_id'] not in video_ids_train_sub+video_ids_valid_sub:
            if n_videos_cate_valid_sub[anno['category_id']-1] <= n_videos_each_cate_valid_sub and len(anno["segmentations"]) <= 100:
                video_ids_valid_sub.append(anno['video_id'])
            else:
                video_ids_train_sub.append(anno['video_id'])

        if anno['video_id'] in video_ids_valid_sub:
            dataset_valid_sub['annotations'].append(anno)
            n_videos_cate_valid_sub[anno['category_id'] - 1] += 1
        else:
            dataset_train_sub['annotations'].append(anno)

    for video in videos:
        if video['id'] in video_ids_valid_sub:
            dataset_valid_sub['videos'].append(video)
        else:
            dataset_train_sub['videos'].append(video)

    file_path_train_sub = annotation_file.replace('train.json', 'train_sub.json')
    file_path_valid_sub = annotation_file.replace('train.json', 'valid_sub.json')
    print("Saving videos to {}".format(file_path_train_sub))
    with open(file_path_train_sub, 'w') as fp:
        json.dump(dataset_train_sub, fp)

    print("Saving videos to {}".format(file_path_valid_sub))
    with open(file_path_valid_sub, 'w') as fp:
        json.dump(dataset_valid_sub, fp)
