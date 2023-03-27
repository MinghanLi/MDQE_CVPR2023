# Prepare Datasets for MDQE

MDQE has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  coco/
  ytvis_2019/
  ytvis_2021/
  ovis/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

<!-- The [model zoo](https://github.com/facebookresearch/MaskFormer/blob/master/MODEL_ZOO.md)
contains configs and models that use these builtin datasets. -->

## STEP-1: Prepare Image & VIS datasets
### Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
coco/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

### Expected dataset structure for [YouTubeVIS 2019](https://competitions.codalab.org/competitions/20128):

```
ytvis_2019/
  {train,valid,test}.json
  {train,valid,test}/
    JPEGImages/
```

### Expected dataset structure for [YouTubeVIS 2021](https://competitions.codalab.org/competitions/28988)+[TouTubeVIS 2022](https://codalab.lisn.upsaclay.fr/competitions/3410):
For evaluating on valid set of YouTubeVIS 2022, you just need to replace 'valid.json' of YouTubeVIS 2021 with 'valid.json' of YouTubeVIS 2022.

```
ytvis_2021/
  {train,valid,test}.json
  {train,valid,test}/
    JPEGImages/
```

### Expected dataset structure for [OVIS](https://competitions.codalab.org/competitions/32377):

```
ovis/
  {train,valid,test}.json
  {train,valid,test}/
    JPEGImages/
```

## STEP-2: Prepare annotations

### a) Convert COCO to VIS dataset
This part is largely based on [VITA](https://github.com/sukjunhwang/VITA). We are truly grateful for the excellent work.

### b) Split train set into train_sub set and valid_sub set
For convenient model evaluation, we split the training annotations train.json into two sets: train_sub.json and valid_sub.json.
And train_sub.json is used for training, while valid_sub.json is used for validation.

```bash
python convert_dataset.py
```
### Expected final dataset structure for all:
```
$DETECTRON2_DATASETS
+-- coco
|   |
|   +-- annotations
|   |   |
|   |   +-- instances_{train,val}2017.json
|   |   +-- coco2ytvis2019_train.json
|   |   +-- coco2ytvis2021_train.json
|   |   +-- coco2ovis_train.json
|   |
|   +-- {train,val}2017
|       |
|       +-- *.jpg
|
+-- ytvis_2019
|   | 
|   +-- train.json
|   +-- train_sub.json
|   +-- valid.json
|   +-- valid_sub.json
|   +-- test.json
|
+-- ytvis_2021
|   | 
|   +-- train.json
|   +-- train_sub.json
|   +-- valid.json
|   +-- valid_sub.json
|   +-- test.json
|
+-- ovis
|   | 
|   +-- train.json
|   +-- train_sub.json
|   +-- valid.json
|   +-- valid_sub.json
|   +-- test.json
|
```