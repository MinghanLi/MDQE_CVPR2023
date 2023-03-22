# MDQE: Mining Discriminative Query Embeddings to Segment Occluded Instances on Challenging Videos

[LI Minghan](https://scholar.google.com/citations?user=LhdBgMAAAAAJ), [LI Shuai](https://scholar.google.com/citations?user=Bd73ldQAAAAJ&hl=zh-TW), [XIANG Wang Meng](https://scholar.google.com.hk/citations?user=LFNwNF4AAAAJ&hl=en), [ZHANG Lei](https://www4.comp.polyu.edu.hk/~cslzhang/)

[[`arXiv`](https://arxiv.org/abs/2206.04403)] 

<div align="center">
  <img src="mdqe_overview.jpg" width="100%" height="100%"/>
</div><br/>

## Updates
* **`Nov 30, 2022`:** Code and pretrained weights are now available! 

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

We provide a script `train_net.py`, that is made to train all the configs provided in MDQE.

Before training: To train a model with "train_net.py" on VIS, first
setup the corresponding datasets following
[Preparing Datasets for MDQE](./datasets/README.md).

Then download pretrained weights in the Model Zoo into the path 'pretrained/coco/*.pth', and run:
```
python train_net.py --num-gpus 8 \
  --config-file configs/R50_ovis.yaml 
```

To evaluate a model's performance, use
```
python train_net.py \
  --config-file configs/R50_ovis.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## <a name="ModelZoo"></a>Model Zoo

### Pretrained weights on COCO 
| Name |                                                               R-50                                                                |                                                               R-101                                                                |                                                                                    Swin-L                                                                                     |
|:----:|:---------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| MDQE | [model](https://drive.google.com/file/d/1qolsM1Qdwdut3ckJQj8RDwtA6pwMXW6a/view?usp=share_link), [config](./configs/R50_coco.yaml) | [model](https://drive.google.com/file/d/1Ia7mdy8016u9Tah9fHpIdUxU9HQBSL3F/view?usp=share_link), [config](./configs/R101_coco.yaml) | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/EYF08Kl4z8dHuiSKvH3T7MUBz8oEU3LBaRL1CXC3HzTIxA?e=CiOku4), [config](./configs/swinl_coco.yaml) |

### OVIS
| Name | Backbone | Frames  |  AP  | AP50 | AP75 | AR1  | AR10 |                                                               Download                                                                |
|:----:|:--------:|:-------:|:----:|:----:|:----:|:----:|:----:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| MDQE |  R-50    | f3+360p | 31.0 | 55.9 | 32.0 | 14.5 | 36.3 | [model](https://drive.google.com/file/d/1uaDUECTt6hnO65gLGTlkyBWa_pHkfIdk/view?usp=share_link), [config](./configs/R50_ovis_360.yaml) |
| MDQE |  R-50    | f3+720p | 33.0 | 57.4 | 32.2 | 15.4 | 38.4 |                                    [model](), [config](./configs/R50_ovis_720.yaml)                                                   |                            
| MDQE |  Swin-L  |   f2    | 41.6 | 68.1 | 42.8 | 18.1 | 45.9 |                                            [model](), [config](./configs/swinl_ovis.yaml)                                             |

### YouTubeVIS-2021 + YouTubeVIS-2022 Long videos
| Name | Backbone | Frames |  AP  | AP50 | AP75 | AR1  | AR10  | AP<sup>L</sup> | AP50<sup>L</sup> | AP75<sup>L</sup> | AR1<sup>L</sup> | AR10<sup>L</sup> |                                                               Download                                                               |
|:----:|:--------:|:------:|:----:|:----:|:----:|:----:|:-----:|:--------------:|:-----:|:-----:|:----:|:-----:|:------------------------------------------------------------------------------------------------------------------------------------:|
| MDQE |   R-50   |   f3   | 44.6 | 68.3 | 49.5 | 37.9 | 50.3  |
| MDQE |   R-50   |   f1   | 44.9 | 67.8 | 48.1 | 37.8 | 49.4  |      33.2      | 63.1  | 33.1  | 27.0 | 37.1  | [model](https://drive.google.com/file/d/1W-BHfur9iz81xJBVWa2n8WZzQjnDdTZA/view?usp=share_link), [config](./configs/R50_ytvis21.yaml) |
| VITA |  Swin-L  |   f1   | 56.2 | 80.0 | 61.1 | 44.9 | 59.1  |                |       |       |      |       |                                          [model](), [config](./configs/swinl_ytvis21.yaml)                                           |


### YouTubeVIS-2019
| Name | Backbone | Frames |  AP  | AP50 | AP75 | AR1  | AR10 |                     Download                     |
|:----:|:--------:|:------:|:----:|:----:|:----:|:----:|:----:|:------------------------------------------------:| 
| MDQE |   R-50   |   3    | 47.3 | 66.9 | 53.1 | 42.9 | 52.9 | [model](), [config](./configs/R50_ytvis19.yaml)  |
| MDQE |  Swin-L  |   3    | 63.0 | 86.9 | 67.9 | 56.3 | 68.1 | [model](), [config](./configs/R101_ytvis19.yaml) |

## License
The majority of MDQE is licensed under a [Apache-2.0 License](LICENSE).
However portions of the project are available under separate license terms: Detectron2([Apache-2.0 License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)), IFC([Apache-2.0 License](https://github.com/sukjunhwang/IFC/blob/master/LICENSE)), VITA([Apache-2.0 License](https://github.com/sukjunhwang/VITA)), and Deformable-DETR([Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE)).

## <a name="CitingVITA"></a>Citing MDQE

If you use VITA in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@inproceedings{li2022mdqe,
  title={MDQE: Mining Discriminative Query Embeddings to Segment Occluded Instances on Challenging Videos},
  author={Minghan Li, Shuai Li, Wangmeng Xiang, and Zhang Lei},
  journal={.},
  year={2022}
}
```

## Acknowledgement

Our code is largely based on [Detectron2](https://github.com/facebookresearch/detectron2), [IFC](https://github.com/sukjunhwang/IFC), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [VITA](https://github.com/sukjunhwang/VITA). We are truly grateful for their excellent work.
# MDQE_CVPR2023
