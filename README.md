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
