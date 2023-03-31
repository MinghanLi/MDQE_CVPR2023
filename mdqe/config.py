# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_mdqe_config(cfg):
    """
    Add config for MDQE.
    """
    cfg.DATASETS.DATASET_RATIO = []

    cfg.MODEL.MDQE = CN()
    cfg.MODEL.MDQE.NUM_CLASSES = 80

    # DataLoader
    cfg.INPUT.PRETRAIN_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_NUM = 3
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = []  # "brightness", "contrast", "saturation", "rotation"

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ (not used in our MDQE)
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    # LOSS
    cfg.MODEL.MDQE.BOX_WEIGHT = 2.0
    cfg.MODEL.MDQE.MASK_WEIGHT = 4.0
    cfg.MODEL.MDQE.DICE_WEIGHT = 4.0
    cfg.MODEL.MDQE.DEEP_SUPERVISION = True
    cfg.MODEL.MDQE.NO_OBJECT_WEIGHT = 1
    cfg.MODEL.MDQE.MASK_STRIDE = 4
    cfg.MODEL.MDQE.MATCH_STRIDE = 4
    cfg.MODEL.MDQE.MASK_DIM = 32
    cfg.MODEL.MDQE.NUM_MASK_LAYERS = 1

    # TRANSFORMER
    cfg.MODEL.MDQE.NHEADS = 8
    cfg.MODEL.MDQE.DROPOUT = 0.1
    cfg.MODEL.MDQE.MLP_RATIO = 4
    cfg.MODEL.MDQE.ENC_LAYERS = 6
    cfg.MODEL.MDQE.DEC_LAYERS = 6
    cfg.MODEL.MDQE.PRE_NORM = False

    # Deformable DETR
    cfg.MODEL.MDQE.HIDDEN_DIM = 256
    cfg.MODEL.MDQE.NUM_OBJECT_QUERIES = 200
    cfg.MODEL.MDQE.NUM_FEATURE_LEVELS = 4
    cfg.MODEL.MDQE.ENC_NUM_POINTS = 4
    cfg.MODEL.MDQE.DEC_NUM_POINTS = 4
    cfg.MODEL.MDQE.DEC_TEMPORAL = True

    # inter-frame query association
    cfg.MODEL.MDQE.QUERY_EMBED_DIM = 64
    cfg.MODEL.MDQE.WINDOW_INTER_FRAME_ASSOCIATION = 5

    # inter-instance mask repulsion loss
    cfg.MODEL.MDQE.INTERINST_MASK_LOSS_ENABLED = True
    cfg.MODEL.MDQE.INTERINST_MASK_THRESHOLD = 0.1

    # Evaluation
    cfg.MODEL.MDQE.CLIP_STRIDE = 1
    cfg.MODEL.MDQE.SAMPLING_FRAME_NUM_TEST = 5
    cfg.MODEL.MDQE.WINDOW_FRAME_NUM_TEST = 20
    cfg.MODEL.MDQE.MAX_NUM_INSTANCES = 50
    cfg.MODEL.MDQE.MERGE_ON_CPU = False
    cfg.MODEL.MDQE.MULTI_CLS_ON = True
    cfg.MODEL.MDQE.APPLY_CLS_THRES = 0.05

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.NUM_PRETRAIN_FRAMES = 1
