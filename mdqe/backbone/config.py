# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_swint_config(cfg):
    # SwinT backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.OUT_FEATURES = ["stage3", "stage4", "stage5"]
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 8
    cfg.MODEL.SWIN.MLP_RATIO = 4
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # addition
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"


def add_swins_config(cfg):
    # SwinS backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.OUT_FEATURES = ["stage3", "stage4", "stage5"]
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 16
    cfg.MODEL.SWIN.MLP_RATIO = 4
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # addition
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"


def add_swinb_config(cfg):
    # SwinB backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.EMBED_DIM = 128
    cfg.MODEL.SWIN.OUT_FEATURES = ["stage3", "stage4", "stage5"]
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [4, 8, 16, 32]
    cfg.MODEL.SWIN.WINDOW_SIZE = 16
    cfg.MODEL.SWIN.MLP_RATIO = 4
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # addition
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"


def add_swinl_config(cfg):
    # SwinL backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.EMBED_DIM = 192
    cfg.MODEL.SWIN.OUT_FEATURES = ["stage3", "stage4", "stage5"]
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [6, 12, 24, 48]
    cfg.MODEL.SWIN.WINDOW_SIZE = 24
    cfg.MODEL.SWIN.MLP_RATIO = 4
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # addition
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"



