#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:31:14 2020

@author: edward
"""

from yacs.config import CfgNode as CN


_C = CN()
# =============================================================================
# HYPERPARAM
# =============================================================================
_C.HYP = CN()
_C.HYP.BATCH_SIZE = 512
_C.HYP.LR = 0.001
_C.HYP.NUM_EPOCHS = 500
_C.HYP.MAX_GRAD_NORM = 2.0
_C.HYP.WEIGHT_DECAY = 0.001

# =============================================================================
# MODEL
# =============================================================================
_C.MODEL = CN()
_C.MODEL.NAME = "resnet18"
_C.MODEL.TYPE = "concat"
_C.MODEL.INPUT_CHANS = 4
_C.MODEL.TRANSFORMER_LAYERS = 4
_C.MODEL.TRANSFORMER_HEADS = 4

# =============================================================================
# DATASET
# =============================================================================
_C.DATASET = CN()
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.PATH = "data/nfmm_data/train/"
_C.DATASET.TRAIN.LIMIT = 0


_C.DATASET.VAL = CN()
_C.DATASET.VAL.PATH = "data/nfmm_data/train/"
_C.DATASET.VAL.LIMIT = 0



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
