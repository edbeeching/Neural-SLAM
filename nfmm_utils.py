#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:20:01 2020

@author: edward
"""


import  numpy as np
import matplotlib.pyplot as plt
import torch
from arguments import get_args
from env.habitat import make_env_fn
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from env.habitat.habitat_api.habitat_baselines.config.default import get_config as cfg_baseline

import matplotlib
matplotlib.use('Qt5Agg')
#plt.scatter([0,1], [1,0])
#assert 0

def setup_config_env(args, scene_id=0):
    basic_config = cfg_env(config_paths=
                           ["env/habitat/habitat_api/configs/" + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
    
    
    config_env = cfg_env(config_paths=
                             ["env/habitat/habitat_api/configs/" + args.task_config]) 
    print(scenes)
    config_env.defrost()
    config_env.DATASET.CONTENT_SCENES = scenes[scene_id: scene_id+1]
    config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    config_env.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]    

    config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
    config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

    config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

    config_env.SIMULATOR.TURN_ANGLE = 10
    config_env.DATASET.SPLIT = args.split

    config_env.freeze()
    
    return config_env
    

