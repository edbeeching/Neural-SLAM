#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:18:46 2020

@author: edward
"""
import os 
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"
from skimage.io import imsave
from nfmm_utils import setup_config_env
from arguments import get_args
from env.habitat import make_env_fn
from env.habitat.habitat_api.habitat_baselines.config.default import get_config as cfg_baseline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

if __name__ == '__main__':
    args = get_args()
    args.map_size_cm = 5120
    rank = 0
    
    out_dir = 'data/nfmm_data/{}/{:003}_{:003}.png'
    
    scene_lengths = {
        'train': 72,
        'val': 14,
        }
   
    # args.split = 'val'
    # config_env = setup_config_env(args, scene_id=1)
    # dataset = PointNavDatasetV1(config_env.DATASET)
    #env = make_env_fn(args, config_env, cfg_baseline(), rank)    
    
    #env.habitat_env._sim.reconfigure
    
    #obs, info = env.reset()
    #explorable_map = env.explorable_map    
    
    # assert 0
    for stage in scene_lengths.keys():
        args.split = stage
        for scene_id in range(scene_lengths[stage]):
            config_env = setup_config_env(args, scene_id=scene_id)
            env = make_env_fn(args, config_env, cfg_baseline(), rank)
            
            for j in range(100):
            
                obs, info = env.reset()
                explorable_map = env.explorable_map
                
                # plt.figure()
                # plt.imshow(explorable_map)
                imsave(out_dir.format(stage,
                                      scene_id,
                                      j), 
                       explorable_map)
            env.close()
        
