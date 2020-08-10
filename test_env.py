import time
from collections import deque

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gym
import logging
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module

import algo

import sys
import matplotlib

import matplotlib.pyplot as plt

args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main():
    # Setup Logging
    # Logging and loss variables

    # Starting environments
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    torch.set_num_threads(1)
    print('creating envs')
    envs = make_vec_envs(args)
    print('envs created')
    
    obs, infos = envs.reset()
    for i in range(10000):
        if i % 100 == 0:    
            print(i)
        obs, infos = envs.reset()
        
        
    print('envs reset')
    envs.close()

if __name__ == "__main__":
    main()
