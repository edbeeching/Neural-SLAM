#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:21:05 2020

@author: edward
"""

import os, glob
import  numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.io import imread
from io import BytesIO 
from torch.utils.data import Dataset, DataLoader
from nfmm_config import get_cfg_defaults

def load_compressed_image(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
        
    return data

def uncompress_image_data(data):
    image = imread(BytesIO(data))
    return image

def test_image_loading(filepath):
    
    data = load_compressed_image(filepath)
    image = uncompress_image_data(data)
    image2 = imread(filepath)
    
    assert np.array_equal(image, image2), 'loading compressed data to memory is not the same'
    
    print('compression test passed', filepath)


def get_bounding_box(mat):
    assert np.sum(mat) > 0
        
    i_max, j_max = np.max(np.where(mat), 1)
    i_min, j_min = np.min(np.where(mat), 1)
    
    assert i_min < i_max
    assert j_min < j_max
    
    return i_min, i_max, j_max, j_min
    
    
def crop_mat(mat, i_min, i_max, j_max, j_min, padding=10):
    assert i_max - i_min > 2*padding
    assert j_max - j_min > 2*padding
    
    
    return mat[i_min - padding: i_max + padding,
               j_min - padding: j_max + padding]
    
    
    
    

class NfmmDataset(Dataset):
    def __init__(self, config):
        files = glob.glob(os.path.join(config.PATH, '*.png'))
        self.compressed_data = [load_compressed_image(f) for f in files]
        
        
    def __len__(self):
        return len(self.compressed_data)
    
    def __getitem__(self, idx): 
        mat = uncompress_image_data(self.compressed_data[idx])
        
        i_min, i_max, j_max, j_min = get_bounding_box(mat)
        
        return crop_mat(mat, i_min, i_max, j_max, j_min)


    


if __name__ == '__main__':
    config = get_cfg_defaults()
    
    
    
    dataset = NfmmDataset(config.DATASET.TRAIN)
    
    mat = dataset[100]
    
    plt.imshow(mat)
