#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 18:12:34 2018

@author: Pedro Diego López Maroto
"""

import numpy as np
from Dsbowl_dataset import Dsbowl_dataset
from matplotlib import pyplot as plt

class Dataset_Viewer(object):
    
    def __init__(self, dir_dataset):
        
        self.dataset = Dsbowl_dataset(dir_dataset)
        
        
    def visualize_one_random_batch_train(self):
        
        generator = self.dataset.generator('train')
        
        x_batch, y_batch = next(generator)
        x_batch = 255* x_batch
        x_batch = x_batch.astype(np.uint8)
        
        ids = self.dataset.ids_train[0:32]

        for i in range(x_batch.shape[0]):       
            plt.subplot(131)
            plt.imshow(x_batch[i,])
            plt.title(ids[i])
            plt.subplot(132)
            plt.imshow(y_batch[i, :, :, 0])
            plt.subplot(133)
            plt.imshow(y_batch[i, :, :, 1])
            plt.waitforbuttonpress()
            
            
    def visualize_testset(self):
        
        for i in range(self.dataset.test.shape[0]):
            plt.imshow(self.dataset.test[i,])
            plt.waitforbuttonpress()