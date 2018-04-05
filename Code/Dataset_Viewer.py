#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 18:12:34 2018

@author: Pedro Diego López Maroto
"""

import cv2
import numpy as np
from Dsbowl_Dataset import Dsbowl_Dataset
from utils import square_to_original
from matplotlib import pyplot as plt

class Dataset_Viewer(object):
    
    def __init__(self, dir_dataset):
        
        self.dataset = Dsbowl_Dataset(dir_dataset)
        
        
    def visualize_one_random_batch_train(self):
        
        generator = self.dataset.generator('train')
        
        x_batch, y_batch = next(generator)
        x_batch = 255* x_batch
        x_batch = x_batch.astype(np.uint8)
        
        ids = self.dataset.ids_train[0:32]

        for i in range(x_batch.shape[0]):       
            plt.subplot(131)
            try:
                plt.imshow(x_batch[i,])
            except:
                plt.imshow(x_batch[i, :, :, 0])
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
            
    
    def visualize_predictions(self, predict):
        
        for i in range(len(predict)):
            test = square_to_original(self.dataset.test[i,], self.dataset.test_shape[i,])
            plt.subplot(121)
            plt.imshow(test)
            plt.subplot(122)
            plt.imshow(predict[i] / np.max(predict[i]))
            plt.waitforbuttonpress()
            
    def compare_two_predictions(self, predict, predict2):
        
        for i in range(len(predict)):
            test = square_to_original(self.dataset.test[i,], self.dataset.test_shape[i,])
            plt.subplot(131)
            plt.imshow(test)
            plt.subplot(132)
            plt.imshow(predict[i])
            plt.subplot(133)
            plt.imshow(predict2[i])
            plt.waitforbuttonpress()
            
    
    def visualize_segmentation(self, predict):
        
        for i in range(len(predict)):
            test = square_to_original(self.dataset.test[i,], self.dataset.test_shape[i,])
            pred = predict[i]
            out = cv2.cvtColor(np.uint8(255 * test), cv2.COLOR_RGB2BGR)
            for e in range(1, np.max(pred)):
                
                cn = (pred == e).astype(np.uint8)
                cnt = cv2.findContours(cn, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1][0]              
                out = cv2.drawContours(out, [cnt], -1, (0,255,0))
                
            out = np.uint8(out)
            cv2.imshow('I', out)
            cv2.waitKey()
            