#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:25:35 2018

@author: Pedro Diego López Maroto
"""

import cv2
import os
import glob
import numpy as np
from utils import get_square_img    

class Dsbowl_dataset(object):
    
    def __init__(self, dir_dataset):
        
        self.dir_dataset
        self.ids = os.listdir(dir_dataset)
        
        # Datos
        self.x = self.load_images(mode=0)
        self.y = self.load_images(mode=1)
    
        self.size = self.x.shape[0]
    
    """
    Funcion que carga las imagenes de un directorio, como una sola imagen.
    """
    def load_images(self, mode):
        
        # Modo 0 cargo images; modo 1 cargo mascaras
        if mode == 0:
            data_type = 'images'
        else:
            data_type = 'masks'
        
        data = []      
        for idx in self.ids:
            
            images = glob.glob(self.dir_database + '/' + idx + '/' + data_type + '/*.png')
            
            x = []
            try:            
                for image in images:
                    im = cv2.imread(image)
                    x.append(get_square_img(im))
            except:
                continue
            
            if len(x) > 1:
                x = np.sum(x)
            else:
                x = np.array(x)
            
            x = cv2.resize(x, (160, 160), interpolation=cv2.INTER_AREA)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        
            if mode == 0:
                x /= 255
            
            data.append(x)
            
        return np.array(data)
    
    """
    Funcion que realiza un barajeo en todo el dataset
    """
    def shuffle(self):
        
        index = np.array(range(0, self.size)) 
        np.random.shuffle(index)
        self.x = self.x[index,]
        self.y = self.y[index,]
    
    """
    Generador de muestras para entrenar
    """
    def generator(self, batch_size=32, shuffle=True):
        
        full_batchs = self.size // batch_size
        alone_samples = self.size % batch_size
        
        if alone_samples == 0:
            total_batchs = full_batchs
        else:
            total_batchs = full_batchs + 1
            
        while True:
            
            if shuffle:
                self.shuffle()
            
            for i in range(total_batchs):
                
                if i < full_batchs:
                    x_batch = self.x[i * batch_size : i * batch_size + batch_size,].copy()
                    y_batch = self.y[i * batch_size : i * batch_size + batch_size,].copy()
                else:
                    x_batch = self.x[-alone_samples,].copy()
                    y_batch = self.y[-alone_samples,].copy()
                    
                yield x_batch, y_batch
                