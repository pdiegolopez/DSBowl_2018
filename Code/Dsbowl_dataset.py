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
from utils import get_square_img, transformations 
from multiprocessing import Pool



class Dsbowl_dataset(object):
    
    def __init__(self, dir_database):
        
        self.dir_database = dir_database
        self.ids = os.listdir(self.dir_database)
        
        # Datos
        self.x = self.load_images(mode=0)
        self.y = self.load_images(mode=1)   
        
        # Entrenamiento
        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []
        
        # Dividir en conjuntos de entrenamiento y validacion
        self.split_train_val()
             
        # Pool
        self.pool = Pool()
    
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
                x = np.array(x)
                x = np.sum(x, axis=0).astype(np.uint8)
            else:
                x = np.array(x)
                x = np.squeeze(x, axis=0)
            
            # Todas al mismo tamaño y blanco y negro.
            x = cv2.resize(x, (160, 160), interpolation=cv2.INTER_AREA)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            
            # Rango 0 a 1
            x = x / 255
            
            data.append(x)
            
        return np.array(data)
    
    """
    Funcion que coge un 20% aleatorio de las muestras para validación
    """
    def split_train_val(self):
        self.x_train = self.x
        self.y_train = self.y
        self.shuffle()
        
        split_nb = int(0.2 * self.x_train.shape[0])
        
        # El conjunto de entrenamiento ya barajado se divide.
        self.x_valid = self.x_train[-split_nb:,]
        self.y_valid = self.y_train[-split_nb:,]
        
        self.x_train = self.x_train[0:-split_nb,]
        self.y_train = self.y_train[0:-split_nb,]
        
    
    """
    Funcion que realiza un barajeo en todo el conjunto de entrenamiento
    """
    def shuffle(self):
        
        index = np.array(range(0, self.x_train.shape[0])) 
        np.random.shuffle(index)
        self.x_train = self.x_train[index,]
        self.y_train = self.y_train[index,]
               
    """
    Generador de muestras para entrenar
    """
    def generator(self, phase, batch_size=32, shuffle=True, data_aug=True):
        
        # Train or valid phase
        if phase == 'train':
            x_tot = self.x_train
            y_tot = self.y_train      
        else:
            x_tot = self.x_valid
            y_tot = self.y_valid 
            shuffle = False
            data_aug = False
                      
        full_batchs = x_tot.shape[0] // batch_size
        alone_samples = x_tot.shape[0] % batch_size
        
        if alone_samples == 0:
            total_batchs = full_batchs
        else:
            total_batchs = full_batchs + 1
            
        while True:
            
            if shuffle:
                self.shuffle()
            
            for i in range(total_batchs):
                
                if i < full_batchs:
                    x_batch = x_tot[i * batch_size : i * batch_size + batch_size,].copy()
                    y_batch = y_tot[i * batch_size : i * batch_size + batch_size,].copy()
                else:
                    x_batch = x_tot[-alone_samples,].copy()
                    y_batch = y_tot[-alone_samples,].copy()
                    
                if data_aug:    
                    # Las paso a lista para poder meterla en el pool.map
                    x_batch = list(x_batch)
                    y_batch = list(y_batch)
                    out = self.pool.map(transformations, list(zip(x_batch, y_batch)))
                    
                    # Transformo la lista de tuplas en lista de dos listas
                    out = list(zip(*out))
                    
                    # Vuelta a array
                    x_batch = np.array(out[0])
                    y_batch = np.array(out[1])
                
                yield x_batch, y_batch