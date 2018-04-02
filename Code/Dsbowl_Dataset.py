#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:25:35 2018

@author: Pedro Diego López Maroto
"""

import os
import glob
import numpy as np
from utils import transformations, load_images, load_label, generate_valid
from utils import get_test_shape
from multiprocessing import Pool


class Dsbowl_Dataset(object):
    
    def __init__(self, dir_database, gauss=True):
        
        self.dir_database = dir_database
                
        # Pool
        self.pool = Pool()
    
   
    def load_train(self):
        
        self.preload = os.path.exists(self.dir_database + '/x_train.npy')
        
        if self.preload:
            print('Loading pre-saved data')
            self.x_train = np.load(self.dir_database + '/x_train.npy')
            self.y_train = np.load(self.dir_database + '/y_train.npy')
            self.x_valid = np.load(self.dir_database + '/x_valid.npy')
            self.y_valid = np.load(self.dir_database + '/y_valid.npy')
            
            self.ids_train = list(np.load(self.dir_database + '/ids_train.npy'))
            self.ids_valid = list(np.load(self.dir_database + '/ids_valid.npy'))

        else:
            print('Doing preprocessing load')
            # Training set
            ids = os.listdir(self.dir_database + '/train')
            self.image_id = ids     
            ids = [self.dir_database + '/train/' + idx for idx in ids]
              
            # Images
            self.x = self.pool.map(load_images, ids)
            self.x = np.array(self.x)
            
            # Labels
            self.y = []
            self.valids = []
            
            self.load_labels()
            self.load_valids()
            
            self.y = np.array(self.y)
            self.valids = np.array(self.valids)
            self.y = np.stack((self.y, self.valids), axis=3)
      
            # Entrenamiento
            self.x_train = []
            self.y_train = []
            self.ids_train = []
            self.x_valid = []
            self.y_valid = []
            self.ids_valid = []
            
            # Dividir en conjuntos de entrenamiento y validacion
            self.split_train_val()    
            self.save_data_train()
       
        
    def load_test(self, dir_test=None):
        
        if dir_test is not None:
            dir_database = dir_test + '/'
        else:
            dir_database = self.dir_database + '/test/'
        
        self.preload = os.path.exists(dir_database + '/test.npy')
        
        if self.preload:
            print('Loading pre-saved data')
            self.test = np.load(dir_database + '/test.npy')
            self.ids_test = list(np.load(dir_database + '/ids_test.npy'))
            self.test_shape = np.load(dir_database + '/test_shape.npy')
        else:           
            # Test set
            ids = os.listdir(dir_database)  
            self.ids_test = ids
            ids = [dir_database + idx for idx in ids] 
            
            # Images
            self.test = self.pool.map(load_images, ids)
            self.test_shape = self.pool.map(get_test_shape, ids)
            self.test_shape = np.array(self.test_shape)
            self.test = np.array(self.test)       
            self.save_data_test()
        
        
    def load_labels(self, gauss=True):
        ids = os.listdir(self.dir_database + '/train')
        ids = [self.dir_database + '/train/' + idx for idx in ids]
        
        for idx in ids:
            dir_images = glob.glob(idx + '/masks/*.png')

            images = self.pool.map(load_label, dir_images)

            label = np.zeros((160, 160), dtype=np.float32)
            
            for im in images:
                label = np.maximum(label, im)
                
            self.y.append(label)
            
            
    def load_valids(self):
        self.valids = self.pool.map(generate_valid, self.y)
    
    
    """
    Funcion que coge un 20% aleatorio de las muestras para validación
    """
    def split_train_val(self):
        self.x_train = self.x
        self.y_train = self.y
        self.ids_train = self.image_id
        self.shuffle()
        
        split_nb = int(0.2 * self.x_train.shape[0])
        
        # El conjunto de entrenamiento ya barajado se divide.
        self.x_valid = self.x_train[-split_nb:,]
        self.y_valid = self.y_train[-split_nb:,]
        self.ids_valid = self.image_id[-split_nb:]
        
        self.x_train = self.x_train[0:-split_nb,]
        self.y_train = self.y_train[0:-split_nb,]
        self.ids_train = self.image_id[0:-split_nb]
        
    
    """
    Funcion que realiza un barajeo en todo el conjunto de entrenamiento
    """
    def shuffle(self):
        
        index = np.array(range(0, self.x_train.shape[0])) 
        np.random.shuffle(index)
        self.x_train = self.x_train[index,]
        self.y_train = self.y_train[index,]
        self.ids_train = np.array(self.ids_train)
        self.ids_train = self.ids_train[index]
        self.ids_train = list(self.ids_train)
               
        
    def save_data_train(self):
        np.save(self.dir_database + '/x_train.npy', self.x_train)
        np.save(self.dir_database + '/y_train.npy', self.y_train)
        np.save(self.dir_database + '/x_valid.npy', self.x_valid)
        np.save(self.dir_database + '/y_valid.npy', self.y_valid)
        np.save(self.dir_database + '/ids_train.npy', np.array(self.ids_train))
        np.save(self.dir_database + '/ids_valid.npy', np.array(self.ids_valid))

                
    def save_data_test(self):
        np.save(self.dir_database + '/test.npy', self.test)        
        np.save(self.dir_database + '/ids_test.npy', np.array(self.ids_test))
        np.save(self.dir_database + '/test_shape.npy', self.test_shape)

        
    """
    Generador de muestras para entrenar
    """
    def generator(self, phase, batch_size=32, shuffle=True, data_aug=True):
        
        # Train or test phase
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
                    x_batch = x_tot[-alone_samples:,].copy()
                    y_batch = y_tot[-alone_samples:,].copy()
                    
                if data_aug:    
                    # Las paso a lista para poder meterla en el pool.map
                    x_batch = list(x_batch)
                    y_batch = list(y_batch)
                    out = self.pool.map(transformations, list(zip(x_batch, y_batch)))
                    
                    # Transformo la lista de tuplas en lista de dos listas
                    out = list(zip(*out))
                    
                    # Vuelta a array
                    x_batch = np.array(out[0], dtype=np.float32)
                    y_batch = np.array(out[1], dtype=np.float32)
                    
                yield x_batch, y_batch