#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:49:10 2018

@author: Pedro Diego López Maroto
"""

import cv2
import glob
import numpy as np
from scipy.ndimage.filters import gaussian_filter

"""
Padding en una imagen para convertirla en cuadrada.
"""
def get_square_img(img):
    
    h, w = img.shape[0:2]
    
    if h > w:
        if len(img.shape) == 3:
            new_img = np.zeros((h, h, 3), dtype=np.uint8)
        else:
            new_img = np.zeros((h, h), dtype=np.uint8)
        pad = (h - w) / 2
        index_start = np.floor(pad).astype(int)
        index_end = np.ceil(pad).astype(int)
        new_img[:, index_start:-index_end,] = img
        return new_img
    elif w > h:
        if len(img.shape) == 3:
            new_img = np.zeros((w, w, 3), dtype=np.uint8)
        else:
            new_img = np.zeros((w, w), dtype=np.uint8)
        pad = (w - h) / 2
        index_start = np.floor(pad).astype(int)
        index_end = np.ceil(pad).astype(int)
        new_img[index_start:-index_end,] = img
        return new_img
    else:
        return img
    
    
"""
Usada por POOL
x is an image
y is an image
"""
def transformations(tupla):
    x = tupla[0]
    y = tupla[1]
    
    # Prob flip
    p_fh = np.random.uniform()
    p_fv = np.random.uniform()
    
    # 0 izq, 1 drch. 0 arriba, 1 abajo.
    dir_x = np.random.randint(0, 2)
    dir_y = np.random.randint(0, 2)
    
    # Cantidad de 0 a 20.
    q_x = np.random.randint(0, 21)
    q_y = np.random.randint(0, 21)

    # X translation
    if q_x > 0:
        if dir_x == 0:
            aux_x = np.zeros(x.shape)
            aux_y = np.zeros(y.shape)
            aux_x[:, 0:-q_x, :] = x[:, q_x:, :]
            aux_y[:, 0:-q_x, :] = y[:, q_x:, :]
            x = aux_x
            y = aux_y
        else:
            aux_x = np.zeros(x.shape)
            aux_y = np.zeros(y.shape)
            aux_x[:, q_x:, :] = x[:, 0:-q_x, :]
            aux_y[:, q_x:, :] = y[:, 0:-q_x, :]
            x = aux_x
            y = aux_y
    
    # Y translation
    if q_y > 0:
        if dir_y == 0:
            aux_x = np.zeros(x.shape)
            aux_y = np.zeros(y.shape)
            aux_x[0:-q_y, :, :] = x[q_y:, :, :]
            aux_y[0:-q_y, :, :] = y[q_y:, :, :]
            x = aux_x
            y = aux_y
        else:
            aux_x = np.zeros(x.shape)
            aux_y = np.zeros(y.shape)
            aux_x[q_y:, :, :] = x[0:-q_y, :, :]
            aux_y[q_y:, :, :] = y[0:-q_y, :, :]
            x = aux_x   
            y = aux_y
    
    # Flips
    if p_fh > 0.5:
        x = cv2.flip(x, 1)
        y = cv2.flip(y, 1)
    
    if p_fv > 0.5:
        x = cv2.flip(x, 0)
        y = cv2.flip(y, 0)  
    
    return (x, y)


"""
Carga una imagen dada la ruta a su carpeta id.
"""
def load_images(image_id):
    
    filename = glob.glob(image_id + '/images/*.png')
    im = cv2.imread(filename[0])
    im = get_square_img(im)
    
    im = cv2.resize(im, (160, 160), interpolation=cv2.INTER_AREA)
    
    if len(im.shape) < 3:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    
    im = im / 255
    
    return im
    

"""
Carga una etiqueta aplicandole el preprocesado.
"""
def load_label(dir_label):
    
    label = cv2.imread(dir_label)
    label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
    cnt = cv2.findContours(label, 1, 2)[0]
    
    mask = label > 0
    
    M = cv2.moments(cnt)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    new_label = np.zeros(label.shape, dtype=np.float32)
    new_label[cy, cx] = 1
    
    factor = 1.5
    seguir = True
    status = 0
    
    while seguir:
        gaussian_factor = int((np.log2(np.sum(mask)) / np.log2(factor)) - (100 / np.sum(mask)))
    
        candidate = gaussian_filter(new_label, gaussian_factor, mode='wrap')
    
        candidate = mask * candidate
    
        candidate = candidate / np.max(candidate)
        
        min_value = np.min(candidate[mask>0])
        
        if 0.01 <= min_value <= 0.15:
            seguir = False
            new_label = candidate
        elif min_value < 0.01:          
            if factor > 1.1 and status < 1:
                factor -= 0.1
                status = -1
            else:
                seguir = False
                new_label = candidate
        elif min_value > 0.15:
            factor += 0.1
            status = 1
            
    new_label *= 255
    new_label = get_square_img(new_label)
    
    new_label = cv2.resize(new_label, (160, 160), interpolation=cv2.INTER_AREA)
    new_label = new_label / 255
    
    return new_label
    

"""
Genera la matriz de posiciones validas para una etiqueta.
"""
def generate_valid(label):
    
    valid = label > 0
    num_pos = np.sum(valid)
    num_neg = 0
    shape = label.shape
    
    if num_pos >= shape[0] * shape[1] // 2:
        valid = np.ones(shape, dtype=bool)
    else:      
        while num_neg != num_pos:
            cx = np.random.randint(low=0, high=shape[1])
            cy = np.random.randint(low=0, high=shape[0])
            if valid[cy, cx] == 0:
                valid[cy, cx] = True
                num_neg += 1
    
    return valid
