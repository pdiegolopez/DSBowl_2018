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
        index_end = np.round(pad).astype(int)
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
    
    p_fh = np.random.uniform()
    p_fv = np.random.uniform()
    
    if p_fh > 0.5:
        x = cv2.flip(x, 1)
        y = cv2.flip(y, 1)
    
    if p_fv > 0.5:
        x = cv2.flip(x, 0)
        y = cv2.flip(y, 0)  
    
    return (x, y)

"""
Carga una imagen dada la ruta a su carpeta id
"""
def load_images(image_id):
    
    filename = glob.glob(image_id + '/images/*.png')
    im = cv2.imread(filename[0])
    im = get_square_img(im)
    
    im = cv2.resize(im, (160, 160), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    im = im / 255
    
    return im
    

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
    
    gaussian_factor = int((np.log2(np.sum(mask)) / np.log2(1.4)) - (100 / np.sum(mask)))
    
    new_label = gaussian_filter(new_label, gaussian_factor, mode='wrap')
    
    new_label = mask * new_label
    
    new_label = new_label / np.max(new_label)
    
    new_label *= 255
    new_label = get_square_img(new_label)
    
    new_label = cv2.resize(new_label, (160, 160), interpolation=cv2.INTER_AREA)
    new_label = new_label / 255
    
    return new_label
    