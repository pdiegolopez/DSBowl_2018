#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:49:10 2018

@author: Pedro Diego López Maroto
"""

import cv2
import numpy as np

"""
Padding en una imagen para convertirla en cuadrada.
"""
def get_square_img(img):
    
    h, w = img.shape[0:2]
    
    if h > w:
        new_img = np.zeros((h, h, 3), dtype=np.uint8)
        pad = (h - w) / 2
        index_start = np.floor(pad).astype(int)
        index_end = np.round(pad).astype(int)
        new_img[:, index_start:-index_end, :] = img
        return new_img
    elif w > h:
        new_img = np.zeros((w, w, 3), dtype=np.uint8)
        pad = (w - h) / 2
        index_start = np.floor(pad).astype(int)
        index_end = np.round(pad).astype(int)
        new_img[index_start:-index_end, :, :] = img
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