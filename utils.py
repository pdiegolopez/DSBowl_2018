#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:49:10 2018

@author: Pedro Diego López Maroto
"""

import numpy as np

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