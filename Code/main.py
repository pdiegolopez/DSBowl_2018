#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:34:55 2018

@author: pdiegolopez
"""

import argparse
from Model_Container import Model_Container
from Dataset_Viewer import Dataset_Viewer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir_database', '-db', dest='dir_database', required=True,
                        help='Path to database folder')
 
    parser.add_argument('--visualize', '-v', dest='visualize', required=False, default='False',
                        help='Path to database folder')    
    
    args = parser.parse_args()
    
    visualize = eval(args.visualize)
    
    md = Model_Container(args.dir_database)
    
    print('Generating solution')
    
    predict = md.predict_with_watershed('../Models/RCN5_BASE/base.hdf5', '../Models/RCN5_CENTER/center.hdf5')
    
    if visualize:
        v = Dataset_Viewer(args.dir_database)
        v.visualize_predictions(predict)
    
    md.generate_submission(predict)