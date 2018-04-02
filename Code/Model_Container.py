#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:55:56 2018

@author: Pedro DIego López Maroto
"""

import math
import cv2
from utils import binary_crossentropy_valid, square_to_original, prob_to_rles, post_processing
from utils import do_watershed, apply_watershed, do_dilate
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input
from keras import backend as K
import pandas as pd
import numpy as np
from Dsbowl_Dataset import Dsbowl_Dataset
from matplotlib import pyplot as plt

class Model_Container(object):
    
    def __init__(self, dir_database):
        
        self.dataset = Dsbowl_Dataset(dir_database)
        self.model = None
        self.history = None
        
    def do_training(self, model, epochs, batch_size=32):
        
        self.dataset.load_train()
        
        if model == 0:
            self.rcn5_model()
        
        gen_train = self.dataset.generator('train', batch_size=batch_size)
        gen_valid = self.dataset.generator('valid', batch_size=batch_size)
        
        train_size = self.dataset.x_train.shape[0]
        valid_size = self.dataset.x_valid.shape[0]
        
        print('Entrenando en %d muestras. Validando en %d muestras' %
              (train_size, valid_size))
        
        train_steps = math.ceil(train_size / batch_size)
        valid_steps = math.ceil(valid_size / batch_size)

        checkpoint = ModelCheckpoint('../Models/weightsADAM.{epoch:02d}.hdf5')

        self.history = self.model.fit_generator(gen_train,
                                           epochs=epochs,
                                           steps_per_epoch=train_steps,
                                           verbose=1,
                                           validation_data=gen_valid,
                                           validation_steps=valid_steps,
                                           callbacks=[checkpoint])
        
        train_loss = self.history.history['loss']
        valid_loss = self.history.history['val_loss']
        
        x = list(range(1, epochs + 1))
        plt.plot(x, train_loss, label='train_loss')
        plt.plot(x, valid_loss, label='val_loss')
        plt.title('Average of the categorical crossentropy for each landmark')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()
      
        # Saving losses
        np.save('../Models/train_loss.npy', train_loss)
        np.save('../Models/valid_loss.npy', valid_loss)
        
        # Show best validation model
        best_idx = np.argmin(valid_loss)
        print('Best validation model: ' + str(best_idx + 1) + ' with score: ' + str(valid_loss[best_idx]))


    def predict_with_model(self, model_name):
               
        model = load_model('../Models/' + model_name,
                           custom_objects={'binary_crossentropy_valid' : binary_crossentropy_valid})
        test = self.dataset.test.copy()
        if len(test.shape) == 3:
            test = np.expand_dims(test, -1)
        predict = model.predict(test)
        predict = predict[:, : ,:, 0]
        out = []
        for i in range(predict.shape[0]):           
            pred = square_to_original(predict[i,], self.dataset.test_shape[i,])
            out.append(post_processing(pred))
            
        return out
        
    
    def predict_with_watershed(self, mask_model_name, marker_model_name):
                
        mask_model = load_model('../Models/' + mask_model_name,
                           custom_objects={'binary_crossentropy_valid' : binary_crossentropy_valid})
        predict = mask_model.predict(self.dataset.test)
        predict = predict[:, : ,:, 0]
        pred = [square_to_original(predict[i,], self.dataset.test_shape[i,]) > 0.5 for i in range(predict.shape[0])]
        
        marker_model = load_model('../Models/' + marker_model_name,
                           custom_objects={'binary_crossentropy_valid' : binary_crossentropy_valid})
        sure_fg = marker_model.predict(self.dataset.test)
        sure_fg = sure_fg[:, : ,:, 0]
        sure = [square_to_original(sure_fg[i,], self.dataset.test_shape[i,]) > 0.65 for i in range(sure_fg.shape[0])]
        
        out = []
        for i in range(sure_fg.shape[0]):
            img = cv2.cvtColor(pred[i].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            w = do_watershed(img, pred[i], sure[i])
            aux = apply_watershed(pred[i], w)
            out.append(do_dilate(aux))
            
        return out


    def generate_submission(self, predict, do_labelling=False):
        
        new_test_ids = []
        rles = []
        for n, id_ in enumerate(self.dataset.ids_test):
            rle = list(prob_to_rles(predict[n], do_labelling=do_labelling))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))
        
        # Create submission DataFrame
        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv('../sub-dsbowl2018-1.csv', index=False)        


    def rcn5_model(self):
        inp = Input(shape=(160, 160, 3))
        
        num_filters = 16
    
        #Creo rama 5
        with K.name_scope('Inicio_Rama5'):
            t5_1 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t5_1')(inp)
    
        #Creo rama 4
        with K.name_scope('Inicio_Rama4'):
            t4_1 = MaxPooling2D(pool_size=(2, 2), name='pool4')(t5_1)
            t4_1 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t4_1')(t4_1)
     
        #Creo rama 3
        with K.name_scope('Inicio_Rama3'):
            t3_1 = MaxPooling2D(pool_size=(2, 2), name='pool3')(t4_1)
            t3_1 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t3_1')(t3_1)
        
        #Creo rama 2
        with K.name_scope('Inicio_Rama2'):
            t2_1 = MaxPooling2D(pool_size=(2, 2), name='pool2')(t3_1)
            t2_1 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t2_1')(t2_1)
        
        #Creo rama 1
        with K.name_scope('Inicio_Rama1'):
            t1_1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(t2_1)
            t1_1 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t1_1')(t1_1)
          
        #Avanzo rama 1
        with K.name_scope('Rama1'):
            t1_2 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t1_2')(t1_1)
            t1_3 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t1_3')(t1_2)
            t1_4 = Conv2DTranspose(num_filters, (2,2), strides=2)(t1_3)
        
        #Avanzo rama 2
        with K.name_scope('Rama2'):
            t2_2 = concatenate([t2_1,t1_4],axis=-1)
            t2_2 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t2_2')(t2_1)
            t2_3 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t2_3')(t2_2)
            t2_4 = Conv2DTranspose(num_filters, (2,2), strides=2)(t2_3)
        
        #Avanzo rama 3    
        with K.name_scope('Rama3'):
            t3_2 = concatenate([t3_1,t2_4],axis=-1)
            t3_3 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t3_2')(t3_2)
            t3_4 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t3_3')(t3_3)
            t3_5 = Conv2DTranspose(num_filters, (2,2), strides=2)(t3_4)
          
        #Avanzo rama 4
        with K.name_scope('Rama4'):
            t4_2 = concatenate([t4_1,t3_5],axis=-1)
            t4_3 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t4_2')(t4_2)
            t4_4 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t4_3')(t4_3)
            t4_5 = Conv2DTranspose(num_filters, (2,2), strides=2)(t4_4)
            
        #Avanzo rama 5
        with K.name_scope('Rama5'):
            t5_2 = concatenate([t5_1,t4_5],axis=-1)
            t5_3 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t5_2')(t5_2)
            t5_4 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name = 'conv_t5_3')(t5_3)
            
        #Capas finales
        with K.name_scope('Final'):
            final = Conv2D(num_filters,(3,3), padding='same', activation='relu', name = 'conv_final')(t5_4)
            x_class = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(final)

        self.model = Model(inputs=inp, outputs=x_class)
        #opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1., clipvalue=0.5) 
        self.model.compile(optimizer='sgd', loss=binary_crossentropy_valid)
        
        