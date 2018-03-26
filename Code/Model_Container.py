#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:55:56 2018

@author: Pedro DIego López Maroto
"""

import math
from utils import rpn_loss_cls, square_to_original
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input
from keras import optimizers
from keras import backend as K
from Dsbowl_Dataset import Dsbowl_Dataset


class Model_Container(object):
    
    def __init__(self, dir_database):
        
        self.dataset = Dsbowl_Dataset(dir_database)
        self.model = None
        
        
    def do_training(self, model, epochs, batch_size=32):
        
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

        history = self.model.fit_generator(gen_train,
                                           epochs=epochs,
                                           steps_per_epoch=train_steps,
                                           verbose=1,
                                           validation_data=gen_valid,
                                           validation_steps=valid_steps,
                                           callbacks=[checkpoint])
        return history


    def predict_valid(self, model_name):
        
        model = load_model('../Models/' + model_name,
                           custom_objects={'rpn_loss_cls' : rpn_loss_cls})
        gen_valid = self.dataset.generator('valid')
        valid_size = self.dataset.x_valid.shape[0]
        valid_steps = math.ceil(valid_size / 32)    
        predict = model.predict_generator(gen_valid, valid_steps)
        
        return predict


    def predict_test(self, model_name):
        
        model = load_model('../Models/' + model_name,
                           custom_objects={'rpn_loss_cls' : rpn_loss_cls})
        predict = model.predict(self.database.test)
        predict = predict[:, : ,:, 0]
        for i in range(predict.shape[0]):           
            predict[i,] = square_to_original(predict[i,], self.dataset.test_shape[i,])
            
        return predict
        

    def rcn5_model(self):
        inp = Input(shape=(160,160,3))
        
        num_filters = 64
    
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
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1., clipvalue=0.5) 
        self.model.compile(optimizer='sgd', loss=rpn_loss_cls)
        
        