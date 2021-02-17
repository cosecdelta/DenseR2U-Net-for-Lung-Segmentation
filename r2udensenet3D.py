# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:56:40 2021

@author: kdutta01
"""

import tensorflow as tf
import os
import numpy as np
import skimage.transform as trans
from keras.models import Model
from keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, concatenate, Input, Dropout
from keras.optimizers import Adam
#from keras.utils import plot_model
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')

#Defining Loss Functions and Accuracy

smooth = 1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1 -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def common_layer(filters, layer):
    layer = Conv3D(filters, 
                   kernel_size = (3,3,3), 
                   padding = 'same',
                   activation = 'relu',
                   kernel_initializer = 'glorot_normal')(layer)
    return layer

def rec_layer(filters,layer):
    reconv1 = Conv3D(filters, kernel_size = (3,3,3), padding = 'same', activation='relu')(layer)
    reconc1 = concatenate([layer, reconv1], axis=4)
    drop_inter = Dropout(0.3)(reconc1)
    reconv2 = Conv3D(filters, kernel_size = (3, 3, 3), activation='relu', padding='same')(drop_inter)
    reconc2 = concatenate([layer, reconv2], axis=4)
    return reconc2

image_depth = 16
image_row = 128
image_col = 128

def r2udense3Dnet(pretrained_weights = None):
    inputs = Input((image_depth, image_row, image_col, 1))
    conv1 = rec_layer(32, inputs)
    conv1 = rec_layer(32,conv1)
    conc1 = concatenate([inputs, conv1], axis = 4)
    pool1 = MaxPooling3D(pool_size = (2,2,2))(conc1)
    
    conv2 = rec_layer(64, pool1)
    conv2 = rec_layer(64, conv2)
    conc2 = concatenate([pool1, conv2], axis = 4)
    pool2 = MaxPooling3D(pool_size = (2,2,2))(conc2)
    
    conv3 = rec_layer(128, pool2)
    conv3 = rec_layer(128, conv3)
    conc3 = concatenate([pool2, conv3], axis = 4)
    pool3 = MaxPooling3D(pool_size = (2,2,2))(conc3)
    
    conv4 = rec_layer(256, pool3)
    conv4 = rec_layer(256, conv4)
    conc4 = concatenate([pool3, conv4], axis = 4)
    pool4 = MaxPooling3D(pool_size = (2,2,2))(conc4)
    
    conv5 = rec_layer(512, pool4)
    conv5 = rec_layer(512, conv5)
    conc5 = concatenate([pool4, conv5], axis = 4)
    drop5 = Dropout(0.5)(conc5)
    
 
    up2 = Conv3DTranspose(256, 2, strides = (2,2,2), padding = 'same')(conv5)
    merge2 = concatenate([up2,conv4], axis = 4)
    deconv2 = rec_layer(256, merge2)
    deconv2 = rec_layer(256, deconv2)
    deconc2 = concatenate([merge2,deconv2], axis = 4)
    
    
    up3 = Conv3DTranspose(128, 2, strides = (2,2,2), padding = 'same')(deconc2)
    merge3 = concatenate([up3,conv3], axis = 4)
    deconv3 = rec_layer(128, merge3)
    deconv3 = rec_layer(128, deconv3)
    deconc3 = concatenate([merge3,deconv3], axis = 4)
    
    up4 = Conv3DTranspose(64, 2, strides = (2,2,2), padding = 'same')(deconc3)
    merge4 = concatenate([up4,conv2], axis=4)
    deconv4 = common_layer(64, merge4)
    deconv4 = common_layer(64, deconv4)
    deconc4 = concatenate([merge4,deconv4], axis = 4)
    
    up5 = Conv3DTranspose(32, 2, strides = (2,2,2), padding = 'same')(deconc4)
    merge5 = concatenate([up5,conv1], axis = 4)
    deconv5 = rec_layer(32,merge5)
    deconv5 = rec_layer(32,deconv5)
    deconc5 = concatenate([merge5,deconv5], axis = 4)
    
    deconv_final = Conv3D(1,(1,1,1), activation = 'sigmoid')(deconc5)
    
    model = Model(inputs = [inputs], outputs = [deconv_final])
    
    model.compile(optimizer = Adam(learning_rate = 1e-5, beta_1 = 0.9, beta_2 = 0.999), loss = 'binary_crossentropy', metrics = [dice_coef])
          
    if(pretrained_weights):
    	model.load_weights(pretrained_weights) 
    
    return model
