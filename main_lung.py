# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 01:25:09 2020

@author: kdutta01
"""

#from r2udensenet import r2udensenet
#from model2D import unet
#from res_unet import resunet
from model_bcdu_net import BCDU_net_D3
from r2udensenet import r2udensenet
from data_lung import load_train_data, load_test_data
import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    print('=============Loading of Training Data and Preprocessing==============')
    
    # images_train, mask_train = load_train_data()
    # images_validation, mask_validation = load_val_data()
    
    images_train = np.load('train_image_lung.npy')
    mask_train = np.load('train_mask_lung.npy')
    
    images_train = images_train.astype('float32')
    mask_train = mask_train.astype('float32')
    print(images_train.shape)
    
    images_train_mean = np.mean(images_train)
    images_train_std = np.std(images_train)
    images_train = (images_train - images_train_mean)/images_train_std
    mask_train /= 255.
    
    model = r2udensenet()
    weight_directory = 'weights'
    if not os.path.exists(weight_directory):
        os.mkdir(weight_directory)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_directory,'2dUnetLung.hdf5'), monitor = 'loss', verbose = 1, save_best_only=True)
    
    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
    logger = CSVLogger(os.path.join(log_directory,'logLung.csv'), separator = ',', append = False)
    
    start = time()
    history = model.fit(images_train, mask_train, batch_size=4, epochs=200, validation_split = 0.1, callbacks = [model_checkpoint, logger])
    
    
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='lower right')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss ( 1 - Dice )')
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper right')
    plt.show()
    
    print('===============Training Done==========================')
    print('Time taken for Training to complete =',time()-start,' ms')
    

################################### PREDICTION OF THE NETWORK ###################################      

def predict():
    print('============= Beginning of Prediction ================')
    #images_test = load_test_data()
    images_test = np.load('test_image_lung.npy')
    images_test = images_test.astype('float32')
    
    images_test_mean = np.mean(images_test)
    images_test_std = np.std(images_test)
    images_test = (images_test - images_test_mean)/images_test_std
    
    model = r2udensenet()
    weight_directory = 'weights'
    model.load_weights(os.path.join(weight_directory,'2dUnetLung.hdf5'))
    masks_test = model.predict(images_test, batch_size=1, verbose =1)    
    masks_test = np.squeeze(masks_test, axis = 3)
    #masks_test = np.around(masks_test, decimals = 0)
    masks_test = (masks_test*255.).astype(np.uint8)
    
    pred_directory = './prediction/lung/'
    if not os.path.exists(pred_directory):
        os.mkdir(pred_directory)
    
    count = 0
    for i in range(0, masks_test.shape[0]):
        imsave(os.path.join(pred_directory,  str(count) + '_pred' + '.png' ), masks_test[i])
        count = count + 1
    
    print('===========Prediction Done ==============')
    
        
if __name__ == '__main__':
    train()
    predict()
