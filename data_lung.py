# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:30:10 2020

@author: kdutta01
"""

import os 
import natsort
import numpy as np
from skimage.io import imread
import cv2
import re
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split

train_datapath = './Data/2d_images/'
mask_datapath = './Data/2d_masks/'

def atoi(text) : 
    return int(text) if text.isdigit() else text
def natural_keys(text) :
    return [atoi(c) for c in re.split('(\d+)', text)]

for root, dirnames, image_filenames in os.walk(train_datapath):
    image_filenames.sort(key = natural_keys)
    rootpath = root
    
for root, dirnames, mask_filenames in os.walk(mask_datapath):
    mask_filenames.sort(key = natural_keys)
    rootpath = root
    

img_train, img_test, mask_train, mask_test = train_test_split(image_filenames, mask_filenames, test_size=0.2, random_state=42)

image_row_size = 256
image_col_size = 256
train_image = np.ndarray((len(img_train), image_row_size , image_col_size), dtype = np.uint8)
train_mask = np.ndarray((len(img_train), image_row_size , image_col_size), dtype = np.uint8)

test_image = np.ndarray((len(img_test), image_row_size , image_col_size), dtype = np.uint8)
test_mask = np.ndarray((len(mask_test), image_row_size , image_col_size), dtype = np.uint8)

count = 0
for i in range(len(img_train)):
    image = cv2.imread(os.path.join(train_datapath,img_train[i]),0)
    image_resize = cv2.resize(image,(256,256))
    img = np.array([image_resize])
    train_image[count] = img
    
    mask = cv2.imread(os.path.join(mask_datapath,mask_train[i]),0)
    mask_resize = cv2.resize(mask,(256,256))
    msk = np.array([mask_resize])
    train_mask[count] = msk
    
    count = count + 1
    
count = 0
for i in range(len(img_test)):
    image = cv2.imread(os.path.join(train_datapath,img_test[i]),0)
    image_resize = cv2.resize(image,(256,256))
    img = np.array([image_resize])
    test_image[count] = img
    
    mask = cv2.imread(os.path.join(mask_datapath,mask_test[i]),0)
    mask_resize = cv2.resize(mask,(256,256))
    msk = np.array([mask_resize])
    test_mask[count] = msk
    
    count = count + 1   
    
np.save('train_image_lung.npy', train_image)  
np.save('train_mask_lung.npy', train_mask) 
np.save('test_image_lung.npy', test_image) 
np.save('test_mask_lung.npy', test_mask) 

def load_test_data():
    images_test = np.load('test_image_lung.npy')
    print('======Loading of Test Data=======')
    return images_test
    

def load_train_data():
    images_train = np.load('train_image_lung.npy')
    mask_train = np.load('train_mask_lung.npy')
    return images_train, mask_train