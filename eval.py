# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 00:52:19 2020

@author: kdutta01
"""

import numpy as np
import os
import cv2
from skimage.io import imsave
from skimage.measure import label
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, accuracy_score
from natsort import natsorted
test_lung = np.load('test_mask_lung.npy')

def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    largest=max(list_seg, key=lambda x:x[1])[0]
    labels_max=(labels == largest).astype(int)
    return labels_max

# pred_directory = './gt_lung/'
# if not os.path.exists(pred_directory):
#     os.mkdir(pred_directory)
# count = 0
# for i in range(0, test_lung.shape[0]):
#     test = test_lung[i]
#     #test_large = getLargestCC(test)
#     imsave(os.path.join(pred_directory,  str(count) + '.png' ), test)
#     count = count + 1

count = 0

pred_directory = './prediction/lung/'
test_directory = './gt_lung/'
imgs_test = natsorted(os.listdir(test_directory))

list_files = os.listdir(pred_directory)

jacc = np.ndarray((len(list_files), len(imgs_test)), dtype = np.float32)
f1 = np.ndarray((len(list_files), len(imgs_test)), dtype = np.float32)
precision = np.ndarray((len(list_files), len(imgs_test)), dtype = np.float32)
recall = np.ndarray((len(list_files), len(imgs_test)), dtype = np.float32)
sensitivity = np.ndarray((len(list_files), len(imgs_test)), dtype = np.float32)
specificity = np.ndarray((len(list_files), len(imgs_test)), dtype = np.float32)
auc = np.ndarray((len(list_files), len(imgs_test)), dtype = np.float32)
acc = np.ndarray((len(list_files), len(imgs_test)), dtype = np.float32)

count1 = 0
for lists in list_files:
    new_dir = os.path.join(pred_directory, lists)
    imgs = natsorted(os.listdir(new_dir))
    count2=0
    for img,img_test in zip(imgs,imgs_test):
        gt = cv2.imread(os.path.join(new_dir,img),0).flatten()
        gt = gt.astype('float32')
        gt = gt/255.
        gt[gt<=0.5] = 0
        gt[gt>0.5] = 1
        
        pred = cv2.imread(os.path.join(test_directory,img_test),0).flatten()
        pred = pred.astype('float32')
        pred = pred/255.
        pred[pred<=0.5] = 0
        pred[pred>0.5] = 1
        
        jacc[count1,count2] = jaccard_score(gt,pred,average='binary')
        f1[count1,count2] = f1_score(gt,pred,average='binary')
        precision[count1,count2] = precision_score(gt,pred,average='binary')
        recall[count1,count2] = recall_score(gt,pred,average='binary')
        test_conf = confusion_matrix(gt,pred)
        sensitivity[count1,count2] = test_conf[0,0]/(test_conf[0,0] + test_conf[0,1])
        specificity[count1,count2] = test_conf[1,1]/(test_conf[1,1] + test_conf[1,0])
        auc[count1,count2] = roc_auc_score(gt,pred)
        acc[count1,count2] = accuracy_score(gt,pred)
        
        count2 = count2 + 1
        
    count1 = count1 + 1    
        
        
    
