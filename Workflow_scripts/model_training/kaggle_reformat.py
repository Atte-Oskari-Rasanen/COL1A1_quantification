#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:32:35 2021

@author: atte
"""

from PIL import Image
import numpy as np
from import_images_masks_patches import *

np_path = '/home/inf-54-2020/experimental_cop/scripts/np_data/s512/'

X_train_k, Y_train_k = import_kaggledata(TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#or if the data has already been saved as .npy files, then they can be imported in 
#the following manner:
#X_train_k = np.load(np_path +'X_train_s512.npy')
#Y_train_k = np.load(np_path +'Y_train_s512.npy')

#the user may specify own directory structure 
kaggle_images = '/home/inf-54-2020/experimental_cop/Train_H_Final/kaggle/images/'
kaggle_masks = '/home/inf-54-2020/experimental_cop/Train_H_Final/kaggle/masks/'

#names for the files
ki = 'kaggle_img_'
km = 'kaggle_mask_'
k = 0
m = 0
print(X_train_k.shape)
for im in X_train_k:
    print(im.shape)
#    im = np.squeeze(im, axis=2)
    img = Image.fromarray((im * 255).astype(np.uint8))
   # print(img.shape)
    fname = kaggle_images + ki + str(k) + '.png'
    img.save(fname)
    k += 1
    print(k)
for im in Y_train_k:
    print(im.shape)
    im = np.squeeze(im, axis=2)
    
    print(im.shape)
    img = Image.fromarray((im * 255).astype(np.uint8))
    fname = kaggle_masks + km + str(m) + '.png'
    img.save(fname)
    m += 1
    print(m)

print('done!')

