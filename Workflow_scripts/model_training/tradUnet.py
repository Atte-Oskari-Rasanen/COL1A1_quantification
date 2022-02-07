#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: atte

Code adapted from https://github.com/bnsreenu/python_for_microscopists
"""

import tensorflow as tf
import os
import random
import numpy as np
# from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
from tensorflow.keras import Model
import matplotlib
matplotlib.use('Agg')

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.optimizers import Adam
from import_images_masks_patches import *
from U_net_function import * 
from PIL import Image
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import math
from datetime import date
from Models_unet_import import *

run_date = date.today()
from keras import backend as K
tf.keras.metrics.AUC(
    num_thresholds=200,
    curve="ROC",
    summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    num_labels=None,
    label_weights=None,
    from_logits=False,
)
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
f1_m
def dice_coef_loss(y_true, y_pred):
    return 1-f1_m(y_true, y_pred)
    #return 1-dice_coef(y_true, y_pred)

seed = 42
np.random.seed = seed
def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x
def save_random_im(X_train, Y_train):
    random_no = random.randint(0, len(X_train))
    im1 = X_train[random_no]
    im = np.array(Image.fromarray((im1 * 255).astype(np.uint8)))
    im = Image.fromarray(im)
    im.save("random_im.png")
    
    mask1 = Y_train[random_no]
    mask = np.array(Image.fromarray((mask1 * 255).astype(np.uint8)))
    mask = Image.fromarray(mask)
    mask.save("random_mask.png")


IMG_CHANNELS = 3

# TRAIN_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/'
# MASK_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/'

IMG_PROP = 512
IMG_PROP = int(sys.argv[1])

IMG_HEIGHT = IMG_WIDTH = IMG_PROP
# IMG_HEIGHT = int(sys.argv[3])
# IMG_WIDTH = int(sys.argv[4])
IMG_CHANNELS = 3
# batch_size = 32
batch_size = int(sys.argv[2])

plots_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/Plots/"
# plots_path = "/home/inf-54-2020/experimental_cop/scripts/plots_unet/"
model_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/Models/"
# model_path = "/home/inf-54-2020/experimental_cop/scripts/unet_models/"

histories_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/Histories/"

# histories_path= "/home/inf-54-2020/experimental_cop/scripts/"
# TRAIN_PATH = "/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/"
TRAIN_PATH = sys.argv[3]
MASK_PATH = sys.argv[4] 
data = sys.argv[5]
test_path_p = sys.argv[6]
test_mask_p = sys.argv[7]

print('Train_path: ' + TRAIN_PATH)
print(len(os.listdir(TRAIN_PATH)))
# MASK_PATH = "/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/"
print('Train_path: ' + MASK_PATH)

# TRAIN_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/'
# MASK_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/'
# img_patch = gen_patches(img, split_width, split_height)

X_train = import_images(TRAIN_PATH, IMG_HEIGHT,IMG_WIDTH, 3)
print('X_train imported!')

Y_train = import_masks(MASK_PATH, IMG_HEIGHT,IMG_WIDTH)
print('All imported!')



#or the user can use the following function in case the data is kaggle DSB2018 folder format:
#X_train, Y_train = import_kaggledata(TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# batch_size=128
all_train_imgs = len(os.listdir(TRAIN_PATH))

# all_train_imgs = X_train.shape[0]
# print(all_train_imgs)
# def calculate_spe(y):
#   return int(math.ceil((1. * y) / batch_size))
# steps_per_epoch = calculate_spe(all_train_imgs)
steps_per_epoch = len(X_train)//batch_size
# validation_steps = (len(X_train)*0.3)//batch_size # if you have test data

#normalise
X_train = X_train / 255.
Y_train = Y_train / 255.

X_train = X_train.astype(np.float64)
Y_train = Y_train.astype(np.float64)

print("dtype X:", X_train.dtype)
print("dtype Y:", Y_train.dtype)

all_train_imgs = len(os.listdir(TRAIN_PATH))
all_train_imgs = X_train.shape[0]
print('number of images:' + str(all_train_imgs))

# def calculate_spe(y):
#   return int(math.ceil((1. * y) / batch_size))
# steps_per_epoch = calculate_spe(all_train_imgs)
batch_size_multpl = batch_size * 5
steps_per_epoch = len(X_train)//batch_size_multpl

print(steps_per_epoch)
epochs = 10
print('epochs:' + str(epochs))

print(run_date)

print('Compiling Unet...')
'''
unet
'''
#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255.)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.0)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
batch1 = BatchNormalization(axis=3)(conv1)

p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.0)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.0)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.0)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.0)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#exp paths dropout at c6 and c7 used to be 0.2, changed to 0.1
#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.0)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.0)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.0)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.0)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9) 
 
unet_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss=[dice_coef_loss], 
              #BinaryFocalLoss(gamma=2)
              metrics=[dice_coef, keras.metrics.Precision(), keras.metrics.Recall(), tf.keras.metrics.AUC(), precision, recall, jacard_coef, jacard_coef_loss, recall_m, precision_m, f1_m]
              )


# att_unet_model.compile(optimizer=Adam(lr = 1e-5), loss=[dice_coef_loss], 
#               metrics=[dice_coef])

save_path_model = model_path + 'All_UNet_Dice_lrshec_'+str(IMG_PROP) + '_'  + str(batch_size) + '_' + str(run_date)  +'_8' +'.h5'
from keras.callbacks import CSVLogger

csv_logger = CSVLogger(histories_path+ 'CSV_log_tradU_All_history_df_10_' + str(IMG_PROP) + '_' + data + '_'+ str(run_date) + '_' + str(batch_size) + '.csv', append=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, monitor='val_loss',restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs'), csv_logger]

print(unet_model.summary())
print('steps per epoch: ' + str(steps_per_epoch))
print('number of training files: '+ str(all_train_imgs))
# att_unet_history = att_unet_model.fit(X_train, Y_train, validation_split=0.3,
#                     verbose=1,
#                     batch_size = batch_size,
#                     shuffle=False,
#                     epochs=1)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.5, random_state = 1)

epochs=10
history_unet = unet_model.fit(X_train,Y_train, validation_data=(X_val,Y_val), batch_size=batch_size, epochs=epochs, callbacks=[callbacks])

# stop2 = datetime.now()
#Execution time of the model 

unet_model.save(save_path_model)
import pandas as pd
history_unet = pd.DataFrame(history_unet.history) 
with open(histories_path+ 'Unet_history_df_10_' + str(IMG_PROP) + '_' + data + '_'+ str(run_date) + '_' + str(batch_size) + '.csv', mode='w') as f:
    history_unet.to_csv(f)

epochs = 10




# Test_ims_masks = import_kaggledata(data_path, IMG_PROP, IMG_PROP, 3)
X_test = import_images(test_path_p, IMG_PROP, IMG_PROP, 3)
Y_test = import_masks(test_mask_p, IMG_PROP, IMG_PROP)
# X_train = Test_ims_masks[0]
# Y_train = Test_ims_masks[1]

print(X_test.shape)
print(Y_test.shape)

X_test = X_test/255.
Y_test = Y_test/255.

X_test = X_test.astype(np.float64)
Y_test = Y_test.astype(np.float64)
# evaluate the model

print('-----stats------')
loss, accuracy, precision, recall, f1 = unet_model.evaluate(X_test, Y_test, verbose=0)
results = unet_model.evaluate(X_test, Y_test, verbose=0)
print(results)


print('steps per epoch:' + str(steps_per_epoch))
