#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 00:13:40 2021

@author: atte
Code adapted from https://github.com/bnsreenu/python_for_microscopists

"""
# from tensorflow.keras.optimizers import schedules

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
from datetime import datetime 
from datetime import date

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
from Models_unet_import import *
from import_images_masks_patches import *


from datetime import date
run_date = date.today()

run_date = date.today()
from keras import backend as K


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

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

seed = 42
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




plots_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/Plots/"
model_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/Models/"
histories_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/Histories/"


IMG_PROP = int(sys.argv[1])  #size of the image patch
IMG_HEIGHT = IMG_WIDTH = IMG_PROP
# IMG_HEIGHT = int(sys.argv[3])
# IMG_WIDTH = int(sys.argv[4])
IMG_CHANNELS = 3
batch_size= int(sys.argv[2])

TRAIN_PATH = sys.argv[3]
MASK_PATH = sys.argv[4] 

data = sys.argv[5]
test_path_p = sys.argv[6]
test_mask_p = sys.argv[7]

# TRAIN_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/'
# MASK_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
# img_patch = gen_patches(img, split_width, split_height)

X_train = import_images(TRAIN_PATH, IMG_HEIGHT,IMG_WIDTH, 3)
Y_train = import_masks(MASK_PATH, IMG_HEIGHT,IMG_WIDTH)

#or if using the folder structure used by kaggle's datascience bowl 2018 competition:
#X_train, Y_train = import_kaggledata(TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#normalise

X_train = X_train/255.
Y_train = Y_train/255.

X_train = X_train.astype(np.float64)
Y_train = Y_train.astype(np.float64)




all_train_imgs = len(os.listdir(TRAIN_PATH))
all_train_imgs = X_train.shape[0]
print('number of images:' + str(all_train_imgs))

steps_per_epoch = len(X_train)//batch_size
batch_size_multpl = batch_size * 5
steps_per_epoch = len(X_train)//batch_size_multpl

epochs = 10
print('epochs:' + str(epochs))

print(run_date)


'''
Attention UNet
'''
input_shape = (IMG_PROP,IMG_PROP,3)
attres_unet_model = Attention_ResUNet(input_shape)

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-6,
#     decay_steps=100,
#     decay_rate=0.9)
# optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

attres_unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss=[dice_coef_loss], 
              #BinaryFocalLoss(gamma=2)
              metrics=[dice_coef, tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), jacard_coef, jacard_coef_loss]
              )


# att_unet_model.compile(optimizer=Adam(lr = 1e-5), loss=[dice_coef_loss], 
#               metrics=[dice_coef])

# save_path_model = model_path + 'All_Attention_UNet_Dice_lrshec_'+str(IMG_PROP) + '_' + str(run_date)  +'_8.h5'
save_path_model = model_path + 'All_Attres_UNet_Dice_lrshec_'+str(IMG_PROP) + '_'  + str(batch_size) + '_' + str(run_date)  + '.h5'

from keras.callbacks import CSVLogger

csv_logger = CSVLogger(histories_path+ 'CSV_log_Attres_All_history_df_10_' + str(IMG_PROP) + '_' + data + '_'+ str(run_date) + '_' + str(batch_size) + '.csv', append=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss',restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs'), csv_logger]

print(attres_unet_model.summary())
start2 = datetime.now() 
print('steps per epoch: ' + str(steps_per_epoch))
print('number of training files: '+ str(all_train_imgs))
# att_unet_history = att_unet_model.fit(X_train, Y_train, validation_split=0.3,
#                     verbose=1,
#                     batch_size = batch_size,
#                     shuffle=False,
#                     epochs=1)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.5, random_state = 1)


history_att_unet = attres_unet_model.fit(X_train,Y_train, validation_data=(X_val,Y_val), batch_size=batch_size, epochs=epochs, callbacks=[callbacks])

stop2 = datetime.now()
#Execution time of the model 
execution_time_Att_Unet = stop2-start2
print("Attention res UNet execution time is: ", execution_time_Att_Unet)

attres_unet_model.save(save_path_model)



import pandas as pd
att_unet_history_df = pd.DataFrame(history_att_unet.history) 

with open(histories_path+'Attres_All_history_df_' + str(IMG_PROP) + '_' + data + '_'+ str(run_date) + '_' + str(batch_size) + '.csv', mode='w') as f:
    att_unet_history_df.to_csv(f)

# test_path_p = '/home/inf-54-2020/experimental_cop/Train_H_Final/Test_set/Images/'
# test_mask_p = '/home/inf-54-2020/experimental_cop/Train_H_Final/Test_set/Masks/'

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

print('-----Attention Res U net------')
# loss, accuracy, precision, recall, f1 = att_unet_model.evaluate(X_test, Y_test, verbose=0)
results = attres_unet_model.evaluate(X_test, Y_test, verbose=0)
print(results)


print('steps per epoch:' + str(steps_per_epoch))


'''
Res_unet
'''
input_shape = (IMG_PROP,IMG_PROP,3)
res_unet_model = Res_Unet(input_shape)

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-6,
#     decay_steps=100,
#     decay_rate=0.9)
# optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

res_unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss=[dice_coef_loss], 
              #BinaryFocalLoss(gamma=2)
              metrics=[dice_coef, tf.keras.metrics.AUC(), tf.keras.metrics.Precision()]
              )


# att_unet_model.compile(optimizer=Adam(lr = 1e-5), loss=[dice_coef_loss], 
#               metrics=[dice_coef])

save_path_model = model_path + 'All_Res_UNet_Dice_lrshec_'+str(IMG_PROP) + '_'  + str(batch_size) + '_' + str(run_date)  + '.h5'
from keras.callbacks import CSVLogger

csv_logger = CSVLogger(histories_path+ 'CSV_log_resUnet_All_history_df_10_' + str(IMG_PROP) + '_' + data + '_'+ str(run_date) + '_' + str(batch_size) + '.csv', append=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss',restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs'), csv_logger]

print(res_unet_model.summary())
start2 = datetime.now() 
print('steps per epoch: ' + str(steps_per_epoch))
print('number of training files: '+ str(all_train_imgs))
# att_unet_history = att_unet_model.fit(X_train, Y_train, validation_split=0.3,
#                     verbose=1,
#                     batch_size = batch_size,
#                     shuffle=False,
#                     epochs=1)
history_res_unet = res_unet_model.fit(X_train,Y_train, validation_data=(X_val,Y_val), batch_size=batch_size, epochs=epochs, callbacks=[callbacks])

stop2 = datetime.now()
#Execution time of the model 
execution_time_Att_Unet = stop2-start2
print("Res UNet execution time is: ", execution_time_Att_Unet)

res_unet_model.save(save_path_model)


import pandas as pd
history_res_unet = pd.DataFrame(history_res_unet.history) 
with open(histories_path+ 'ResUnet_history_df_' + str(IMG_PROP) + '_' + data + '_'+ str(run_date) + '_' + str(batch_size) + '.csv', mode='w') as f:
    history_res_unet.to_csv(f)

# test_path_p = '/home/inf-54-2020/experimental_cop/Train_H_Final/Test_set/Images/'
# test_mask_p = '/home/inf-54-2020/experimental_cop/Train_H_Final/Test_set/Masks/'

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

print('-----Res U net------')
# loss, accuracy, precision, recall, f1 = att_unet_model.evaluate(X_test, Y_test, verbose=0)
results = res_unet_model.evaluate(X_test, Y_test, verbose=0)
print(results)


####
'''
Traditional U-net
'''

input_shape = (IMG_PROP,IMG_PROP,3)
unet_model = Unet(input_shape)

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-6,
#     decay_steps=100,
#     decay_rate=0.9)
# optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss=[dice_coef_loss], 
              #BinaryFocalLoss(gamma=2)
              metrics=[dice_coef, tf.keras.metrics.AUC(), tf.keras.metrics.Precision()]
              )


# att_unet_model.compile(optimizer=Adam(lr = 1e-5), loss=[dice_coef_loss], 
#               metrics=[dice_coef])

save_path_model = model_path + 'UNet_Dice_lrshec_'+str(IMG_PROP) + '_'  + str(batch_size) + '_' + str(run_date)  + '.h5'
from keras.callbacks import CSVLogger

csv_logger = CSVLogger(histories_path+ 'CSV_log_Unet_All_history_df_10_' + str(IMG_PROP) + '_' + data + '_'+ str(run_date) + '_' + str(batch_size) + '.csv', append=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss',restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs'), csv_logger]

print(unet_model.summary())
start2 = datetime.now() 
print('steps per epoch: ' + str(steps_per_epoch))
print('number of training files: '+ str(all_train_imgs))
# att_unet_history = att_unet_model.fit(X_train, Y_train, validation_split=0.3,
#                     verbose=1,
#                     batch_size = batch_size,
#                     shuffle=False,
#                     epochs=1)
unet_model = res_unet_model.fit(X_train,Y_train, validation_data=(X_val,Y_val), batch_size=batch_size, epochs=epochs, callbacks=[callbacks])

stop2 = datetime.now()
#Execution time of the model 
execution_time_Att_Unet = stop2-start2
print("UNet execution time is: ", execution_time_Att_Unet)

unet_model.save(save_path_model)


import pandas as pd
history_res_unet = pd.DataFrame(history_res_unet.history) 
with open(histories_path+ 'Unet_history_df_' + str(IMG_PROP) + '_' + data + '_'+ str(run_date) + '_' + str(batch_size) + '.csv', mode='w') as f:
    history_res_unet.to_csv(f)







