#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:13:09 2021

@author: atte
"""

import shutil

import numpy as np
import matplotlib.pyplot as plt
from smooth_tiled_divisions_edited2 import *

from skimage import data
from skimage.color import rgb2hed, hed2rgb
import cv2
import PIL
from PIL import Image
import sys
import os
import ntpath
import glob
import tensorflow
# import tensorflow
import keras

#Python script for renaming and reorganising the deconvolved images performed by ImageJ and then
#segmenting them.

directory = sys.argv[1]
print('Directory: ' + directory)
patch_size = int(sys.argv[2])
model_path = sys.argv[3]

#make a directory for deconvolved images:
main_dir = directory + '/Deconvolved_ims'

import random
def im_id():
    seed = random.getrandbits(32)
    while True:
       yield seed
       seed += 1

uniq_id = im_id()


try:
    os.mkdir(main_dir)
except FileExistsError:
    print('Attempted to create a directory for deconvolved images but one already exists')
    pass

a = list(range(1,10))
ints = list(map(str,a))
col1_dic= {}

#Create the directory structure: ./Deconvolved_ims -> Animal ID -> col1a1_dir + hunu_dir
#The deconvolved images by ImageJ are in the directory of the specific animal. Take the images from
#here and transfer them to the right location inside Deconvolved_ims/. 

import scandir
for root, subdirectories, files in scandir.walk(directory):
    if 'Deconvolved_ims' in subdirectories:
        subdirectories.remove('Deconvolved_ims')


    for subdir in subdirectories:
        if any(integ in subdir for integ in ints): #check that the dir name contains an int
            animal_id = 'Animal_' + subdir
            animal_dir = main_dir + '/' + animal_id
            #under Deconvolved_im/ make a subdir for each image id
            try:
                os.mkdir(animal_dir)
            except OSError:
                print ("Creation of the directory %s failed, the directory already exists." % animal_dir)
            # else:
            #     print ("Successfully created the directory %s " % animal_dir)
    
            hunu_ch_dir = subdir.rsplit('/')[-1] + '_hunu_ch'
            hunu_ch_dir = animal_dir + '/' + hunu_ch_dir
            col1a1_ch_dir = subdir.rsplit('/')[-1] + '_col1a1_ch'
            col1a1_ch_dir = animal_dir + '/' + col1a1_ch_dir
            try:
                os.mkdir(hunu_ch_dir)
            except OSError:
                print ("Creation of the directory %s failed, the directory already exists." % hunu_ch_dir)
            try:
                os.mkdir(col1a1_ch_dir)
            except OSError:
                print ("Creation of the directory %s failed, the directory already exists." % col1a1_ch_dir)
            
            subdir_path = root + '/' + subdir + '/'
            
            
            #now you have created a directory under Deconvolved_ims with specific animal ID along with subdirs for 
            #hunu and col1a1. Now need to transfer the files from the original location to here.
            im_index = 0
            for file in os.listdir(subdir_path):
                imagepath = subdir_path + file
                if '~' in imagepath:
                    imagepath = imagepath.split('~')[0]
                # if not imagefile.endswith('.tif') or imagefile.endswith('.jpg'): #exclude files not ending in .tif
                #     continue
                #print(imagepath)
                imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
                imagename_orig = imagename.split('.')[0]
                id_spec = next(uniq_id) # specific id for the corresponding hunu and its col1a1 image
                # #subdir designated the animal id, 
                if 'col1a1' in file and not animal_id in file: #not subdir condition to prevent stacking up the new name to several files containing the string match or processing files inside deconvolved dir
                    #give the file the specific id name
                    new_filename = str(subdir) + '_' +str(id_spec) + '_' + str(patch_size) +'_'+ imagename_orig +'.png'
                    old_file = imagepath 
                    new_file = subdir + "/" + new_filename
                    new_file = root + '/'+subdir + "/" + new_filename

                    os.rename(old_file, new_file)
                    imagepath=new_file

                    #transfer the file to its new directory under Deconvolved_ims
                    # Set the directory path where the file will be moved
                    destination_path = col1a1_ch_dir + '/' + new_filename
                    new_location = shutil.move(imagepath, destination_path)
                    
                    #Get the id of the file along with its original filename
                    f_split = new_filename.split('_')
                    file_id = '_'.join(f_split[:3]), '_'.join(f_split[3:])
                    file_id = file_id[0]
                    name1= new_filename.split('col1a1')[0]
                    name2 = name1.split('_')
                    name2 = '_'.join(name2[:3]), '_'.join(name2[3:])
                    common_name = name2[1]
                    col1_dic[common_name]=file_id
                    print("The %s is moved to the location, %s" %(file, new_location))
            
            for hunufile in os.listdir(subdir_path):
                if 'hunu' in hunufile and not animal_id in hunufile:
                    imagepath=subdir_path +  hunufile
                    for id_part in col1_dic.keys():
                        if id_part in hunufile:
                            new_filename = col1_dic[id_part] + '_' + id_part + '_hunu.png'
                            old_file = imagepath
                            new_file = subdir + "/" + new_filename
                            new_file = root + '/'+subdir + "/" + new_filename

                            os.rename(old_file, new_file)
                            imagepath=new_file
        
                            destination_path = hunu_ch_dir + '/' + new_filename
                            new_location = shutil.move(imagepath, destination_path)
                            print("The %s is moved to the location, %s" %(imagepath, new_location))
    
                im_index += 1
        else:
            continue
print('Files reorganised! Starting segmentation...')


model_segm = keras.models.load_model(model_path, compile=False) #need to set compile as false since we are predicting, >

[x[0] for x in os.walk(directory)]
for root, subdirectories, files in scandir.walk(main_dir):
    for subdir in subdirectories:
        subdir_path = root +'/' +subdir + '/'
        imagepath=root + '/'+subdir + "/" + file
        if '~' in imagepath:
            imagepath = imagepath.split('~')[0]
        imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
        imagename = imagename.split('.')[0]
        id_hunu_col = next(uniq_id) # specific id for the corresponding hunu and its col1a1 image

        imagename = str(im_index) + '_' +str(id_hunu_col) +'_'+ subdir + '_' #+ str(patch_size)

        print('SUBDIR PATH FOR SEGMENTATION: ' +subdir_path)
        #print(glob.glob(subdir_path))
        for imagefile in os.listdir(subdir_path):
            if imagefile.endswith('.png'): 
                if 'hunu' in imagefile:
                    imagepath=subdir_path + "/" + imagefile
                    img = cv2.imread(imagepath)
                    img = img/255.
                    img = img.astype(np.float64) #earlier 32


                    imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
                    imagename_orig = imagename.split('.')[0]
                    
                    #apply predict_img_with_smooth_windowing function which segments the image using the trained model
                    #supplied by cutting the image into patches of the appointed size, segmenting them, then combining 
                    #the patches
                    img_segm_grids_removed = predict_img_with_smooth_windowing(img,
                        window_size=patch_size,
                        subdivisions=4,  # Minimal amount of overlap for windowing. Must be an even number.
                        nb_classes=1,
                        pred_func=(
                            lambda img_batch_subdiv: model_segm.predict((img_batch_subdiv))
                        )
                    )
                    img_segm_grids_removed = np.squeeze(img_segm_grids_removed, axis = 2)
                    im_final = Image.fromarray((img_segm_grids_removed * 255).astype(np.uint8))
                    im_final.save(subdir_path + imagename_orig + '_Segm.png')
                    print('Segmented image saved as: '+ subdir_path + imagename + str(patch_size) + '_Segm.png')
                else:
                    continue





print("SEGMENTED!")
