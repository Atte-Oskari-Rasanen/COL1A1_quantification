#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:35:27 2021

@author: atte
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from PIL import Image, ImageFilter
from skimage import measure, filters
import scandir
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage as ndi
import os
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import sys


#for the images segmented with script U2.py need to invert colours
from skimage.morphology import disk
from scipy.ndimage.filters import gaussian_filter
from skimage import util 
import cv2
import numpy as np
import PIL
import os
import re
from PIL import Image, ImageFilter
from skimage.filters import threshold_otsu, rank
import scandir
#go over the deconvolved folder, find folders that have Segmented in their names, enter the folder
#and apply hunu_ch_import_TH on it. Find the corresponding col1a1 image , threshold
import PIL.ImageOps    

def crop_image(image, angle):
    h, w = image.shape
    tan_a = abs(np.tan(angle * np.pi / 180))
    b = int(tan_a / (1 - tan_a ** 2) * (h - w * tan_a))
    d = int(tan_a / (1 - tan_a ** 2) * (w - h * tan_a))
    return image[d:h - d, b:w - b]

def TH_local_otsu(img_p,radius, sigma):
    img = cv2.imread(img_p,0)
    selem = disk(radius)
    im_blur = gaussian_filter(img, sigma=sigma)
    # im_blur = cv2.medianBlur(im_blur, 3)
    local_otsu = rank.otsu(im_blur, selem)
    binary = im_blur >= local_otsu
    binary = binary.astype(np.uint8)
    binary = crop_image(binary,2)
    binary = Image.fromarray(np.uint8(binary * 255))
    binary = PIL.ImageOps.invert(binary)
    return binary

print('Starting the Stain_channels_postprocess script!')

def col1a1_ch_import_TH(im_path):
    img = cv2.imread(im_path,0)
    thresh = threshold_otsu(img)
    im_gray = Image.fromarray(img)
    im_blur = im_gray.filter(ImageFilter.GaussianBlur(5))
    im_blur = np.asarray(im_blur)
    
    (T, threshInv) = cv2.threshold(img, thresh, 255,
    	cv2.THRESH_BINARY_INV)
    kernel_small_particl = np.ones((5,5),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    
    threshInv = cv2.bitwise_not(threshInv)
    threshInv = cv2.dilate(threshInv,kernel_small_particl,iterations = 1)

    threshInv = cv2.bitwise_not(threshInv)
    threshInv = cv2.dilate(threshInv,kernel,iterations = 3)

    threshInv = crop_image(threshInv,2)
    return threshInv


# main_dir = sys.argv[1]
main_dir = '/home/atte/Documents/PD_images/batch8_retry/Deconvolved_ims'
# main_dir = '/home/atte/Documents/PD_images/batch8_retry/18_rest/Deconvolved_ims'

#main_dir = sys.argv[1]
print(main_dir)

# create the colocalisation when you colocalise the images since prior to colocalisation you need
# to find the matching files. create the directory within this same match condition
# create a colocalised folder under each animal id 
for root, subdirectories, files in scandir.walk(main_dir):
    print(subdirectories)
    for subdir in subdirectories:
        if not 'Coloc' in subdir:
            coloc_dir = main_dir + '/Coloc'
            try:
                os.mkdir(coloc_dir)
            except OSError:
                print ("Failed to create directory %s " % coloc_dir)
            else:
                print ("Succeeded at creating the directory %s " % coloc_dir)

#get all images in a list
segm_dirs = []
all_ims_paths = []
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]

print('all imgs paths:')
#get all images that match the pattern
for f in all_ims_paths:
    #print(f)
    im_name = f.split('/')[-1]
    n = 18
    im_id = a = [im_name[i:i+n] for i in range(0, len(im_name), n)] #extracts the animal id and the code of the image
    im_id = im_id[0]
    match_hunu_col1 = list(filter(lambda x: im_id in x, all_ims_paths))
    #now you have matching image ids for col1a1, hunu and hunu_segm. you now go through 
    #the list containing all the images that were saved earlier to find the corresponding ones
    #and take the col1a1 and hunu_segm
    for file_path in match_hunu_col1:
        filename = os.path.basename(file_path)
        filename = filename.split('.')[0]
        print('file_path: ' + file_path)
        animal_id = file_path.split('_')[-3]
        if 'col1a1' in filename and not 'TH' in filename:
            col1a1= col1a1_ch_import_TH(file_path)
            splt_char = "/"
            n = len(file_path.split('/')) #get number of elements in list created by splitting file path
            save_path = "/".join(file_path.split("/", n)[:-1])  #save path is the same directory as where the file was found
            cv2.imwrite(save_path + '/' + filename + '_TH.png', col1a1)
            # col1a1.save(save_path + '/' + filename + '_TH.png')
            print('thresholded col1a1 saved at '+ save_path)
        if 'hunu' in filename and 'Segm' in filename and not 'TH' in filename:
            hunu = TH_local_otsu(file_path,30,1)
            n = len(file_path.split('/')) #get number of elements in list created by splitting file path
            save_path = "/".join(file_path.split("/", n)[:-1])  #save path is the same directory as where the file was found
            hunu.save(save_path + '/' + filename + '_TH.png')
            print('thresholded hunu saved at '+ save_path)
