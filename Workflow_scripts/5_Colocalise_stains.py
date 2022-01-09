#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:14:25 2021

@author: atte
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.filters import threshold_otsu
from PIL import Image, ImageFilter
from skimage import measure, filters
import scandir
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import os
from skimage.feature import peak_local_max

import sys


def colocalise(hunu_im, col1a1_im):
    
    #import the images, they are already thresholded but they are imported as non binary so
    #apply general thresholding to get the numpy array value as binary. 
    hunu_im = cv2.imread(hunu_im,0)
    col1a1_im = cv2.imread(col1a1_im,0)
    
    ret2,hunu_im = cv2.threshold(hunu_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret2,col1a1_im = cv2.threshold(col1a1_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h,w = col1a1_im.shape
    #the images should already be same size but resize hunu image according to col1a1 image
    #dimensions as a security check:
    hunu_im = Image.fromarray(hunu_im)
    hunu_im = hunu_im.resize((w,h))
    hunu_im = np.asarray(hunu_im)    
    #find contours of the col1a1 image
    cnts, _ = cv2.findContours(col1a1_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
    #create a mask of plain zeros based on hunu image
    out_mask = np.zeros_like(hunu_im)
    #draw the contours to this mask
    cv2.drawContours(out_mask, cnts, -1, 255, cv2.FILLED, 1)
    out=hunu_im.copy()
    
    #apply out_masks values to the original binary hunu image, visualising only the
    #nuclei within the contours
    out[out_mask == 0] = 255 #makes nuclei white on the black background
    #flip the colours so that the nuclei are white
    out = cv2.bitwise_not(out)
    return(out)
    print('colocalised image created!')

main_dir = sys.argv[1]
# main_dir = "/home/atte/Documents/GitHub/analysis_folder/Fiji.app/original_images/Deconvolved_ims"
coloc_dir = main_dir + '/Coloc'

try:
    os.mkdir(coloc_dir)
except OSError:
    print ("Failed to create directory %s " % coloc_dir)
else:
    print ("Succeeded at creating the directory %s " % coloc_dir)

ids= []
segm_TH_dirs = []
all_ims_paths = []
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]
#get all images that match the pattern
file_pairs = {} #key: hunu_ws_th file, value: col1a1_th
for f in all_ims_paths:
    filename = os.path.basename(f)
    if 'WS' in filename:
        for f2 in all_ims_paths:
            filename2 = os.path.basename(f2)
            if 'col1a1' in filename2 and 'TH' in filename2 and filename[:18] in filename2:
                file_pairs[f] = f2
#Now we perform colocalisation:
for hunu, col1a1 in file_pairs.items():
    filename = os.path.basename(hunu)
    filename = filename.split('.')[0]
    coloc_im = colocalise(hunu,col1a1)

    cv2.imwrite(coloc_dir +'/' + filename + "_Coloc.png",coloc_im)
    print('Colocalised image saved at :' + coloc_dir +'/' + filename + "_Coloc.png" )
print("ALL COLOCALISED")