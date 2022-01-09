#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import cv2
from skimage import measure, filters
import os 
import pandas as pd
import sys
import glob
from PIL import Image, ImageFilter

import matplotlib.pyplot as plt
def calculations_hunu(img):
    #create an empty list of different values that will be recorded from the image
    stats_list = []
    binary = np.asarray(img).astype(np.uint8)
    #get contours of the binary image
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    #calculate the area of the contours by iterating over all of them (saved in cnts variabe)
    #and save into the contour_areas list
    contour_areas = []
    for cell in range(len(cnts)):
        contour_area = cv2.contourArea(cnts[cell])
        contour_areas.append(contour_area)
    #sum up the contents of the contour_areas list and append to the stats_list
    total_area_nuclei = sum(contour_areas)
    # print('Total area taken up by the nuclei: '+ str(sum(contour_areas)))
    stats_list.append(total_area_nuclei)
    # binary = cv2.bitwise_not(binary)
    
    #count the number of cells. measure lables gets the arrays of the cells and max()
    #provides the total number of them
    binary = measure.label(binary) 
    cell_count_coloc=binary.max() 
    stats_list.append(cell_count_coloc)
    #count portion that the cells take from the total area. not needed but calculated 
    #as a security check
    total_a = binary.shape[0] * binary.shape[1]
    portion_from_total = round(total_area_nuclei / total_a, 5)
    # contour_area = cv2.contourArea(contours[0])
    stats_list.append(portion_from_total)
    return(stats_list)

def calculations_col1a1(img):
    stats_list = []
    binary = np.asarray(img).astype(np.uint8)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
    contour_area = cv2.contourArea(cnts[0])

    contour_areas = []
    for cell in range(len(cnts)):
        contour_area = cv2.contourArea(cnts[cell])
        contour_areas.append(contour_area)

    total_area_col1 = sum(contour_areas)
    # print('Total area taken up by the nuclei: '+ str(sum(contour_areas)))
    stats_list.append(total_area_col1)
    total_a = binary.shape[0] * binary.shape[1] 
    portion_from_total = round(total_area_col1 / total_a, 5)
    stats_list.append(portion_from_total)

    stats_list.append(total_a)
    return(stats_list)

def Global_otsu(img):
    thresh = threshold_otsu(img)
    im_gray = Image.fromarray(img)
    im_blur = im_gray.filter(ImageFilter.GaussianBlur(5))
    im_blur = np.asarray(im_blur)
    
    (T, threshInv) = cv2.threshold(im_blur, thresh, 255,
    	cv2.THRESH_BINARY_INV)
    threshInv=np.asarray(threshInv)
    return(threshInv)


segm_TH_dirs = []
all_ims_paths = []

main_dir = sys.argv[1]
#get all image paths:
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]

print('Main dir: ' + main_dir)
#get all images that match the pattern
# matches_list = []
for f in all_ims_paths:
    # print(f)
    im_name = f.split('/')[-1]
    # print(im_name)
    n = 18
    im_id = a = [im_name[i:i+n] for i in range(0, len(im_name), n)] #extracts the animal id and the code of the image
    im_id = im_id[0]
    match_hunu_col1 = list(filter(lambda x: im_id in x, all_ims_paths))
    # matches_list.append(match_hunu_col1)
    for file_path in match_hunu_col1:
        filename = os.path.basename(file_path)
hunu_stats_dict = {} #the value part contains: [filename, nuclei A, nuclei N, total_A]
hunu_coloc_stats_dict = {} 

from skimage.filters import threshold_otsu, rank

#gather the information and calculate the stats (no of nuclei and total area they take up)
# for the hunu image with all nuclei and for the colocalised image with the nuclei
# colocalised with col1a1 as well as col1a1 area.

#The first set of loops focuses on extracting the info from hunu images (pure hunu and the
# colocalised ones). The second loop goes through the col1a1 ones. They should be matched to the
#colocalised image with the corresponding id and this should be added into the column.
import scandir
a = []
All_files = []
for root, subdirectories, files in scandir.walk(main_dir):
    for subdir in subdirectories:
        filepaths = root + '/' + subdir
        files = glob.glob(filepaths +'/*.png')

        All_files.append(files)

for root, subdirectories, files in scandir.walk(main_dir):
    for subdir in subdirectories:
        #calculate the colocalised images stats
        if 'Coloc' in subdir:
            subdir_p = root + '/' + subdir + '/'
            files_subdir = glob.glob(subdir_p + '*.png')
            for file in files_subdir:
                if 'WS' in file:
                    im_name = file.split('/')[-1]
                    filename = im_name.split('.')[0]
                    # print(file)
                    hunu_coloc_stats_list = []
                    im_path = file
                    # print('IM TO OPEN:' + im_path)
                    im_gray = cv2.imread(im_path,0)
                    # im_th = Global_otsu(im_gray)  #pois
                    # im_gray = cv2.bitwise_not(im_gray)  #pois

                    Stats = calculations_hunu(im_gray)
                    hunu_coloc_stats_dict[filename]=Stats
                    
        if 'hunu' in subdir:
            # print('hunu path:')
            subdir_p = root + '/' + subdir + '/'
            # print('subdir path: '+ subdir_p)
            # files_subdir_h = glob.glob(subdir_p + '*WS.png')
            files_subdir_h = glob.glob(subdir_p +'*WS*')
            for file_h in files_subdir_h: 
                if 'WS' in file_h:
                    hunu_stats_list = []
                    im_name = file_h.split('/')[-1]
                    filename = im_name.split('.')[0]
                    # print('name: ' + filename)
                    # print(file)
                    # filenames.append(file)
                    im_path = file_h
                    # print('im path: ' + im_path)

                    im_gray = cv2.imread(im_path, 0)
                    # im_gray = Global_otsu(im_gray)
                    im_gray = cv2.bitwise_not(im_gray) #need to invert colours since we had the colours other way around after imagej watershedding etc
                    # plt.imshow(im_gray)
                    Stats = calculations_hunu(im_gray)
                    hunu_stats_dict[filename]=Stats

#Now we have a dictionary with the key being the imagename (pure hunu image or coloc one) and values
#containing the relevant info (Nuclei Area, Nuclei count and total Area of the image-this last one is a sanity check)

#Iterate over the COL1A1 images, count the area of each one's stain, then match to the dict key of the hunu_coloc dict

#get all im paths so that you can get the matching colocalised hunu with col1a1 stains

all_ims_paths = []
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]

col1a1_dict = {}
for root, subdirectories, files in scandir.walk(main_dir):
    for subdir in subdirectories:
        if 'col1a1' in subdir:
            subdir_p = root + '/' + subdir + '/'
            files_subdir = glob.glob(subdir_p + '*.png')
            for file in files_subdir:
                im_name = f.split('/')[-1]
                if 'col1a1_TH' in file:
                    im_name = file.split('/')[-1]
                    filename_c = im_name.split('.')[0]
                    filenames.append(filename_c)
                    im_path = file
                    im_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
                    stats_col1 = calculations_col1a1(im_gray)
                    col1a1_dict[filename_c] = stats_col1


#get the matches of coloc_hunu, hunu and col1a1 names. Then retrieve the appropriate values like the area from the value part 
#of the dict and do the calculations with the corresponding col1a1 value.
def getList(dict):
    return dict.keys()

final_info = {}
stats = []
ids = []


Ahunucol1_Acol1_I_Ahunus = []
Acol1_I_Nhunus = []
Acol1_I_Ahunus = []
Total_hunu_cells_list = []
hunu_colocs = []
N_hunu_coloc_I_total_hunus = []
A_col1_I_total_hunus = []
A_col1_I_N_hunus = []
IDs = []
Total_COL1A1_cells_list=[]
Acol1_I_Ahunus = []

Ncoloc_Ahunus = []
Ncoloc_Nhunus = []

#find matching hunu, coloc_hunu and col1a1 ones, get the calculations, save into dict as list value, the key will be the id info of the animal
i = 0
for key_hunu, h_v in hunu_stats_dict.items():
    for key_hunu_coloc, hc_v in hunu_coloc_stats_dict.items():
        for key_col1, h_col_v in col1a1_dict.items():

            if key_hunu_coloc[:18] == key_col1[:18] and key_hunu[:18]==key_col1[:18]:
                i += 1
                im_name = key_col1.split('col1a1')[0]
                # file_id = '_'.join(im_name[:8]), '_'.join(im_name[8:])
                # file_id = file_id[0]

                ID = im_name
                # print('ID --- '+ID[:18])
                ids.append(ID)
                
                
                #retrieve the stats from all the 3 dictionaries
                values_hunu_coloc = hunu_coloc_stats_dict[key_hunu_coloc]
                values_col1 = col1a1_dict[key_col1]
                values_hunu = hunu_stats_dict[key_hunu]
                
                
                # #I-symbol used as division
                
                #Perform the different calculations
                
                #Area of hunu-col1a1+ cells / Area of all hunu cells
                Ahunucol1_Acol1_I_Ahunu = values_hunu_coloc[0] / values_hunu[0]
                Ahunucol1_Acol1_I_Ahunus.append(Ahunucol1_Acol1_I_Ahunu)
                stats.append(Ahunucol1_Acol1_I_Ahunu)
                
                #Area of col1a1+ / Area of all hunu cells -- A(COL1A1+)/A(HUNU+)
                Acol1_I_Ahunu = values_col1[0] / values_hunu[0]
                Acol1_I_Ahunus.append(Acol1_I_Ahunu)
                stats.append(Acol1_I_Ahunu)

                
                #A(COL1A1+HUNU+)/N(HUNU+)
                Acol1_I_Nhunu = values_hunu_coloc[0] / values_hunu[1]
                Acol1_I_Nhunus.append(Acol1_I_Nhunu)
                stats.append(Acol1_I_Nhunu)

                #All hunu cells
                Total_hunu_cells = values_hunu[1]
                Total_hunu_cells_list.append(Total_hunu_cells)
                stats.append(Total_hunu_cells)
                
                
                #A(COL1A1+)/N(HUNU+)
                A_col1_I_N_hunu = values_col1[0] / values_hunu[1]
                A_col1_I_N_hunus.append(A_col1_I_N_hunu)


                #N(COL1A1+HuNu+)/N(Hunu+)             
                Ncoloc_Nhunu = values_hunu_coloc[1] / values_hunu[1]
                Ncoloc_Nhunus.append(Ncoloc_Nhunu)
                
                #N(COL1A1+HuNu+/AHunu+)
                Ncoloc_Ahunu = values_hunu_coloc[1]/ values_hunu[0]
                Ncoloc_Ahunus.append(Ncoloc_Ahunu)

                #COL1A1+ cells
                Total_COL1A1_cells = values_hunu_coloc[1]
                Total_COL1A1_cells_list.append(Total_COL1A1_cells)
                stats.append(Total_COL1A1_cells)

                final_info[ID]=stats

import pandas as pd

#print(final_info)
filenames = final_info.items()
All_stats = list(filenames)
df = pd.DataFrame(list(zip(ids, Ahunucol1_Acol1_I_Ahunus, Acol1_I_Nhunus, Acol1_I_Ahunus, A_col1_I_N_hunus, Total_hunu_cells_list,Total_COL1A1_cells_list, Ncoloc_Nhunus, Ncoloc_Ahunus)))


df.columns = [ "Animal_ID", "A(COL1A1+HUNU+)/A(HUNU+)", "A(COL1A1+HUNU+)/N(HUNU+)", "A(COL1A1+)/A(HUNU+)", "A(COL1A1+)/N(HUNU+)", "N(HUNU+)", "N(COL1A1+HUNU+)", "N(COL1A1+HuNu+)/N(Hunu+)", "N(COL1A1+HuNu+)/(AHunu+)"]

df = df.drop_duplicates()

df = df.round(7)

from datetime import date

run_date = date.today()

output_dir_results = main_dir
with open(output_dir_results + "/Results_" +str(run_date)+".csv", "w") as out: #write the dataframe into an output file
#write the dataframe into an output file
    df.to_csv(out, sep='\t')
    # df.to_string(out, index=None)
    print('output info file saved!')
