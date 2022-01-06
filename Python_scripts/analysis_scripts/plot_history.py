#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 19:22:38 2021

@author: atte
"""

import pandas as pd
import pandas.io.common
import numpy as np
import matplotlib.pyplot as plt
import os


csv_dir = '/home/atte/Documents/Scripts_image_analysis/Plotting_data_PD'
import scipy.stats as st

#data A(col1a1+hunu+)/a(hunu+)
data = [['2_1877526603_256_ject_TileScan_006_Merging_Crop_ch00_', 0.309, 0.321], 
        ['7_1877526590_256_roject_TileScan_010_Merging_Crop_ch00_', 0.4,  0.407],
        ['7_1877526585_256_roject_TileScan_012_Merging_Crop_ch00_',0.341, 0.416],
        ['6_1877526546_256_ject_TileScan_027_Merging_Crop_ch00_',0.509, 0.368], 
        ['14_1877526616_256_14_2572284805_256_14_103913837_256_14_403752692_736_Project_TileScan_013_Merging_Crop_ch00_]', 0.355, 0.408], 
        ['20_1877526598_256__Project_TileScan_020_Merging_Crop_ch00__hunu_manual_th_WS.png', 0.501, 0.441], 
        ['6_1877526556_256_ject_TileScan_026_Merging_Crop_ch00__hunu_manual_th_WS.png', 0.65, 0.429],
        ['12_1877526572_256_roject_TileScan_029_Merging_Crop_ch00__hunu_manual_th_WS.png', 0.6726, 0.475],
        ['18_1877526543_256_roject_TileScan_032_Merging_Crop_ch00__hunu_manual_th_WS.png',0.56, 0.333]]



#data all cells
data = [['2_1877526603_256_ject_TileScan_006_Merging_Crop_ch00_', 361, 226], 
        ['7_1877526590_256_roject_TileScan_010_Merging_Crop_ch00_', 127, 103],
        ['7_1877526585_256_roject_TileScan_012_Merging_Crop_ch00_',222, 117],
        ['6_1877526546_256_ject_TileScan_027_Merging_Crop_ch00_',621, 355], 
        ['14_1877526616_256_14_2572284805_256_14_103913837_256_14_403752692_736_Project_TileScan_013_Merging_Crop_ch00_]', 75, 43], 
        ['20_1877526598_256__Project_TileScan_020_Merging_Crop_ch00__hunu_manual_th_WS.png', 183, 236], 
        ['6_1877526556_256_ject_TileScan_026_Merging_Crop_ch00__hunu_manual_th_WS.png', 855, 355],
        ['12_1877526572_256_roject_TileScan_029_Merging_Crop_ch00__hunu_manual_th_WS.png', 801, 359],
        ['18_1877526543_256_roject_TileScan_032_Merging_Crop_ch00__hunu_manual_th_WS.png',615, 339]]


#data col1a1+ cells
data = [['2_1877526603_256_ject_TileScan_006_Merging_Crop_ch00_', 290, 196], 
        ['7_1877526590_256_roject_TileScan_010_Merging_Crop_ch00_', 117, 99],
        ['7_1877526585_256_roject_TileScan_012_Merging_Crop_ch00_',189, 98],
        ['6_1877526546_256_ject_TileScan_027_Merging_Crop_ch00_',509, 326], 
        ['14_1877526616_256_14_2572284805_256_14_103913837_256_14_403752692_736_Project_TileScan_013_Merging_Crop_ch00_]', 60, 41], 
        ['20_1877526598_256__Project_TileScan_020_Merging_Crop_ch00__hunu_manual_th_WS.png', 263, 219], 
        ['6_1877526556_256_ject_TileScan_026_Merging_Crop_ch00__hunu_manual_th_WS.png', 750, 481],
        ['12_1877526572_256_roject_TileScan_029_Merging_Crop_ch00__hunu_manual_th_WS.png', 423, 335],
        ['18_1877526543_256_roject_TileScan_032_Merging_Crop_ch00__hunu_manual_th_WS.png',492, 298]]



import statistics
#Ahunu+col1a1+/Nhunu+    automatically generated data  Rsqu=0.001050, p_value=0.9340
data = [['2_1877526603_256_ject_TileScan_006_Merging_Crop_ch00_', 226, 0.321], 
        ['7_1877526590_256_roject_TileScan_010_Merging_Crop_ch00_', 103,  0.407],
        ['7_1877526585_256_roject_TileScan_012_Merging_Crop_ch00_',117, 0.416],
        ['6_1877526546_256_ject_TileScan_027_Merging_Crop_ch00_',355, 0.368], 
        ['14_1877526616_256_14_2572284805_256_14_103913837_256_14_403752692_736_Project_TileScan_013_Merging_Crop_ch00_]', 43, 0.408], 
        ['20_1877526598_256__Project_TileScan_020_Merging_Crop_ch00__hunu_manual_th_WS.png', 236, 0.441], 
        ['6_1877526556_256_ject_TileScan_026_Merging_Crop_ch00__hunu_manual_th_WS.png', 355, 0.429],
        ['12_1877526572_256_roject_TileScan_029_Merging_Crop_ch00__hunu_manual_th_WS.png', 359, 0.475],
        ['18_1877526543_256_roject_TileScan_032_Merging_Crop_ch00__hunu_manual_th_WS.png',339, 0.333]]

#Ahunu+col1a1+/Nhunu+    automatically generated data



#Running linear regression and calculating the relevant statistics

# Create the pandas DataFrame
df_lin_reg = pd.DataFrame(data, columns = ['Image', 'Manual A(COL1A1+HUNU+)/A(HUNU+)', 'Segm A(COL1A1+HUNU+)/A(HUNU+)'])


d = np.polyfit(df_lin_reg['Manual A(COL1A1+HUNU+)/A(HUNU+)'],df_lin_reg['Segm A(COL1A1+HUNU+)/A(HUNU+)'],1)
f = np.poly1d(d)

df_lin_reg.insert(3,'Treg',f(df_lin_reg['Manual A(COL1A1+HUNU+)/A(HUNU+)']))


from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.model_selection import train_test_split
x = df_lin_reg.iloc[:, 1].values.reshape(-1, 1)  # iloc[:, 1] is the column of X
y = df_lin_reg.iloc[:, 2].values.reshape(-1, 1)  # df.iloc[:, 4] is the column of Y

x1 = np.reshape(x,9)
y1 = np.reshape(y,9)

np.polyfit(x1,y1,1)

linear_regressor = LinearRegression()
model = linear_regressor.fit(x, y)
y_pred = linear_regressor.predict(x)
x_pred =linear_regressor.predict(y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)



def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    # print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

# solution
a, b = best_fit(x1, y1)

# plot points and fit line
import matplotlib.pyplot as plt

plt.scatter(x1, y1)
yfit = [a + b * xi for xi in x1]
m, b = np.polyfit(x1, y1, 1)
m
b
plt.plot(x1, y1, 'o')

plt.plot(x1, m*x1 + b)

#rename the labels based on which data used
plt.ylabel('U-net N(COL1A1+HUNU+)', fontsize=10)
plt.xlabel('Manual N(COL1A1+HUNU+)', fontsize=10)

plt.plot(x1,yfit,color='red')
plt.scatter(x,y)

plt.show()

plt.plot(x1, yfit, color='red')

rsquared = sklearn.metrics.r2_score(y, y_pred, sample_weight=None, multioutput='uniform_average')
print(rsquared)



#get the p value
import statsmodels.api as sm
from scipy import stats
x2 = sm.add_constant(x)
est = sm.OLS(y, x2)
print(est.fit().f_pvalue)

##################################################
##################################################

#the directories are ones used on personal computer, can be changed
df1 = pd.read_csv('/home/atte/Documents/Scripts_image_analysis/Plotting_data_PD/bs64/Unet_history_df_10_512_ALL_noep_2021-11-28_64.csv')
df2 = pd.read_csv('/home/atte/Documents/Scripts_image_analysis/Plotting_data_PD/bs128/Unet_history_df_10_512_ALL_noep_2021-11-28_128.csv')

#for plots generation
for subdir, dirs, files in os.walk(csv_dir):
    print(subdir)
    if 'bs' in subdir:
        for file in files:
            if file.endswith('.csv') and len(file)>12 and not 'F' in file:
    
            # print(os.path.join(subdir, file))
                csv_df_path=os.path.join(subdir, file)
                csv_df = file
                print(csv_df)
                # try:
                df = pd.read_csv(csv_df_path)
                
                df = pd.DataFrame(df, columns= ['Unnamed: 0','loss', 'val_loss'])
                
                plt.rcParams["figure.autolayout"] = True
                # print(file)
    
                filename = csv_df.split('.')[0]
                # print(filename)
                filename = filename.split('_')
                # print(filename)
                try:
                    title= filename[4] + '_' + filename[-1] + 'Loss'
        
        
                    headers = ['epochs']
                    save_name = subdir + '/' + title + '.png'
                    df.set_index('Unnamed: 0').plot()
                    plt.xlabel('Epochs', fontsize=18)
                    plt.ylabel('Loss', fontsize=10)
        
                    # plt.suptitle(title, fontsize=20)
                    plt.savefig(save_name, bbox_inches='tight')
        
                    plt.show()
                except IndexError:
                    pass
                # except pandas.io.common.EmptyDataError:
                #     pass
    

for subdir, dirs, files in os.walk(csv_dir):
    # print(subdir)
    if 'bs' in subdir:
        # dfs_bs = []
        for file in files:
            if file.endswith('.csv') and len(file)>12 and not 'F' in file:
    
            # print(os.path.join(subdir, file))
                csv_df_path=os.path.join(subdir, file)
                csv_df = file
                # try:
    
                df = pd.read_csv(csv_df_path)
                df = pd.DataFrame(df, columns= ['Unnamed: 0','dice_coef', 'val_dice_coef'])
                # df = pd.DataFrame(df, columns= ['Unnamed: 0','f1_m', 'val_f1_m'])

                plt.rcParams["figure.autolayout"] = True
                # print(file)
    
                filename = csv_df.split('.')[0]
                # print(filename)
                filename = filename.split('_')
                # print(filename)
                try:
                    title= filename[4] + '_' + filename[-1] + 'Accuracy'
        
        
                    headers = ['epochs']
                    save_name = subdir + '/' + title + 'F1.png'
                    df.set_index('Unnamed: 0').plot()
                    plt.xlabel('Epochs', fontsize=10)
                    plt.ylabel('Accuracy', fontsize=10)
        
                    # plt.suptitle(title, fontsize=20)
                    plt.savefig(save_name, bbox_inches='tight')
        
                    plt.show()
                except IndexError:
                    pass
            # except pandas.io.common.EmptyDataError:
            #     pass


csv_dir = '/home/atte/Documents/Scripts_image_analysis/Plotting_data_PD/'



for subdir, dirs, files in os.walk(csv_dir):
    # print(subdir)
    if 'bs' in subdir:
        dfs_bs = []
        print(subdir)
        for file in files:
            if file.endswith('.csv') and len(file)>12 and not 'F' in file:
                print(file)
            # print(os.path.join(subdir, file))
                csv_df_path=os.path.join(subdir, file)
                csv_df = file
                # try:
    
                df = pd.read_csv(csv_df_path)
                df = pd.DataFrame(df, columns= ['Unnamed: 0','dice_coef', 'val_dice_coef'])
                # df = pd.DataFrame(df, columns= ['Unnamed: 0','f1_m', 'val_f1_m'])
                filename = csv_df.split('.')[0]
                # print(filename)
                filename = filename.split('_')
                # print(filename)
                
                title= [filename[4] + '_' + filename[-1]] * len(df)
    
                df2 = df.insert(3,'File',title, True)

                # df = df.append(title, ignore_index = True)

                dfs_bs.append(df)
        
        #after you have saved the relevant info from the dataframses using different
        #then plot all them together
        for df_i in range(len(dfs_bs)):
            df_first = dfs_bs[0]
            ax = df_first.plot()
            next_i = df_i + 1
            next_df = dfs_bs[next_i]
            next_df.plot(ax=ax)


        merged_dfs = pd.concat(dfs_bs)

        plt.rcParams["figure.autolayout"] = True
        # print(file)
    
    
    
        headers = ['epochs']
        savefile = subdir.split('/')[-1]
        save_name = subdir + '/' + savefile + '_256_512_736.png'
        
        df.set_index('Unnamed: 0').plot()
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
    
        # plt.suptitle(title, fontsize=20)
        plt.savefig(save_name, bbox_inches='tight')
    
        plt.show()
            # except pandas.io.common.EmptyDataError:


#take val accuracies of each batch size, plot them 

for subdir, dirs, files in os.walk(csv_dir):
    #print(subdir)
    if 'bs' in subdir:
        print(subdir)
        for file in files:
            if file.endswith('.csv') and len(file)>12:
                csv_df_path=os.path.join(subdir, file)
                csv_df = file
                df = pd.read_csv(csv_df_path)
                # cols = ['dice_coef', 'jacard_coef', 'recall_m', 'precision_m', 'val_dice_coef', 'jacard_coef_loss']
                df = pd.DataFrame(df, columns=['dice_coef', 'jacard_coef', 'auc_1', 'val_dice_coef', 'val_jacard_coef', 'val_auc_1'])

                df = df.tail(1)

    
                filename = csv_df.split('.')[0]
                # print(filename)
                filename = filename.split('_')
                # print(filename)
                title= filename[4] + '_' + filename[-1]
                save_name = subdir + '/' + title + 'F.csv'
                # df.round({'dice_coef':3, 'jacard_coef':3, 'auc_1':3, 'val_dice_coef':3, 'val_jacard_coef':3, 'val_auc_1':3})
                # df = df.round(4)
                df.to_csv(save_name, sep='\t', encoding='utf-8',index=False)
    print('done!')
    
file = '/home/atte/Documents/Scripts_image_analysis/Plotting_data_PD/bs_Own/Unet_history_df_10_736_own_noep_2021-11-30_32.csv'
file_save = '/home/atte/Documents/Scripts_image_analysis/Plotting_data_PD/bs_Own/test.csv'
df = pd.read_csv(csv_df_path)
df2 = pd.DataFrame(df, columns= ['Unnamed: 0', 'dice_coef', 'f1_m', 'jacard_coef','precision_m', 'recall_m', 'val_dice_coef','val_jacard_coef', 'val_f1_m', 'val_precision_m', 'val_recall_m'])
cols= ['Unnamed: 0', 'dice_coef', 'f1_m', 'jacard_coef','precision_m', 'recall_m', 'val_dice_coef','val_jacard_coef', 'val_f1_m', 'val_precision_m', 'val_recall_m']
cols= ['Unnamed: 0', 'dice_coef', 'jacard_coef','recall_m', 'precision_m', 'f1_m','val_dice_coef', 'val_jacard_coef', 'val_recall_m', 'val_precision_m', 'val_f1_m']
df2 = pd.DataFrame(df, columns=cols)
df = df[df.columns[cols]]

df1 = df.tail(1)
print(df1)
df.to_csv(file_save, sep='\t', encoding='utf-8',index=False)

#heatmpats generation
from pandas.table.plotting import table
from pandas.plotting import table 
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('dataframe-image')
import dataframe_image as dfi

csv_df_path = "/home/atte/Documents/GitHub/Quantification_COL1A1/Python_scripts/analysis_scripts/metrics_data/Unet_comparisons.csv"
csv_df_path = "/home/atte/Documents/GitHub/Quantification_COL1A1/Python_scripts/analysis_scripts/metrics_data/Unets_own_data.csv"
csv_df_path = "/home/atte/Documents/GitHub/Quantification_COL1A1/Python_scripts/analysis_scripts/metrics_data/Unets_all_data.csv"


df = pd.read_csv(csv_df_path)
df.fillna(value='', inplace = True)


df.columns.str.match("Unnamed")
df.loc[:,~df.columns.str.match("Unnamed")]

csv_df_path_save = "/home/atte/Documents/GitHub/Quantification_COL1A1/Python_scripts/analysis_scripts/metrics_data"


df_styled = df.style.background_gradient() #adding a gradient based on values in cell
dfi.export(df_styled,csv_df_path_save + '/Unets_data_all_heatmap.png')

