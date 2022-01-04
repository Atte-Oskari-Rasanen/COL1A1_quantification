#!/bin/sh
set -e #terminates the script if an error occurs in any of the subscripts
set -o pipefail #if a subscript stops due to an error, the workflow won't be terminated

#The user can input the data as zipped file (each animal dir zipped) and the directory into which they'll be unzipped. The user also 
#needs to specify into which directory the output files are put into. This will create the relevant subdirectories inside the same
#directory as where the original files are unzipped to. OR the user lets the script to unzip the files into a temporary directory, process
#the files, save the stats and remove this temp directory. This saves memory especially if there are many images and/or large ones.

#temp_dir=$(mktemp -d)


#Part 1: Image convolution using ImageJ. The macro Batchprocess.ijm takes in the input and output directories.
#The file suffix must be .tif. 

echo "Input directory: $Input"
read Input
echo "Patch size: $Patch_size"
read Patch_size
echo "Model path: $Model_path"
read Model_path

Deconv_dir="${Input}/Deconvolved_ims"

#make macro location and file name as passable arguments later!
#./fiji-linux64/Fiji.app/ImageJ-linux64 --ij2 --headless --console --run ./Fiji.app/macros/1_Deconvolution.ijm 'folder=./macros/original_imag>

#Reorganise files and segment using python

python ./Python_scripts/2_File_organise_segment.py $Input $Patch_size $Model_path

echo "Files organised and segmented"


Deconv_dir="${Input}/Deconvolved_ims"
echo "Deconv directory: $Deconv_dir"

python ./Python_scripts/3_Stain_channels_postprocess.py $Deconv_dir
