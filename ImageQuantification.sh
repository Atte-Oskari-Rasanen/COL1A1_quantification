#!/bin/sh

#Analysis Workflow for quantifying COL1A1+ cells

set -e #terminates the script if an error occurs in any of the subscripts

#Part 1: Image convolution using ImageJ. The macro Batchprocess.ijm takes in the input and output directories.
#The file suffix must be .tif. 

echo "1. Input directory: $Input"
read Input
echo "2. Patch size: $Patch_size"
read Patch_size
echo "3. Model path: $Model_path"
read Model_path

Deconv_dir="${Input}/Deconvolved_ims"

#Reorganise files and segment using python

python ./Python_scripts/2_File_organise_segment.py $Input $Patch_size $Model_path


echo "Files organised and segmented"

Deconv_dir="${Input}/Deconvolved_ims"
echo "Deconv directory: $Deconv_dir"

python ./Python_scripts/3_Stain_channels_postprocess.py $Deconv_dir


#Removal of large particles and watershed.
./Fiji.app/ImageJ-linux64 --ij2 --headless --console --run ./Fiji.app/macros/4_Remove_particles_WS.ijm

echo "REMOVED LARGE/SMALL PARTICLES"

python ./Python_scripts/5_Colocalise_stains.py $Deconv_dir

#Record the statistics
python ./Python_scripts/6_Stats_calculation.py $Deconv_dir


echo "Calculations done! The csv file saved into Deconvolv_ims directory."



