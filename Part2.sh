#!/bin/sh
set -e #terminates the script if an error occurs in any of the subscripts
set -o pipefail #if a subscript stops due to an error, the workflow won't be terminated

echo "Enter the location to the Deconvolved_ims directory:"
read Input

python ./Python_scripts/5_Colocalise_stains.py $Input

#Record the statistics
python ./Python_scripts/6_Stats_calculation.py $Input


echo "Calculations done!"
