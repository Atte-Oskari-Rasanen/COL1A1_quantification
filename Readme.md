# Quantification of COL1A1+ cells

Link to the Github page: https://github.com/Atte-Oskari-Rasanen/COL1A1_quantification 
## Motivation

An analysis workflow was built to quantify the number of COL1A1+ cells in IHC images. More specifically, the workflow has been finetuned to work with COL1A1-VectorBlue/HuNu-DAB IHC images although it can work with other stains as well. The images should be of relatively clean quality in order for the segmentation to work although basic image cleaning is also performed by the workflow.


## Requirements

ImageJ(Fiji) can be downloaded from the following website: https://imagej.net/software/fiji/downloads
It is recommended that ImageJ is installed inside the analysis folder (inside Fiji.app). On the project github page
the ImageJ version found inside Fiji.app is for Linux. The directory original_images (inside Fiji.app) is not part of the directories that come along with installed ImageJ. It is instead placed there since it makes running certain analysis steps more streamlined. It is recommended to add own images inside this folder in a format of creating a numbered subdirectory containing the IHC images of the certain group/animal, as shown with the example images.

The workflow script has been written on Ubuntu 20.04 and works primarily on Linux systems although Mac should suffice as well. For running the workflow on windows, for instance Windows Subsystem for Linux (WSL) can be used although this has not been explicitly tested.

For installing the Python packages, conda is recommended since an yml file has been readily created from which the user can install the correct package versions using the command:

```
conda env create -f conda_env_file.yml
```
Instructions for installing conda on your system: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html 

## Repository overview

Quantification_COL1A1
├── README.md
├── conda_env_file.yml
├── saved_models
│     └── trad_unet_256_64.h5
│     │
│     ...
│     
├── Fiji.app
│   ├── original_images
│   │
│   └── macros
│   │    ├── 1_Deconvolution.ijm
│   │     └── 4_Remove_particles_WS.ijm
│   ...
│
└── Python_scripts
    ├── 2_File_organise_segment.py
    ├── 3_Stain_channels_postprocess.py
    ├── 5_Colocalise_stains.py
    ├── 6_Stats_calculation.py
    ├── Models_unet_import.py
    └── smooth_tiled_divisions_edited2.py
    ├──analysis_scripts
    └──model_training



## The Workflow

Example data is included inside Fiji.app - original_images directory.

The workflow consists of python scripts (.py) found inside Python_scripts folder and java (.ijm)
scripts found inside Fiji.app macros subfolder.

The workflow can be run as a whole with two initial adjustments; 1. The first script, 1_Deconvolution.ijm,
needs to be run separately by opening the ImageJ macro IDE (ImageJ --> Quick search -> Macro)
and 2. The user needs to open 4_Remove_particles_WS.ijm and add the input directory to the 'input' variable manually (the full path to the Deconvolved_ims folder). Furthermore, the user needs to specify the ImageJ version if the user is not using Linux OS (currently /Fiji.app/ImageJ-linux64). Alternatively to the step 2 the user can open the  4_Remove_particles_WS.ijm on ImageJ macro IDE and enter the directory via GUI when the user has commented out 'input' and removed the '//' from the front of the line starting #@.

The reason for having to pass the arguments into the ImageJ macros by opening the scripts is due to the fact that ImageJ is not generally speaking created for running via terminal. Some processes had been deprecated with the latest version, potentially affecting the way macros are run via terminal. However, earlier versions did not contain the deconvolution algorithm needed for the analysis. Thus, the compromise was made where some manual handling is required from the user.

Inside the original_images directory each animal is inside their own subfolders designated by number.
When 2_File_organise_segment.py is run, each HuNu channel and COL1A1 channel image is designated
a matching, unique IDs which are used later for colocalisation. After the ID designation, Deconvolved_ims
directory is created inside original_images along with subfolders corresponding to the animal numbers.
Inside these folders COL1A1 subfolder and HuNu subfolder are created. The corresponding images from the original folders are transferred here. HuNu channel images are subsequently segmented, thresholded and watershedded and the COL1A1 channel images
are thresholded.

In case the user wants to use the trained Residual or Attention-Residual U-net model in the workflow,
they can be downloaded from: https://www.dropbox.com/sh/j462f1szxd7xnza/AABhETE-6CUskf9olUOVjZqpa?dl=0. The rest of the
models are found inside ./Quantification_COL1A1/saved_models. The former models could not be saved to Github due to their size. trad_unet_256_64.h5 model is the most optimised one out of the models and is thus the recommended one. The model is not
as efficient at identifying smaller or clumped nuclei as the trad_unet_own_data_256_32.h5. However, this one has the issue
of capturing more background noise, which may not always be removed after postprocessing steps.

## Approach 1
With 1_Deconvolution.ijm, the macro asks for the user to enter the input directory, which is recommended to be the directory called original_images inside Fiji.app folder as well as deconvolution option (A,B,C). A corresponds to the optimised deconvolution, B corresponds to the Hematoxylin-DAB deconvolution and C corresponds Hematoxylin and eosin stain deconvolution. The macro will automatically generate the COL1A1 channel image and HuNu channel images into the directory where the images were originally saved. After this, the workflow can be run as usual using the command:

```
bash ImageQuantification.sh
```

Note that if you are not using Linux, you should change the step 4's ImageJ app to the one specific for your OS. Currently
It is ./Fiji.app/ImageJ-linux64. 


When the script is run on the terminal, the script asks for the path to the original_images folder, patch size along with the path to the model that will be used for segmenting the images. The patch size refers to the size of the patches into which
the imported image is cut into. This must correspond to the patch size with which the model was trained with. 
An example output when the script is run and example input:
```
1. Input directory:
./Fiji.app/original_images
2. Patch size:
256
3. Model path:
./saved_models/trad_unet_256_64.h5  
```

## Approach 2
After running 1_Deconvolution.ijm, the user can run the following scripts in the following order

###### After deconvolution:
```
bash Part1.sh
```
###### Watershed and removal of small particles:
Open 4_Remove_particles_WS.ijm on ImageJ IDE and run the script by clicking 'Run'. The script requires you to specify the input directory.

###### After watershedding:
```
bash Part2.sh
```

## Approach 3
Each step (except step 1) can be run separately from the command line by passing the required arguments:

###### Step 1 - Deconvolve stains by opening the following script on ImageJ macro IDE and click run
```
1_Deconvolution.ijm
```

###### Step 2 - Reorganise files and segment

```
python ./Python_scripts/2_File_organise_segment.py Input Patch_size Model_path
```

###### Step 3 - Postprocess the segmented images
```
python ./Python_scripts/3_Stain_channels_postprocess.py Deconvolved_ims
```

###### Step 4 - Apply watershed to the thresholded images. Change the ImageJ-linux64 to the OS being used in case
it is not Linux.
```
./Fiji.app/ImageJ-linux64 --ij2 --headless --console --run ./Fiji.app/macros/4_Remove_particles_WS.ijm
```

###### Step 5 - Colocalise the COL1A1 channel image with HuNu channel image, producing an image with COL1A1+ cells
```
python ./Python_scripts/5_Colocalise_stains.py Deconv_dir
```

###### Step 6 - Calculate the relevant statistics
```
python ./Python_scripts/6_Stats_calculation.py Deconv_dir
```


## For augmenting data:
The respective locations can be entered via command line.
```
python albument_augmentation.py images_path masks_path img_augmented_path msk_augmented_path
```

## Model training
Models_unet_import.py contains the different U-nets that can be imported into the main file. trad_Unet.py file was used for training the traditional U-net originally. However, the U-net can also be imported from Models_unet_import.py like the residual and attention-residual ones. The files are imported into the main script (trad_Unet.py or Models_unet_import.py)
by using import_images_masks_patches.py script. The function import_images imports images from the target directory when
all images are found inside it and import_masks imports the masks from corresponding mask folder. However, if the user 
decides to use the folder structure found with Kaggle Datascience Bowl (DSB) 2018 data, then the user can use import_kaggledata function. All of these functions are found in import_images_masks_patches.py. In case one wants to put the DSB data into the format of training folder containing all images and mask folder for all masks, then the user can use kaggle_reformat.py. The script requires to import the DSB data as npy, either via using the . This can be done via import_images_masks_patches.py and afterwards save the imported numpy arrays as .npy file. In case the user wants to reformat the DSB data, this data can then be imported into kaggle_reformat.py which organised the images into allocated training and mask directories. 

#singularity file

Definition file contents:
```
Bootstrap: library
From: ubuntu:20.04

%post
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common
    add-apt-repository universe
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        /opt/conda/bin/conda env create -f new_tf.yml
        python3 \
        python3-tk \
        python3-pip \
        python3-distutils \
        python3-setuptools
    # Reduce the size of the image by deleting the package lists we downloaded,
    # which are useless now.
    rm -rf /var/lib/apt/lists/*
    # Install Python modules.
    pip3 install -I joblib wheel scipy==1.4.1 tensorflow==2.6.0 pip install keras==2.6.* focal-loss scikit-learn==0.22.1 numpy opencv-python pandas matplotlib tqdm scandir scikit-image PyOpenGL thinc
```
To create the singularity image from the definition file (./Quantification_COL1A1/Python_scripts/model_training/ML_train.def), use
```
singularity build --remote ML_conda.sif ML_train.def
```

The models were trained within singularity container. An example script template is shown below:
```
#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-87 -p alvis
#SBATCH --gpus-per-node=V100:2 #-C 2xV100 # (only 2 V100 on these nodes, thus twice the RAM per gpu)
#SBATCH -t 0-24:00:00
#SBATCH -o UNET_ALL_diffunet_faster_dr0_bs128_scik_ep5_256.out
#SBATCH --mail-type=ALL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=at0400ra-s@student.lu.se   # Email to which notifications will be sent


#get to the WD
cd /cephyr/NOBACKUP/groups/snic2021-23-496

unzip ./All_data/Masks_by_batches_all.zip -d $TMPDIR/All_data
unzip ./All_data/Test_set.zip -d $TMPDIR/All_data
unzip ./All_data/Train_by_batches_all.zip -d $TMPDIR/All_data

#bs64 s512 tallennettu job_logs3:seen

#Running the scripts
singularity exec ML_conda.sif python3 ./scripts/tradUnet.py 256 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs/AAA_tradU_U2_s256_bs32_ep8Max_1e5_faster_dr01.log
singularity exec ML_conda.sif python3 ./scripts/tradUnet.py 512 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ >./job_logs/AAA_tradU_U2_s512_bs32_ep8Max_1e5_faster_dr01.log
singularity exec ML_conda.sif python3 ./scripts/tradUnet.py 736 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs/AAA_ALL_tradU_s736_bs32_ep8Max_1e5_faster_dr01.log
```


## Plotting data and performing linear regression
plot_history.py file contains steps used for performing linear regression analysis and calculating the relevant statistics and the stages used for plotting the training metrics for each epoch. The script also contains the generation of the heatmaps for the metrics tables. The user can make the appropriate changes based on the data.


## Statistics for the results
pretests_groups.R was used for running the tests for normality, homoscedasticity and plotting the data distribution as histograms for the data prior to choosing the significance test. Subsequently t_tests.R script contains the code for running t tests on samples or the non parametric alternative Wilcoxon test. Violinplots.R script contains the code used for generating the violin plots for the data. The data plotted on the script serves as an example and can be changed based on which data is plotted.
