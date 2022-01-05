#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-87 -p alvis
#SBATCH --gpus-per-node=V100:2 #-C 2xV100 # (only 2 V100 on these nodes, thus twice the RAM per gpu)
#SBATCH -t 0-24:00:00
#SBATCH -o UNET_ALL_diffunet_faster_dr0_bs128_scik_ep5_256.out
#SBATCH --mail-type=ALL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=at0400ra-s@student.lu.se   # Email to which notifications will be sent

#overall this script is a template script showing how the training of the models was performed and not all
#the scripts were run on a single run but rather some of them were commented out based on what was trained.
#this was done due to the length of the scripts and since it made controlling separate steps in scripts easier.

#get to the WD
cd /cephyr/NOBACKUP/groups/snic2021-23-496

unzip ./All_data/Masks_by_batches_all.zip -d $TMPDIR/All_data
unzip ./All_data/Test_set.zip -d $TMPDIR/All_data
unzip ./All_data/Train_by_batches_all.zip -d $TMPDIR/All_data

#When training traditional U-net
singularity exec ML_conda.sif python3 ./scripts/tradUnet.py 256 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs/AAA_tradU_U2_s256_bs32_ep8Max_1e5_faster_dr01.log
singularity exec ML_conda.sif python3 ./scripts/tradUnet.py 512 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ >./job_logs/AAA_tradU_U2_s512_bs32_ep8Max_1e5_faster_dr01.log
singularity exec ML_conda.sif python3 ./scripts/tradUnet.py 736 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs/AAA_ALL_tradU_s736_bs32_ep8Max_1e5_faster_dr01.log

#when training attention residual and residual U-nets. The different_unets_nodatagen.py also contains the traditional U-net training in the end in case one
#wants to train all the U-nets in a single script.

singularity exec ML_conda.sif python3 ./scripts/different_unets_nodatagen.py 256 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL_ep20 $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs4_dif/AAA_ALL_res_att_U2_s256_bs128_ep20Max_1e5_faster_dr01.log
singularity exec ML_conda.sif python3 ./scripts/different_unets_nodatagen.py 512 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL_ep20 $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs4_dif/AAA_ALL_res_att_U2_s512_bs128_ep20Max_1e5_faster_dr01.log
singularity exec ML_conda.sif python3 ./scripts/different_unets_nodatagen.py 736 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL_ep20 $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs4_dif/AAA_ALL_res_att_U2_s736_bs128_ep20Max_1e5_faster_dr01.log
