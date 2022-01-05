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

#$TMPDIR/kagglezip/kaggle_data/
#singularity exec ML_conda.sif python3 ./scripts/different_unets_nodatagen.py 256 16 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_dataMasks_by_batches/Masks/ $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/
#singularity exec ML_conda.sif python3 ./scripts/tradUnet.py 256 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs/AAA_tradU_U2_s256_bs32_ep8Max_1e5_faster_dr01.log
#singularity exec ML_conda.sif python3 ./scripts/tradUnet.py 512 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ >./job_logs/AAA_tradU_U2_s512_bs32_ep8Max_1e5_faster_dr01.log
#singularity exec ML_conda.sif python3 ./scripts/tradUnet.py 736 64 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs/AAA_ALL_tradU_s736_bs32_ep8Max_1e5_faster_dr01.log

#sing diff vika 96 oliki 64! eli siis 736
########singularity exec ML_conda.sif python3 ./scripts/different_unets_nodatagen.py 256 16 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_$
#singularity exec ML_conda.sif python3 ./scripts/different_unets_nodatagen.py 256 128 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL_ep20 $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs4_dif/AAA_ALL_res_att_U2_s256_bs128_ep20Max_1e5_faster_dr01.log
#singularity exec ML_conda.sif python3 ./scripts/different_unets_nodatagen.py 512 128 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL_ep20 $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs4_dif/AAA_ALL_res_att_U2_s512_bs128_ep20Max_1e5_faster_dr01.log
singularity exec ML_conda.sif python3 ./scripts/different_unets_nodatagen.py 736 128 $TMPDIR/All_data/Train_by_batches/Images/ $TMPDIR/All_data/Masks_by_batches/Masks/ ALL_ep20 $TMPDIR/All_data/Test_set/Images/ $TMPDIR/All_data/Test_set/Masks/ > ./job_logs4_dif/AAA_ALL_res_att_U2_s736_bs128_ep20Max_1e5_faster_dr01.log
#singularity exec unet_1_tf_gpu.sif python3 ./scripts/v3_DS_U_net_in_batches.py $TMPDIR/Train_by_batches/ $TMPDIR/Masks_by_batches/ $TMPDIR/Al$
#unet_1_old_tf2_0.sif
#unet_1_final.sif
