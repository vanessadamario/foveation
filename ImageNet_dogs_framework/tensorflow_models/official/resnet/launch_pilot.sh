#!/bin/bash
#SBATCH -n 1
#SBATCH --array=0
#SBATCH -c 2
#SBATCH --job-name=resnet
#SBATCH --mem=2GB
#SBATCH -t 00:20:00
#SBATCH --qos=normal

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python imagenet_main.py  \
--data_dir=/om/user/vanessad/ImageNet_dogs_framework/TFRecords \
--num_gpus=1 \
--batch_size=64 \
--model_dir=/om/user/vanessad/ImageNet_dogs_framework/tensorflow_models/official/resnet/pilot


