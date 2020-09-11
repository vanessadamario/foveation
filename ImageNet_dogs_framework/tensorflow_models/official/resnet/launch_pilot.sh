#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 2
#SBATCH --job-name=resnet
#SBATCH --gres=gpu:tesla-k80:2
#SBATCH --mem=20GB
#SBATCH -t 00:30:00
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1
export PYTHONPATH="$PYTHONPATH:/om/user/vanessad/ImageNet_dogs_framework/tensorflow_models/official/resnet"

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python imagenet_main.py  \
--data_dir=/om/user/vanessad/ImageNet_dogs_framework/TFRecords \
--num_gpus=2 \
--batch_size=64 \
--train_epochs=20 \
--model_dir=/om/user/vanessad/ImageNet_dogs_framework/tensorflow_models/official/resnet/pilot_cropped


