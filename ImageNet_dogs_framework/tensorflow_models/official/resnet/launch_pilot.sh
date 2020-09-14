#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 4
#SBATCH --job-name=resnet
#SBATCH --gres=gpu:tesla-k80:8
#SBATCH --mem=30GB
#SBATCH -t 03:00:00
#SBATCH --qos=cbmm

export PYTHONPATH="$PYTHONPATH:/raid/poggio/home/vanessad/ImageNet_dogs_framework/tensorflow_models/official/resnet"

singularity exec --nv /raid/poggio/home/xboix/containers/xboix-tensorflow1.14.simg \
python imagenet_main.py  \
--data_dir=/raid/poggio/home/vanessad/data/TFRecords \
--num_gpus=8 \
--batch_size=128 \
--train_epochs=90 \
--crop_image=True \
--model_dir=/raid/poggio/home/vanessad/resnet_experiments/foveation/ImageNet_dogs_framework/tensorflow_models/official/resnet/no_crop_all_data


