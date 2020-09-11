#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 4
#SBATCH --job-name=resnet
#SBATCH --gres=gpu:tesla-k80:4
#SBATCH --mem=30GB
#SBATCH -t 01:00:00
#SBATCH --qos=cbmm

export PYTHONPATH="$PYTHONPATH:/om/user/vanessad/ImageNet_dogs_framework/tensorflow_models/official/resnet"

singularity exec --nv /raid/poggio/home/xboix/containers/xboix-tensorflow1.14.simg \
python imagenet_main.py  \
--data_dir=/raid/poggio/home/xboix/data/imagenet-tfrecords \
--num_gpus=4 \
--batch_size=128 \
--train_epochs=20 \
--model_dir=/raid/poggio/home/vanessad/resnet_experiments/foveation/ImageNet_dogs_framework/tensorflow_models/official/resnet/pilot


