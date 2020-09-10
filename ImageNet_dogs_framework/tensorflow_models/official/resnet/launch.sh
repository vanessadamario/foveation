#!/bin/bash
#SBATCH -n 1
#SBATCH --array=0
#SBATCH --nodelist=node0[29]
#SBATCH -c 24
#SBATCH --job-name=resnet
#SBATCH --mem=24GB
#SBATCH -t 50:00:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm


export PYTHONPATH="$PYTHONPATH:/om/user/xboix/src/models/"

cd /om/user/xboix/src/models/official/resnet

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om  --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python imagenet_main.py  \
--data_dir=/om/user/xboix/data/ImageNet \
--num_gpus=8 \
--batch_size=512 \
--model_dir=/om/user/xboix/src/models/official/resnet/models/resnet50


