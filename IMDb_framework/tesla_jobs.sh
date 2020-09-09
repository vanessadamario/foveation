#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=reproduce_txt_classification
#SBATCH --mem=10GB
#SBATCH -t 09:00:00
#SBATCH --gres=gpu:titan-x:1  tesla-k80:
#SBATCH -D /om/user/vanessad/IMDb_framework/slurm_output/reproduce_pipeline
#SBATCH --partition=use-everything

hostname
module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-vanessa.simg \
python /om/user/vanessad/IMDb_framework/main.py \
--experiment_index 2914 \
--offset_index 0 \
--host_filesystem om \
--run train \
--repetition_folder_path sst2_exps
