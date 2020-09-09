#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=12-35
#SBATCH --job-name=foveation
#SBATCH --mem=40GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 36:00:00
#SBATCH --qos=cbmm
#SBATCH --partition=cbmm
# SBATCH -D /om/user/vanessad/IMDb_framework/slurm_output/


hostname
module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-vanessa.simg \
python /om/user/vanessad/IMDb_framework/main.py \
--host_filesystem om \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--offset_index 0 \
--run train \
--repetition_folder_path kim_init_weights

