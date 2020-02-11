#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0-999

#SBATCH --job-name=synthetic
#SBATCH --mem=40GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 09:00:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --qos=cbmm
#SBATCH --partition=cbmm
#SBATCH -D /om/user/vanessad/synthetic_framework/slurm_output/exp_4

hostname
module add openmind/singularity/3.4.1

arrayexperiments=(0 1000 2000)

for i in "${arrayexperiments[@]}"
do
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
python /om/user/vanessad/synthetic_framework/main.py \
--host_filesystem om \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--offset_index $i \
--run train \
--dataset_name dataset_5 \
--repetition_folder_path 5
done



