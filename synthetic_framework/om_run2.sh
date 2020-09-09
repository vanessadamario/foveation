#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=40-59,347-527,840-995
#SBATCH --job-name=synthetic
#SBATCH --mem=40GB
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -D /om/user/vanessad/synthetic_framework/slurm_output/scenario_4
#SBATCH --partition=use-everything

hostname
module add openmind/singularity/3.4.1

arrayexperiments=(3 4 5 6)

for i in "${arrayexperiments[@]}"
do
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
python /om/user/vanessad/synthetic_framework/main.py \
--host_filesystem om \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--offset_index 0 \
--run train \
--dataset_name dataset_$i \
--repetition_folder_path $i
done



