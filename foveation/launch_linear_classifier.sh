#!/bin/bash



#SBATCH -n 1
#SBATCH --array=0-9
#SBATCH -c 2
#SBATCH --job-name=experiment1
#SBATCH --mem=12GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 20:00:00
#SBATCH --partition=cbmm
#SBATCH -D ./

hostname

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
python /om/user/vanessad/foveation/linear_classifier.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} --experiment_design=1 --dataset_dimension=28

