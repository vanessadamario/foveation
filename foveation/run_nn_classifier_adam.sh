#!/bin/bash


#SBATCH -n 1
#SBATCH --array=0-4
#SBATCH -c 2
#SBATCH --job-name=experiment1
#SBATCH --mem=12GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 20:00:00
#SBATCH --partition=cbmm
#SBATCH -D ./

hostname

module add openmind/singularity/3.4.1

arrayexp=(1 2 3 4)

for e_ in "${arrayexp[@]}"
do
  singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
python /om/user/vanessad/foveation/nn_classifier_adam.py \
-experiment_index=${SLURM_ARRAY_TASK_ID} --experiment_design=$e_
done