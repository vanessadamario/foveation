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

arraydim=(28 36 40 56 80)
arrayexp=(1)
arraylr=(0.001 0.005 0.01 0.05 0.1 0.2)

for e_ in "${arrayexp[@]}"
do
  for d_ in "${arraydim[@]}"
  do
    for lr_ in "${arraylr[@]}"
    do
      singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
python /om/user/vanessad/foveation/linear_classifier_sgd.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} --experiment_design=$e_ --dataset_dimension=$d_ --learning_rate=$lr_
    done
  done
done

