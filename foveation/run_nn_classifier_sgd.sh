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

for e_ in "${arrayexp[@]}"
do
  for d_ in "${arraydim[@]}"
  do
    singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
python /om/user/vanessad/foveation/nn_classifier_sgd.py \
-experiment_index=${SLURM_ARRAY_TASK_ID} --experiment_design=$e_ --data_dim=$d_
  done
done