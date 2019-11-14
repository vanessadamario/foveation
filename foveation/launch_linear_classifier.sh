#!/bin/bash

#SBATCH --nodes=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vanedamario@gmail.com

module add openmind/singularity/3.4.1

array=(28 36 40 56 80)

for k in ${array[@]}
do
    echo $k
    singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python linear_classifier.py 1 $k
done
