#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vanedamario@gmail.com

for k in {0..10}
do
    python testBash.py $k 
done
