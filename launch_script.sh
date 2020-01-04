#!/bin/bash

#SBATCH  -n1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg python generate_images.py
