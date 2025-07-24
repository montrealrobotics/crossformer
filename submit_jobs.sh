#!/bin/bash

#HORIZON=(4 8 16 64)
HORIZON=(4)
SEED=( 235 368 10)


for h in "${HORIZON[@]}"; do
    for s in "${SEED[@]}"; do
        echo "Submitting job for horizon $h and seed $s"
        #./train_octo_libero.sh $h $s 
        sbatch  cross_former_quadruped.sh $s
done
done
