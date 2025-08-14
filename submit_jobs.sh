#!/bin/bash

#HORIZON=(4 8 16 64)
HORIZON=(4)
SEED=(235 10)

DATASET=("flat_quadruped_dataset" "history_proprioceptive_quadruped_dataset")


for d in "${DATASET[@]}"; do
    for s in "${SEED[@]}"; do
        echo "Submitting job for horizon $h and seed $s"
        #./train_octo_libero.sh $h $s 
        sbatch  cross_former_quadruped.sh $s $d
done
done
