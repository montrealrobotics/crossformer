#!/bin/bash

#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=unkillable
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --exclude=cn-a[001-011],cn-k003

#--gres=gpu:a100l:1 --mem 32G
module load miniconda/3

conda activate octo 

#python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --config.save_dir=./experiments/libero-finetune_batch_512/ --config.batch_size=512 --config.shuffle_buffer_size=75000

#python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --config.save_dir=./experiments/libero-finetune_batch_256/ --config.batch_size=256

#python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --config.save_dir=./experiments/libero-finetune_batch_128/ --config.batch_size=128

SAVEDIR="/home/mila/m/michael.przystupa/scratch/crossformer-quadruped/$2/$1/"

#SAVEDIR="/home/mila/m/michael.przystupa/scratch/octo_exp_action_heads/test/"

PARAMS="--config ./scripts/configs/finetune_quadruped_config.py --config.save_dir=$SAVEDIR --config.batch_size=128 --config.wandb.project=quadruped --config.num_steps=100000"

ARGS="$PARAMS --config.seed=$1 --config.dataset_kwargs.name=$2"
echo $ARGS
python scripts/finetune.py $ARGS 
