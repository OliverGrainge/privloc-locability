#!/bin/bash

#SBATCH --job-name=cvae
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=swarm_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G


echo "Starting sequential run..."

python train.py --config runs/conditionalvae/basic/config.yaml
python test.py --config runs/conditionalvae/basic/config.yaml

echo "All tasks finished."