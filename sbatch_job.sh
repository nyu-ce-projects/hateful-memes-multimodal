#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:10:00
#SBATCH --mem=16GB
#SBATCH --job-name=mmgnn
#SBATCH --output=./logs/mmgnn_%j.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate hpml-env

cd /scratch/am11533/multimodal-gnn-attn

python main.py --batchsize 128 --epochs 10 --lr=0.01