#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:10:00
#SBATCH --mem=64GB
#SBATCH --job-name=vgae

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate ssl-gnn-env

cd /scratch/am11533/ssl-gnn-multimodal

# python main.py --batchsize 16 --epochs 20 --optim Adam --model MMSAGE --workers 4 --lr 0.01

python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model VGAE --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/