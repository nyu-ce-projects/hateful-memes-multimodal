#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=24:10:00
#SBATCH --mem=64GB
#SBATCH --job-name=mmsage
#SBATCH --output=./logs/mmsage_rtx8000_%j.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate ssl-gnn

cd /scratch/am11533/multimodal-gnn-attn

# python main.py --batchsize 16 --epochs 20 --optim Adam --model MMSAGE --workers 4 --lr 0.01

python main.py --batchsize 16 --epochs 1 --optim Adam --model MMSAGE --workers 4 --lr 0.01 --resume ./checkpoints/MMSAGE/0.547_0.5/