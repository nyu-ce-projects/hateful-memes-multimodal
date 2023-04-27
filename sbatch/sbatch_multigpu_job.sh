#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --time=24:10:00
#SBATCH --mem=64GB
#SBATCH --job-name=mmgat
#SBATCH --output=./logs/mmgat_rtx8000_2gpu_%j.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate hpml-env

cd /scratch/am11533/multimodal-gnn-attn

# python main.py --batchsize 16 --epochs 20 --optim Adam --model MMSAGE --workers 4 --lr 0.01
python main.py --batchsize 16 --epochs 20 --optim Adam --model MMGAT --workers 4 --lr 0.01