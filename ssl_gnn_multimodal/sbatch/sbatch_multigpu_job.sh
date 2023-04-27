#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=24:10:00
#SBATCH --mem=64GB
#SBATCH --job-name=logs/vgae_classifier_multigpu

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate ssl-gnn-env

cd /scratch/am11533/ssl-gnn-multimodal

# python main.py --batchsize 16 --epochs 20 --optim Adam --model MMSAGE --workers 4 --lr 0.01

# python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model VGAE_UNSUPERVISED --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/

python ssl_gnn_multimodal/main.py --batchsize 8 --epochs 20 --optim Adam --model MLP_EMBED_CLASSIFIER --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/  --resume ./checkpoints/VGAE_v2