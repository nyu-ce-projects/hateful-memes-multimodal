#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=24:10:00
#SBATCH --mem=64GB
#SBATCH --job-name=vgae_classifier

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate ssl-gnn-env

cd /scratch/am11533/ssl-gnn-multimodal

# python main.py --batchsize 16 --epochs 20 --optim Adam --model MMSAGE --workers 4 --lr 0.01

# python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model VGAE_UNSUPERVISED --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/

python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model MLP_EMBED_CLASSIFIER --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/  --resume ./checkpoints/VGAE_v2

# nohup python -u ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model MLP_EMBED_CLASSIFIER --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/  --resume ./checkpoints/VGAE_v2 > output_ar.log &