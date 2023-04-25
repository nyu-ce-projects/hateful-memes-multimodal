#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=24:10:00
#SBATCH --mem=64GB
#SBATCH --job-name=vgae_gat
#SBATCH --output=logs/vgae_gat_cc_%j.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate ssl-gnn-env

cd /scratch/am11533/ssl-gnn-multimodal

# python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model GMAE --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/ --dataset HatefulMeme
# python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model GMAE --workers 4 --lr 0.01 --data_path ../datasets/cc12m/ --dataset ConceptualCaption

# python main.py --batchsize 16 --epochs 20 --optim Adam --model MMSAGE --workers 4 --lr 0.01

python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model VGAE_UNSUPERVISED --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/ --dataset HatefulMeme

# nohup python -u ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model VGAE_UNSUPERVISED --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/ > output_vgae_pretrain_distilbert_$(date +%s).log &

# python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model VGAE_UNSUPERVISED --workers 4 --lr 0.01 --data_path ../datasets/cc12m/ --dataset ConceptualCaption

# nohup python -u ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model MLP_EMBED_CLASSIFIER --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/  --resume ./checkpoints/VGAE_v2 > output_ar.log &

# source ~/.bashrc
# cd /scratch/am11533/ssl-gnn-multimodal
# conda activate ssl-gnn-env
