#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=24:10:00
#SBATCH --mem=64GB
#SBATCH --job-name=vgae_classifier
#SBATCH --output=logs/vgae_classifier_%j.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate ssl-gnn-env

cd /scratch/am11533/ssl-gnn-multimodal

# python main.py --batchsize 16 --epochs 20 --optim Adam --model MMSAGE --workers 4 --lr 0.01

# python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model VGAE_UNSUPERVISED --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/

python ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model MLP_EMBED_CLASSIFIER --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/  --resume ./checkpoints/VGAE_UNSUPERVISED/0.8395604682843558_0.7575308748579077/

# nohup python -u ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model MLP_EMBED_CLASSIFIER --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/  --resume ./checkpoints/VGAE_v2 > output_ar.log &

# nohup python -u ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model MLP_EMBED_CLASSIFIER --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/  --resume ./checkpoints/VGAE_UNSUPERVISED/0.8395604682843558_0.7575308748579077/ > output_vgae_gat.log &

# ``nohup python -u ssl_gnn_multimodal/main.py --batchsize 16 --epochs 20 --optim Adam --model GMAE --workers 4 --lr 0.01 --data_path ../datasets/hateful_memes/ --dataset HatefulMeme > output_gmae_pretrain_$(date +%s).log &``