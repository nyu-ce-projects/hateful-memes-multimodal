#!/bin/bash
#SBATCH --job-name=hpml-4xRTX8000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mp6021@nyu.edu

echo "Submitting job : main.py for Multimodal_finetunning-CASS-V2-CASS-loss-multi-para-optim-100-BEST-nodp-aggressive-down with 4 GPUs"

module purge
module load anaconda3/2020.07
module load cuda/11.3.1
module load python/intel/3.8.6
eval "$(conda shell.bash hook)"
conda activate idls
cd /scratch/mp6021/repo/hateful

python main.py


