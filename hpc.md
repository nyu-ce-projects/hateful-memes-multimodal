module load python/intel/3.8.6

curl -sSL https://install.python-poetry.org | POETRY_HOME=/scratch/am11533/poetry python3 -




conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c huggingface transformers datasets