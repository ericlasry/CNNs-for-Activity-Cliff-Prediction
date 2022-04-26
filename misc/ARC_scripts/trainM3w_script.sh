#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=QUICK
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --constraint='gpu_mem:32GB'
#SBATCH --mem=256000
#SBATCH --mail-user=eric.lasry@keble.ox.ac.uk
#SBATCH --mail-type=ALL

module load Anaconda3/2021.11
module load CUDA/11.4.1-GCC-10.3.0
module load cuDNN/8.1.1.33-CUDA-11.2.2

source activate $DATA/moon-env

python3 $DATA/notebooks/trainM3w.py