#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=QUICK
#SBATCH --mem=256000
#SBATCH --mail-user=eric.lasry@keble.ox.ac.uk
#SBATCH --mail-type=ALL

module load Anaconda3/2021.11

source activate $DATA/moon-env

python3 $DATA/notebooks/trainM3w.py