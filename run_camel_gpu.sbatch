#!/bin/bash
#SBATCH -J camel_bert
#SBATCH -o camel_bert.out
#SBATCH -e camel_bert.err
#SBATCH -t 01:00:00
#SBATCH -p nvidia
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --mem=64G

module load miniconda
module avail -t 2>&1 | grep -i cuda | head -1 | xargs -r module load
eval "$(conda shell.bash hook)"
conda activate capstone_project

echo "Job started on $(date)"
echo "Running on $(hostname)"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

mkdir -p predictions_out/camelmorph/dev/
bash evaluate.bash

echo "Job completed on $(date)"