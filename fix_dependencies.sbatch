#!/bin/bash
#SBATCH -J fix_deps
#SBATCH -o fix_deps.out
#SBATCH -e fix_deps.err
#SBATCH -t 00:10:00
#SBATCH -p compute
#SBATCH -n 1

echo "Starting dependency fix"

# Load miniconda and initialize
module load miniconda
eval "$(conda shell.bash hook)"

# Activate environment
conda activate capstone_project

# Fix the dependency issue - either downgrade markupsafe or upgrade jinja2
echo "Installing compatible versions of markupsafe and jinja2"
pip install markupsafe==2.0.1 jinja2==3.0.0

# Check versions
echo "Checking installed versions:"
pip list | grep -E 'markupsafe|jinja2'

echo "Fix completed"
