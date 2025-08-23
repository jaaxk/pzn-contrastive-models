#!/bin/bash
#SBATCH -p long-40core
#SBATCH -t 48:00:00
#SBATCH --output res.txt

module load node.js
source venv/bin/activate

python -u get_dataset.py
