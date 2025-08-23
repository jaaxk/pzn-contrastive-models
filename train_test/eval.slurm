#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --output res.txt
#SBATCH --mem 250G

nvidia-smi
module load ffmpeg

source venv/bin/activate
python eval.py \
    --test_set ../data/test_tracks.csv
