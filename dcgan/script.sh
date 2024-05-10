#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=128G
# Request 1 CPU core
#SBATCH -n 4
#SBATCH -t 24:00:00

module load gcc cuda

export PYTHON=/usr/bin/python

$PYTHON main.py