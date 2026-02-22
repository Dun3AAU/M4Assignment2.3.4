#!/bin/bash
#SBATCH --job-name=prompts
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

# 1. Create logs directory
mkdir -p logs

source .venv/bin/activate

python3 baselineTrainEvalExport.py