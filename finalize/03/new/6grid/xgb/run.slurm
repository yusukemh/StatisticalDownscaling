#!/bin/bash
##SBATCH --account=sadow
## CHECK FILE NAME

#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time 10:00:00
#SBATCH --error=./error_%j.err
#SBATCH --output=./output_%j.out
#SBATCH --mem=128G
#SBATCH --mail-user=yusukemh@hawaii.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export PATH="/home/yusukemh/sadow_lts/personal/yusukemh/Anaconda3/bin:$PATH"
source activate climate
python3 xgb.py