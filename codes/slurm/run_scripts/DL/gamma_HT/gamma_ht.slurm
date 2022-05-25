#!/bin/bash
##SBATCH --account=sadow
## CHECK FILE NAME

#SBATCH --partition=sadow
#SBATCH --account=sadow
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time 12:00:00
#SBATCH --error=./error_%j.err
#SBATCH --output=./output_%j.out
#SBATCH --mem=32G
#SBATCH --mail-user=yusukemh@hawaii.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export PATH="/home/yusukemh/sadow_lts/personal/yusukemh/Anaconda3/bin:$PATH"
source activate tfp
python gamma_ht.py