#!/bin/bash
##SBATCH --account=sadow
## CHECK FILE NAME

#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time 6:00:00
#SBATCH --error=./error_%j.err
#SBATCH --output=./output_%j.out
#SBATCH --mem=128G
#SBATCH --mail-user=yusukemh@hawaii.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
echo hello
srun module load lang/Anaconda3
srun source activate climate
## srun /opt/apps/software/lang/Anaconda3/2020.11/bin/python test.py
srun /home/yusukemh/.conda/envs/climate/bin/python3 random_forest.py