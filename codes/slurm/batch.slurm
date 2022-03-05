#!/bin/bash
##SBATCH --account=sadow
## CHECK FILE NAME

#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time 6:00:00
#SBATCH --error=./outputs/error_%j.err
#SBATCH --output=./outputs/output_%j.out
#SBATCH --mem=64G
#SBATCH --mail-user=yusukemh@hawaii.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
echo hello
srun module load lang/Anaconda3
srun source activate climate
## srun /opt/apps/software/lang/Anaconda3/2020.11/bin/python test.py
srun /home/yusukemh/.conda/envs/climate/bin/python3 run_scripts/elevation/xg_boost.py