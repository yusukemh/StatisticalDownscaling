#!/bin/bash
#SBATCH --partition=sadow
#SBATCH --account=sadow
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time 12:00:00
#SBATCH --error=error.err
#SBATCH --output=output.out
#SBATCH --mem=20G
echo hello
srun module load lang/Anaconda3
srun source activate climate
## srun /opt/apps/software/lang/Anaconda3/2020.11/bin/python test.py
srun /home/yusukemh/.conda/envs/climate/bin/python3 test.py