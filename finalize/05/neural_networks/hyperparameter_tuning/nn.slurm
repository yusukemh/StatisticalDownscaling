#!/bin/bash


##SBATCH --account=sadow
#SBATCH --partition=kill-shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00
#SBATCH --error=./error_%j.err
#SBATCH --output=./output_%j.out
#SBATCH --mem=64G
#SBATCH --mail-user=yusukemh@hawaii.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
## echo hello
srun module load lang/Anaconda3
srun source activate climate
srun /home/yusukemh/.conda/envs/climate/bin/python3 nn.py