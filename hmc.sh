#!/bin/bash
#SBATCH -p compute
#SBATCH --time 48:00:00
#SBATCH -o slurm-%j.log
#SBATCH --cpus-per-task=8
#SBATCH --job-name=hmc

module load anaconda3/personal

python hmc.py