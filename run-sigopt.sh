#!/bin/bash
#SBATCH -p compute
#SBATCH --time 48:00:00
#SBATCH -o slurm-%j.log
#SBATCH --cpus-per-task=8
#SBATCH --job-name=WolfQuap_sigopt

module load anaconda3/personal

sigopt optimize -e experiment.yml python sigopt-model.py > sigopt-model.out