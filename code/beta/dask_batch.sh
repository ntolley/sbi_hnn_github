#!/usr/bin/env bash

#SBATCH -J dask-worker
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=19G
#SBATCH -t 01:00:00

/users/ntolley/anaconda/sbi/bin/python /users/ntolley/Jones_Lab/sbi_hnn_github/code/beta/python_example.py