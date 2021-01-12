#!/bin/bash
#SBATCH --mem=20g --ntasks-per-node=16 -N 1
#SBATCH -t 1:00:00
#SBATCH -A carney-sjones-condo

/users/ntolley/anaconda/sbi/bin/python /users/ntolley/Jones_Lab/sbi_hnn_github/code/beta/python_example.py