#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --time=6-06:00:00   # specify job time
#SBATCH --mem=16G         # specify memory
#SBATCH --output=job_output_PASNet_real_test.log

module load cuda-toolkit/12.2.2  # load necessary modules

srun python PASNet_real_test.py  # run your job
