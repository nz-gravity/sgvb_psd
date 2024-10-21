#!/bin/bash
#SBATCH --job-name=et_analysis       # Job name
#SBATCH --array=0-9                    # Job array indices from 0 to 9
#SBATCH --time=5:00:00                 # Time limit (5 hours)
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --mem=4G                       # Memory per node
#SBATCH --output=et_logs/psd_analysis_%A_%a.out # Output file name with array index
#SBATCH --error=et_logs/psd_analysis_%A_%a.err  # Error file name with array index

# Load necessary modules (if applicable)
# module load python/3.8  # Adjust to your Python version or environment


python et_study.py --case "caseA" --label $SLURM_ARRAY_TASK_ID
