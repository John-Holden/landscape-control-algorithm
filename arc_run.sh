#!/bin/bash
###########__________Run script__________#############
################ Hpc machine ################
module load python/3.6.5
module load python-libs/3.1.0
#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -t 1
export HPC_MODE=TRUE

python3 scripts/over_beta_processing.py $SGE_TASK_ID