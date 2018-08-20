#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 100
#SBATCH --time=00:10:00
#SBATCH --job-name=CNDFIT
#SBATCH --output=CND.out
#SBATCH --error=CND.err

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python main.py

