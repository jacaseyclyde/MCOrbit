#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 100
#SBATCH --time=00:01:00
#SBATCH --job-name=CNDFIT
#SBATCH --output=CND.out
#SBATCH --error=CND.err

module purge

module load intel-python3
module load openmpi-2.0/intel

mpiexec python main.py

