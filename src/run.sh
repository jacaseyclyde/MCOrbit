#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --job-name=CNDFIT
#SBATCH --output=CND.out

module load intel-python3
module load mvapich2-2.2/intel
mpiexec python main.py