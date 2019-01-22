#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 100
#SBATCH --time=24:00:00
#SBATCH --job-name=CNDFIT
#SBATCH --output=CND.out
#SBATCH --error=CND.err

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python main.py --mpi -d ~/MCOrbit/dat/ --nmax 5000 --walkers 100
