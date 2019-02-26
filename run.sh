#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 250
#SBATCH --time=10-00:00:00
#SBATCH --job-name=cnd_250w2500i
#SBATCH --output=cnd_250w.out
#SBATCH --error=cnd_250w.err
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python3 mcorbit/main.py --mpi -d ~/MCOrbit/dat/ --nmax 2500 --walkers 250
