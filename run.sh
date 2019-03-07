#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 100
#SBATCH --time=10-00:00:00
#SBATCH --job-name=cnd_100
#SBATCH --output=cnd_100w_10kit.out
#SBATCH --error=cnd_100w_10kit.err
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python3 mcorbit/main.py --mpi -d ~/MCOrbit/dat/ --nmax 10000 --walkers 100 --sub 0.005
