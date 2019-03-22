#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 504
#SBATCH --time=30-00:00:00
#SBATCH --job-name=cnd_fit
#SBATCH --output=cnd_fit.out
#SBATCH --error=cnd_fit.err
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python3 mcorbit/main.py --mpi -d ~/MCOrbit/dat/ --nmax 500000 --walkers 504 --out 20190322 -s 0.5
