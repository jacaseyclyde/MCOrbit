#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 300
#SBATCH --time=30-00:00:00
#SBATCH --job-name=cnd_10sparse
#SBATCH --output=cnd_300_10sparse.out
#SBATCH --error=cnd_300_10sparse.err
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python3 mcorbit/main.py --mpi -d ~/MCOrbit/dat/ --nmax 50000 --walkers 300 --out 20190314b -s 0.1
