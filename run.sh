#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 200
#SBATCH --time=30-00:00:00
#SBATCH --job-name=cnd_200
#SBATCH --output=cnd_200w_50kit.out
#SBATCH --error=cnd_200w_50kit.err
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python3 mcorbit/main.py --mpi -d ~/MCOrbit/dat/ --nmax 50000 --walkers 200 --out 20190313
