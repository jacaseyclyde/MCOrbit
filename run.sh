#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 200
#SBATCH --time=10-00:00:00
#SBATCH --job-name=cnd_200
#SBATCH --output=cnd_200w_5kit.out
#SBATCH --error=cnd_200w_5kit.err
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python3 mcorbit/main.py --mpi -d ~/MCOrbit/dat/ --nmax 5000 --walkers 200 --sub 0.005 --out 20190307
