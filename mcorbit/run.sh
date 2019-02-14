#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 100
#SBATCH --time=4-00:00:00
#SBATCH --job-name=CNDFIT
#SBATCH --output=CND.out
#SBATCH --error=CND.err
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python3 main.py --mpi -d ~/MCOrbit/dat/ --nmax 100000 --walkers 100
