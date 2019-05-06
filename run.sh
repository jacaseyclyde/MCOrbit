#!/bin/sh
#SBATCH --partition=nodes
#SBATCH -n 126
#SBATCH --time=30-00:00:00
#SBATCH --job-name=cnd_south
#SBATCH --output=cnd_south.out
#SBATCH --error=cnd_south.err
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module purge

module load intel-python3
module load mvapich2-2.2/intel

mpiexec python3 mcorbit/main.py --mpi -d ~/MCOrbit/dat/ --nmax 100000 --walkers 504 --out apsides -s 0.5 --sample
