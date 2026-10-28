#!/bin/bash
#SBATCH --mem=0
#SBATCH --nodes=5
#SBATCH --tasks-per-node=192
echo Running with $SLURM_NTASKS tasks

mpirun -np $SLURM_NTASKS ./original50

mpirun -np $SLURM_NTASKS ./original100

mpirun -np $SLURM_NTASKS ./original200
