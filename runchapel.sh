#!/bin/bash
#SBATCH --mem=0
#SBATCH --nodes=5
#SBATCH --cpus-per-task=192
echo Running on $SLURM_NNODES nodes

./block -nl $SLURM_NNODES --rows=50_000
./stencil -nl $SLURM_NNODES --rows=50_000

./block -nl $SLURM_NNODES --rows=100_000
./stencil -nl $SLURM_NNODES --rows=100_000

./block -nl $SLURM_NNODES --rows=200_000
./stencil -nl $SLURM_NNODES --rows=200_000
