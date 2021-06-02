#!/bin/bash  
#SBATCH  -J NeurifyJProp5
#SBATCH  -o NeurifyJProp5.%j.out
#SBATCH  -t 1-00:00:00
#SBATCH  --nodes=1 
#SBATCH  --ntasks=2
#SBATCH  --tasks-per-node=2
#SBATCH  --cpus-per-task=3

set echo on

cd  $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module  purge
module  load  postech

date 

julia ~/DNNVerification.jl/acasxu_tests/neurify/prop5/prop5_neurify.jl

date

squeue --job $SLURM_JOBID