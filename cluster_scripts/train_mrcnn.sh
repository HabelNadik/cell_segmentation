#!/bin/bash

#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH -o "/nobackup/lab_gsf/bhaladik/ExTrAct-AML_methods/logs/nucleus_mrcnn_training/mrcnn_nucleus_training_out_%j.txt"
#SBATCH -e "/nobackup/lab_gsf/bhaladik/ExTrAct-AML_methods/logs/nucleus_mrcnn_training/mrcnn_nucleus_training_err_%j.txt" 
#SBATCH --mail-user=bhaladik@cemm.oeaw.ac.at
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=end
#SBATCH --mem=256000

echo "======================"
echo $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NAME
echo $SLURM_JOB_PARTITION
echo $SLURM_NTASKS
echo $SLURM_NPROCS
echo $SLURM_JOB_ID
echo $SLURM_JOB_NUM_NODES
echo $SLURM_NODELIST
echo $SLURM_CPUS_ON_NODE
echo "======================" 

## input folder, output folder, sample id, cellprofiler pipeline and dapi channel should be submitted to this script with  the --export flag
echo "Starting training for nucelus MRCNN"

module load Python/3.8.2-GCCcore-9.3.0
export PATH="/home/bhaladik/.local/bin:$PATH"


python /research/lab_gsf/bhaladik/ExTrAct-AML_methods/tf_2_4_1_keras_2_4_3_code/samples/nucleus/nucleus.py train --dataset=/nobackup/lab_gsf/bhaladik/ExTrAct-AML_methods/mrcnn_training_data --subset=stage1_train --weights=last