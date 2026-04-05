#!/bin/bash
#SBATCH --job-name=restructure_reports
#SBATCH --output=mllm/preprocessing/cache-%j.out
#SBATCH --nodes=1
#SBATCH --nodelist=compute-cuda-02
#SBATCH --cpus-per-task=10

#SBATCH --mem=48GB
#SBATCH --partition=odap-gpu
#SBATCH --signal=USR2@120
#SBATCH --time=14400

source $HOME/.conda_rc
conda activate ollama
cd $HOME/projects/voxelFM
export PYTHONPATH=$PWD

python3 mllm/preprocessing/restructure_reports.py