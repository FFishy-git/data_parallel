#!/bin/bash
#SBATCH --job-name=c2s_vllm_dpp          # Job name
#SBATCH --partition=gpu                 # Partition name
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --ntasks=1                      # Single MPI task
#SBATCH --cpus-per-task=8               # CPUs for tokenization/DataLoader
#SBATCH --time=24:00:00                 # 24h walltime
#SBATCH --output=/gpfs/radev/project/zhuoran_yang/hs925/Bio_Transcoder/dataset_prepare/bio_data_check/slurm_outputs/%j.out
#SBATCH --error=/gpfs/radev/project/zhuoran_yang/hs925/Bio_Transcoder/dataset_prepare/bio_data_check/slurm_outputs/%j.err
#SBATCH --requeue
#SBATCH --export=NONE

echo "--------------------------------------"
cd ${SLURM_SUBMIT_DIR}
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "SLURM_JOBID=${SLURM_JOBID}"
echo "--------------------------------------"

# Activate your conda env
module load miniconda
eval "$(conda shell.bash hook)"
conda activate LLM2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/hs925/project/Bio_Transcoder/data_parallel

# Point HF cache to fast GPFS scratch
export HF_HOME="/gpfs/radev/scratch/${USER}/huggingface_cache"

# Run the new multi‐GPU vLLM DPP script
python lib/vllm_dpp_eval_c2s.py 

echo "Finished at $(date)"
