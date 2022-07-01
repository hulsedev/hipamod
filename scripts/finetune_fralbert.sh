#!/bin/bash
#SBATCH --job-name=finetune_fralbert
#SBATCH --output=trace/output.txt
#SBATCH --error=trace/error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12         # Ask for 6 CPUs
#SBATCH --gres=gpu:32gb:2              # Ask for 1 GPU
#SBATCH --mem=64G                 # Ask for 10 GB of RAM
#SBATCH --time=3:00:00            # The job will run for 10 minutes

module load python/3.7
module load cuda/10.2/cudnn/7.6
module load cudatoolkit/10.2
source $HOME/live/bin/activate
FORCE_CUDA=1 python3 latency/pilot/qwant/finetune_fralbert.sh