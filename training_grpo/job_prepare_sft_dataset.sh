#!/bin/bash
#SBATCH --job-name=prepare-sft-dataset
#SBATCH --output=/export/home/acs/stud/t/tudor.farcasanu/SSL_research/logs/prepare_sft_%j.out
#SBATCH --error=/export/home/acs/stud/t/tudor.farcasanu/SSL_research/logs/prepare_sft_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source ~/miniconda3/bin/activate
conda activate verl_env

cd /export/home/acs/stud/t/tudor.farcasanu/SSL_research
python training_grpo/prepare_sft_dataset.py \
    --input_file generated_finetuning_data/grpo_finetuning_dataset.jsonl \
    --output_file training_grpo/sft_dataset.jsonl \
    --include_hints
