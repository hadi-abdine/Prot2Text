#!/bin/bash
#SBATCH --job-name=prot2text_base  # name of job
#SBATCH --output=prot2text%j.out  # output file (%j = job ID)
#SBATCH --error=prot2text%j.err # error file (%j = job ID)
#SBATCH --constraint=v100-32g # reserve GPUs with 32 GB of RAM
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1 # reserve 4 tasks (or processes)
#SBATCH --gres=gpu:4 # reserve 4 GPUs
#SBATCH --cpus-per-task=10 # reserve 10 CPUs per task (and associated memory)

set -x
export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -u -m torch.distributed.run \
         --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
          --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
          train.py \
             --decoder_path gpt2 \
             --esm_model_path facebook/esm2_t12_35M_UR50D \
             --use_plm \
             --use_rgcn \
             --warmup_esm \
             --warmup_gpt \    
             --split test \
             --data_path ./data//dataset/ \
             --train_csv_path ./data/train.csv \
             --eval_csv_path ./data/eval.csv \    
             --batch_per_device 4 \
             --nb_epochs 25 \
             --nb_gpus <number of gpus> \
             --gradient_accumulation 1 \ 
             --lr 2e-4 \ 
             --save_model_path ./models/prot2text_base/ \
             --bleu_evaluation'
