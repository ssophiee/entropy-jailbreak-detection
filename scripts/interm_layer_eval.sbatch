#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err

source ~/.bashrc
conda activate adv-attacks

cd ~/adv-attacks-blllm

python -m approaches.laplace_approx.scripts.compute_metrics_intermediate_layer \
  --model_path saved_models/llama31_8b_laplace \
  --safe_dataset xstest --harmful_dataset advbench \
  --n_test max --balance --run_all --log_file logs/run_xstest_advbench_llama8b_intermediate_layer.log
