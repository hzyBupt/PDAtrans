#!/bin/bash

# 清除 LD 变量避免 bash 报错
unset LD_PRELOAD
unset LD_LIBRARY_PATH

export nnUNet_raw="/ai/data/nnUNet_raw/"
export nnUNet_preprocessed="/ai/data/nnUNet_preprocessed/"
export nnUNet_results="/ai/code/nnUNet_results/"
export PYTHONPATH=/ai/code/nnUNet/nnunetv2:$PYTHONPATH

export nnUNet_n_proc_DA=8  # number of process to accelerate the training process