#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
module load python3/3.7.7
module load cuda/9.0
module load cudnn/v7.4.2.24-prod-cuda-9.0
module load ffmpeg/4.2.2
pip3 install --user torch torchvision torchboardX transformers
echo "Running script..."
python3 hpc_train.py --data_dir data/FullDataset --max_sequence_length 60 --valid -ep 5
