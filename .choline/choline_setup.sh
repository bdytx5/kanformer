#!/bin/bash
# Download Miniconda installer
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# Install Miniconda
# bash miniconda.sh -b -p $HOME/miniconda
# Initialize conda
# . $HOME/miniconda/bin/activate
# conda init
# Create environment
# conda create --name choline python=3.10 -y
# Activate environment
# conda activate choline
# Install vim
export cholineremote=true
sudo apt upgrade
sudo apt install vim -y
sudo apt install python3.10 -y
sudo apt install python-is-python3 -y
sudo apt install python3-pip -y
# Set Wandb API key without user interaction
export WANDB_API_KEY=436e73733905370c882fac5ff2f642fe309fefdd
# Log in to Hugging Face CLI
echo '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' | huggingface-cli login --stdin
echo 'n' | huggingface-cli whoami
pip install torch  || pip3 install torch  -y
pip install huggingface || pip3 install huggingface -y
pip install transformers || pip3 install transformers -y
pip install tiktoken || pip3 install tiktoken -y
pip install  || pip3 install  -y
