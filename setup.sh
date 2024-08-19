#!/bin/bash

# Create and activate conda environment
conda create -n myenv python=3.10 -y
conda activate myenv

# Install all required packages
conda install -c conda-forge stable-baselines3 pytorch torchvision torchaudio matplotlib numpy ipykernel scipy seaborn scikit-learn -y

# Add the environment as a Jupyter kernel
python -m ipykernel install --user --name=myenv

echo "Setup complete. You can now activate the environment with 'conda activate myenv'"
