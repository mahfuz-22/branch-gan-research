"""
Experimental Training Script for Branch-GAN Research

This script contains implementation attempts of Branch-GAN with varying configurations,
conducted as part of research at the University of Auckland investigating Branch-GAN's
variable branch depths in resource-constrained environments.

Original experiments were conducted on:
- Google Colab Pro (A100 GPU, 40GB memory)
- IBM Watson (2x V100 GPU, 32GB memory each)

Hardware Requirements (Based on Implementation Analysis):
- Minimum 40GB+ GPU memory for full implementation
- Training time: 40-50 hours per configuration
- Recommended batch size: 8 for stable gradient updates
- Minimum sample size: 12,000 for adequate pattern learning
- Model architecture: 12+ layers, 12+ attention heads

Implementation Challenges Encountered:
1. Memory Management:
   - Multiple sequence copies during branching led to exponential memory growth
   - Even high-end GPUs (A100, V100) couldn't accommodate initial 10,000 sample configuration

2. Training Stability:
   - Reduced batch sizes impacted gradient quality
   - Early termination due to memory constraints
   - Limited model capacity with reduced layers/heads

Dataset:
Custom dataset combining:
- OpenWebText (50%): Web content
- Wikipedia (25%): Encyclopedic knowledge
- BookCorpus (25%): Literary content
Total samples: 12,000 (planned), processed with Pythia-410m tokenizer

Models:
- Generator: Based on Pythia-410M architecture
- Discriminator: Based on Pythia-14M for efficiency

Research Focus:
Investigating impact of variable branch depths (d = {4, 8, 12, 16})
on text generation quality and computational efficiency.

Author: Mahfuz Rahman
School of Computer Science, University of Auckland
Date: 23 October 2024
"""

# Initial setup for Branch-GAN experimental implementation
# This script handles repository setup and verification of required components

# Standard library imports
import os
import shutil
import json
import pickle
from datetime import datetime
import importlib

# Third-party imports
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from google.colab import drive

# Local imports (from project modules)
from ConfigInstances import PileTraining
from Training import SpawnedExperiment, TrainingLoop  # Added TrainingLoop here
from Models import FileManagement

# Repository configuration
# Using public GitHub repository containing Branch-GAN implementation experiments
# conducted at the University of Auckland
repo_url = "https://github.com/mahfuz-22/branch-gan-research.git"
expected_dir = 'branch-gan-research'

# Directory and environment setup
# Ensures correct working directory for experiments with varying branch depths
# and computational resource analysis
if not os.path.basename(os.getcwd()) == expected_dir:
    if os.path.exists(expected_dir):
        %cd {expected_dir}
    else:
        !git clone {repo_url}
        %cd {expected_dir}

print(f"Current working directory: {os.getcwd()}")

# Verify existence of critical components for Branch-GAN implementation
# These components are essential for our experimental configurations
# across different computational environments (Colab Pro, IBM Watson)
expected_files = ['Models', 'Training', 'ConfigInstances', 'StartTraining.py']
missing_files = [f for f in expected_files if not os.path.exists(f)]

if missing_files:
    print(f"Missing essential components: {missing_files}")
    print("Reinitializing repository for clean experimental setup...")
    %cd ..
    if os.path.exists(expected_dir):
        shutil.rmtree(expected_dir)
    !git clone {repo_url}
    %cd {expected_dir}
else:
    print("Experimental environment successfully initialized with all required components.")


# Dependency verification for Branch-GAN implementation
# Checks required packages for experiments across different computational environments
# (Google Colab Pro with A100 GPU and IBM Watson with V100 GPUs)
import importlib

def check_package(package_name):
    """
    Verify installation and version of required packages.
    Critical for reproducing experimental conditions across different
    computing environments used in the research.
    """
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"{package_name} version: {version}")
    except ImportError:
        print(f"{package_name} is not installed")

# Essential packages for Branch-GAN implementation:
# - torch: Base deep learning framework
# - torchvision: Required for image processing utilities
# - torchaudio: Audio processing capabilities
# - transformers: For Pythia-410M (generator) and Pythia-14M (discriminator) models
# - tqdm: Progress tracking for long training sessions
# - numpy: Numerical computations
# - wandb: Experiment tracking and visualization
required_packages = [
    'torch',        # Deep learning framework
    'torchvision',  # Vision utilities
    'torchaudio',   # Audio processing
    'transformers', # For Pythia models
    'tqdm',        # Progress tracking
    'numpy',       # Numerical operations
    'wandb'        # Experiment tracking
]

# Verify all required dependencies
print("Checking essential packages for Branch-GAN implementation:")
for package in required_packages:
    check_package(package)

# Install the 'datasets' package required for CustomPileData preparation
# This package is essential for accessing and processing our custom dataset components:
# - OpenWebText (50%): Web content similar to Pile-CC
# - Wikipedia (25%): Encyclopedic knowledge
# - BookCorpus (25%): Literary content
# Note: While Google Colab provides many pre-installed packages,
# 'datasets' requires explicit installation for our implementation
!pip install datasets

# After installation, verify the package is properly installed
check_package('datasets')


# Configuration Setup for Branch-GAN Experiments
# Implements reduced-scale configurations based on resource constraints identified
# in implementation attempts on Google Colab Pro (A100 GPU) and IBM Watson

# Hardware Configuration Analysis
# Verify available computational resources for experimental setup
# Critical for understanding implementation constraints
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Model Configuration
# Reduced scale configuration based on memory constraints
# Original attempt with 10,000 samples led to memory overflow
datasetConf = PileTraining.datasetConf
generatorConf = PileTraining.generatorConf
discriminatorConf = PileTraining.discriminatorConf
trainingConf = PileTraining.trainingConf

# Dataset Configuration
# Using CustomPileData with reduced sample size due to memory constraints
datasetConf.datasetPath = "CustomPileData"
datasetConf.maxNumSamples = 50  # Reduced from original 10,000 samples

# Training Configuration
# Adapted for resource-constrained environment
trainingConf.device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainingConf.batchSize = 1  # Reduced from original batch size of 8
trainingConf.numEpochs = 5  # Limited epochs for experimental validation
trainingConf.mixedPrecision = True  # Enable mixed precision for memory efficiency

# Model Architecture Configuration
# Reduced model capacity to fit memory constraints
# Original paper used larger configurations
generatorConf.n_layer = 3  # Reduced from original configuration
generatorConf.n_head = 6   # Reduced number of attention heads
discriminatorConf.n_layer = 3
discriminatorConf.n_head = 6

# Memory Management
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Configuration Summary
print(f"Using device: {trainingConf.device}")
print(f"Batch size: {trainingConf.batchSize}")
print(f"Mixed precision: {trainingConf.mixedPrecision}")

# Model Storage Configuration
# Setup Google Drive storage for experiment results and model checkpoints
from google.colab import drive
drive.mount('/content/drive')

import os
from datetime import datetime

# Define storage structure for experimental results
DRIVE_BASE_DIR = "/content/drive/MyDrive/Branch-GAN-Models"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(DRIVE_BASE_DIR, "experiments")
experiment_name = f"Branch-GAN-Pile-experiment_{timestamp}"
experiment_dir = os.path.join(save_dir, experiment_name)

# Create storage directories
os.makedirs(DRIVE_BASE_DIR, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# Ensure clean experiment directory
if os.path.exists(experiment_dir):
    shutil.rmtree(experiment_dir)
os.makedirs(experiment_dir, exist_ok=True)

# Update training configuration with storage parameters
trainingConf.runName = experiment_name
trainingConf.saveModel = True
trainingConf.saveName = experiment_dir

print(f"Models will be saved to: {experiment_dir}")

# Custom Training Metrics and Logging
# Implements detailed logging for experimental analysis of Branch-GAN training
# across different configurations and computational environments

def custom_log_results(validationResult, genData, discData, trainDataset, epochCounter,
                      batchCounter, generatorOptim, discriminatorOptim, trainingConf):
    """
    Enhanced logging function for Branch-GAN training monitoring.
    Provides detailed metrics for both generator and discriminator performance.
    
    Tracks:
    - Validation metrics across different configurations
    - Training metrics for both models
    - Learning rates and generation depth
    - Progress through epochs
    
    Critical for analyzing model behavior in resource-constrained environments.
    """
    def logMetrics(data, prefix, aggregation=np.mean, selectedKeys=None):
        """
        Helper function to log metrics with consistent formatting.
        Aggregates array-based metrics using specified function (default: mean).
        """
        targetKeys = selectedKeys if selectedKeys is not None else data.keys()
        for k in targetKeys:
            if k in data:
                value = data[k]
                if isinstance(value, (np.ndarray, list)):
                    value = aggregation(value)
                print(f"{prefix}-{k}: {value:.4f}")

    # Log training progress
    print(f"\nEpoch: {epochCounter + (batchCounter / len(trainDataset)):.2f}")

    # Log validation metrics
    for name, (valDiscData, valGenData) in validationResult.items():
        print(f"\nValidation Metrics - {name}:")
        print("Discriminator:")
        logMetrics(valDiscData, 'Val')
        print("Generator:")
        logMetrics(valGenData, 'Val')

    # Log training metrics
    print("\nTraining Metrics:")
    print("Discriminator:")
    logMetrics(discData, 'Train')
    print("Generator:")
    logMetrics(genData, 'Train')

    # Log learning rates
    if discriminatorOptim.scheduler is not None:
        lr = discriminatorOptim.scheduler.get_last_lr()[0]
        print(f"Discriminator Learning Rate: {lr:.6f}")
    if generatorOptim.scheduler is not None:
        lr = generatorOptim.scheduler.get_last_lr()[0]
        print(f"Generator Learning Rate: {lr:.6f}")

    # Log generation depth if using depth scheduling
    if trainingConf is not None and trainingConf.depthSchedule is not None:
        print(f"Generation Depth: {trainingConf.depthSchedule.currentDepth}")

    print("-" * 50)  # Visual separator for log readability

# Override default logging with custom implementation
TrainingLoop.logResults = custom_log_results

# Initialize and Execute Training
print("\nInitiating Branch-GAN Training")
print("Configuration Summary:")
print(f"- Dataset: CustomPileData with {datasetConf.maxNumSamples} samples")
print(f"- Model Architecture: {generatorConf.n_layer} layers, {generatorConf.n_head} attention heads")
print(f"- Training: {trainingConf.numEpochs} epochs, batch size {trainingConf.batchSize}")
print(f"- Device: {trainingConf.device}")
print("-" * 50)

# Execute training with configured parameters
SpawnedExperiment.main(datasetConf, generatorConf, discriminatorConf, trainingConf)