# Branch-GAN Research Implementation

This repository contains implementation experiments of Branch-GAN focused on investigating variable branch depths in resource-constrained environments. The research was conducted at the University of Auckland's School of Computer Science.

## Overview

This is a modified version of the [original Branch-GAN codebase](https://github.com/FreddeFrallan/Branch-GAN), adapted to study implementation challenges and computational requirements in resource-constrained environments. Our implementation focuses on practical limitations and hardware requirements for deploying Branch-GAN's branching mechanism.

## Repository Structure

- `CustomPileData/`: Custom dataset (10,000 samples) combining:
  - OpenWebText (50%): Web content
  - Wikipedia (25%): Encyclopedic knowledge
  - BookCorpus (25%): Literary content

- `Models/`: Core model implementations
- `Training/`: Training configurations and loops
- `ConfigInstances/`: Configuration files

### Analysis Tools
- `CustomPileData.py`: Dataset preparation and tokenization script
- `ExperimentTraining.py`: Experimental training implementations
- `TextGenerationTest.py`: Functions for testing text generation
- `MetricsEvaluation.py`: Performance metrics calculation
- `TokenPredictionTest.py`: Next token prediction analysis
- `CheckModelsWeight.py`: Model weight inspection utilities

## Implementation Details

### Hardware Environments
- Google Colab Pro (A100 GPU, 40GB memory)
- IBM Watson (2x V100 GPU, 32GB memory)

### Model Configuration
- Generator: Based on Pythia-410M architecture
- Discriminator: Based on Pythia-14M for efficiency
- Variable branch depths tested: d = {4, 8, 12, 16}

### Resource Requirements
- Minimum 40GB+ GPU memory
- Training time: 40-50 hours per configuration
- Recommended batch size: 8 for stable training
- Dataset size: 10,000 samples implemented (12,000 designed)

## Implementation Challenges

Our experiments revealed several critical challenges:
1. Memory Management: Exponential memory growth with branching sequences
2. Training Stability: Reduced batch sizes impacting gradient quality
3. Resource Constraints: Early termination due to memory limitations
4. Model Capacity: Trade-offs between model size and performance

## Acknowledgments
Based on the original [Branch-GAN implementation](https://github.com/FreddeFrallan/Branch-GAN) by Carlsson et al.