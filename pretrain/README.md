# Pretraining Instructions for CrackMDM

This document provides the setup and training instructions for the **Pretraining Phase** of **CrackMDM**. The pretraining phase uses a masked denoising diffusion process with frequency decomposition to train the model.

## Environment Setup

Please refer to the [MFM Installation Guide](https://github.com/Jiahao000/MFM/blob/master/docs/INSTALL.md) for the environment setup and dependencies installation.

## Training Instructions

### Step 1: Prepare Your Dataset
For pretraining, you only need **unlabeled crack images**, which should be cropped into equal-sized patches.

### Step 2: Configuration

Modify the configuration file config.yaml to specify parameters such as learning rate, batch size, and dataset paths.

### Step 3: Start Pretraining
Run the following command to start pretraining:
python pretrain.py --config config.yaml

```bash
python pretrain.py --config config.yaml
```