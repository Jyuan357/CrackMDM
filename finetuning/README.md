# Fine-tuning Instructions for CrackMDM

This document provides the setup and training instructions for the **Fine-tuning Phase** of **CrackMDM**.

## Dataset Preparation

For fine-tuning, you need a labeled crack segmentation dataset. The dataset should include both crack images and their corresponding segmentation masks.

### Dataset Structure
Ensure that your dataset is organized in the following structure:

```bash
datasets/
    train/
        images/
            image1.png
            image2.png
            ...
        masks/
            mask1.png
            mask2.png
            ...
        train.txt
    test/
        ...
```

## Pretrained Model Loading Strategy

To load the pretrained model, run the following command:

```bash
python finetune.py
```

Make sure that the path to the pretrained model is correctly specified.