# CrackMDM

**CrackMDM: Masked Modeling on DCT-domain for Efficient Pavement Crack Segmentation**

This is the official implementation of **CrackMDM**, a frequency-aware crack segmentation framework designed for remote sensing and pavement inspection tasks.

It consists of:

- **Pretraining Phase**: Masked denoising diffusion process with frequency decomposition.
- **Fine-tuning Phase**: Supervised segmentation using the pretrained encoder as a frozen feature extractor.

## Pretraining

We follow the pretraining strategy proposed in Masked Frequency Modeling ([MFM](https://github.com/Jiahao000/MFM)). 

Please refer to [Pretrain README](./pretrain/README.md) for:

Environment setup

Training instructions

## Fine-tuning

The fine-tuning phase uses the pretrained encoder and attaches a segmentation head. The model is trained on crack segmentation datasets in a supervised manner.

Please refer to [Finetuning README](./finetuning/README.md) for:

Dataset preparation

Pretrained model loading strategy

