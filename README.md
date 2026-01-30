# RoWSFormer-blind-watermarking

This repository contains a **PyTorch-based reproduction** of the research paper:

> **RoWSFormer: A Robust Watermarking Framework with Swin Transformer for Enhanced Geometric Attack Resilience**

The objective of this project is to build a **faithful, lightweight baseline implementation** of a **deep blind image watermarking system** using transformer-based architectures.

---

## 🔍 Project Motivation

Blind image watermarking aims to embed a binary message into an image such that:

- The perceptual quality of the image is preserved
- The watermark can be extracted **without access to the original image**
- The watermark is **robust to common image distortions and geometric attacks**

Recent CNN-based methods show limitations in modeling long-range dependencies and structured geometric variations.
  
RoWSFormer addresses this by combining:

- **Row-wise self-attention** for structured spatial modeling
- **Window-based (Swin) attention** for local feature consistency
- **Frequency-aware global modeling** for robustness

This repository focuses on **reproducing these core ideas** in a clean and extensible codebase.

---

## 🏗️ Architecture Overview

### Core Components

- **Encoder**
  - Patch embedding
  - U-Net–style downsampling / upsampling
  - Locally-Channel Enhanced Swin Transformer Blocks (LCESTB)
  - Frequency-Enhanced Transformer Block (FETB)
  - Residual watermark embedding

- **Attack Layer**
  - Differentiable robustness simulation:
    - Gaussian noise
    - Blur
    - JPEG-like compression
    - Crop + resize
    - Small rotations

- **Decoder**
  - Transformer-based backbone
  - Blind watermark extraction (no access to original image)

---

## 🚀 Getting Started

### Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

### 📁 Dataset

The model is trained on natural image datasets.
No watermark-specific dataset is required.

Supported datasets:
- DIV2K
- COCO (subset)
- ImageNet (resized)

Images are resized to 128×128.

---

## 📊 Evaluation

Evaluate watermark robustness and image quality:

- **PSNR** – visual quality of watermarked image
- **Bit Accuracy** – correctness of extracted watermark bits

---

## 📚 Reference

If you use or build upon this work, please cite the original paper:

```bibtex
@misc{chen2024rowsformer,
  title         = {RoWSFormer: A Robust Watermarking Framework with Swin Transformer for Enhanced Geometric Attack Resilience},
  author        = {Weitong Chen and Yuheng Li},
  year          = {2024},
  eprint        = {2409.14829},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MM},
  url           = {https://arxiv.org/abs/https://arxiv.org/abs/2409.14829}
}
```