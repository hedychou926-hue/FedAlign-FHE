# FedAlign-FHE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper:
**"FedAlign-FHE: Alignment-Enhanced Homomorphic Encrypted Federated Learning for Heterogeneous Smart Healthcare"**

## 📖 Abstract
Federated learning (FL) enables collaborative healthcare model training without sharing raw patient data. However, practical deployments still face data heterogeneity, encryption overhead, and unreliable aggregation under noisy medical data. **FedAlign-FHE** is a privacy-preserving FL framework that introduces two main stages:
- **Stage 1 (Pre-aggregation Alignment):** Uses local teachers, a shared student backbone, and personalized heads to reduce output-level client divergence via knowledge distillation.
- **Stage 2 (Encrypted Shared-Backbone Training):** Encrypts only shared-backbone updates using CKKS Fully Homomorphic Encryption (FHE) while keeping personalized heads local, followed by a quality-aware aggregation (**FHE-QW**) strategy based on sample size and local validation quality.

## 🗂️ Repository Structure

The repository is modularized by dataset. Each dataset folder contains its own complete pipeline (data processing, models, and training stages) for independent reproducibility.

```text
FedAlign-FHE/
├── CDC_Diabetes/           # Pipeline for tabular health indicators
│   ├── data/
│   ├── data_process.py     
│   ├── models.py           
│   ├── stage1_kd.py        
│   ├── stage2_fhe.py       
│   └── main.py             
├── Pneumonia/              # Pipeline for medical images (PneumoniaMNIST)
│   ├── data/
│   ├── data_process.py     
│   ├── models.py           
│   ├── stage1_kd.py        
│   ├── stage2_fhe.py       
│   └── main.py             
├── PTB_XL/                 # Pipeline for 1D ECG signals
│   ├── data/               # Raw and processed data
│   ├── data_process.py     # Dirichlet non-IID partitioning
│   ├── models.py           # Shared Backbone & Personalized Heads for ECG
│   ├── stage1_kd.py        # Knowledge distillation alignment
│   ├── stage2_fhe.py       # FHE-QW encrypted training
│   └── main.py             # Execution script
├── utils/                  # Shared utility functions 
│   ├── crypto_utils.py     # CKKS encryption wrappers
│   └── metrics.py          # EMD, Hellinger Distance, Balanced Accuracy
└── requirements.txt        # Python dependencies

