# PM-LLMoE: Physical-Mechanism-Guided Mixture-of-Experts with Large Language Models for Industrial Time Series Fault Diagnosis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper:  
**"PM-LLMoE: Physical-Mechanism-Guided Mixture-of-Experts with Large Language Models for Industrial Time Series Fault Diagnosis"**.



## 📖 Introduction

**PM-LLMoE** is a novel deep learning framework designed to address the challenges of **system-level compound fault diagnosis** in complex industrial equipment. 

Existing methods often struggle with two main issues:
1.  **The "Accuracy Trap":** High sample-wise accuracy but low diagnostic consistency (Exact Match Ratio).
2.  **Lack of Interpretability:** Inability to disentangle coupled fault features from different physical components.

To solve these, PM-LLMoE integrates **physical prior knowledge** with the semantic reasoning capabilities of **Large Language Models (LLMs)**. The framework consists of:
* **Physical-Mechanism-Guided MoE:** Decomposes signals into component-specific experts (Motor, Gearbox, Axle Boxes) for feature disentanglement.
* **LLM-based Global Reasoning:** Uses a frozen LLaMA backbone to model cross-component dependencies.
* **Dynamic Focal Loss (DFL):** A specialized loss function to handle extreme class imbalance.

## ✨ Key Features

- **Strict Run-to-Run Data Splitting:** We implement a strict file-level splitting strategy to prevent temporal data leakage and ensure realistic performance evaluation.
- **Dual-Metric Evaluation:** We report both **Sample-wise Accuracy (Acc)** and **Exact Match Ratio (EMR)**, revealing the limitations of current SOTA foundation models in precise diagnosis.
- **Superior Performance:** Achieves **99.96% EMR** on the challenging BJTU-RAO system-level diagnosis task.

## 🛠️ Requirements

The code is tested with Python 3.8+ and PyTorch 2.0+.

```bash
pip install -r requirements.txt