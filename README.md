# Next-Generation Multi-Modal Image Retrieval System 🖼️🔍

This repository contains a comprehensive Content-Based Image Retrieval (CBIR) system designed to bridge the "Semantic Gap" between low-level pixel data and high-level human concepts. The system evaluates classical handcrafted descriptors, deep neural manifolds, and a proposed novel hybrid method across three datasets of increasing complexity: MNIST, FashionMNIST, and CIFAR-10.

## 🚀 Features & Paradigms Explored

1. **Classical Handcrafted Features:** Local Binary Patterns (LBP) for spatial texture analysis.
2. **Untrained Neural Manifolds:** Shallow (NN) and Deep (DNN) neural networks to analyze raw flattened pixel representations.
3. **Deep Convolutional Networks (CNN):** A truncated ResNet-18 architecture utilizing ImageNet normalization for deep semantic feature extraction.
4. **Proposed Novelty (SFGD):** A custom **Spatial Frequency Gradient Descriptor** that maps spatial edge boundaries using Sobel gradients and applies a 2D-DCT to capture noise-resistant structural fingerprints.
5. **Color Feature Extraction:** Global 9-D vectors capturing Intra-Color statistics (Mean, Standard Deviation) and Inter-Color channel correlations (Pearson Correlation).
6. **Hybrid Feature Fusion:** An $L_2$-normalized fusion pipeline combining structural priors (SFGD/Color) with deep semantic contexts (CNN) to combat background bias and geometric aliasing.

## 📁 Repository Structure

* `build_db.py`: The offline indexing script. Downloads the datasets, extracts features using all paradigms, and serializes them into `.pkl` databases.
* `features.py`: Contains the mathematical logic for classical and proposed feature extractors (LBP, SFGD, Color Features, and Fusion).
* `models.py`: Contains the PyTorch network architectures (SimpleNN, DeepNN, and CNNFeatureExtractor).
* `evaluate.py`: The evaluation script. Calculates Precision@K, Recall@K, and Mean Average Precision (mAP) for all methods across the generated databases.
* `app.py`: A fully interactive Streamlit web application featuring a UI for uploading query images, selecting feature methods, computing Cosine Similarity, and displaying results.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/aarushdua2004/dlv_assignment.git)
   cd dlcv_assignment
