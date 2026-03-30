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
Install the required dependencies:
```bash
pip install -r requirements.txt
```
💻 How to Use the System
Step 1: Build the Feature Databases
Before querying or evaluating, you must build the offline feature databases. This script will download the datasets (if not present) and generate MNIST_features.pkl, FashionMNIST_features.pkl, and CIFAR10_features.pkl.
```bash
python build_db.py
```
Step 2: Launch the Web App
To interactively test the retrieval system, upload your own images, and visualize the feature fusion in action, run the Streamlit app:
```bash
streamlit run app.py
```
Step 3: Run the Evaluation Suite
To rigorously evaluate the mathematical performance of all methods across all datasets, run the evaluation script:
```bash
python evaluate.py
```
This will output a comparative table of Precision, Recall, and mAP for LBP, NN, DNN, CNN, SFGD, Color, and Hybrid models.

📊 Core Findings
Spatial constraints: Classical spatial methods (LBP) excel on zero-background datasets (MNIST) but suffer from "color-blindness" and background dominance on CIFAR-10.

Resolution collapse: Handling 32x32 CIFAR-10 images in pre-trained CNNs requires explicit ImageNet normalization to prevent spatial pooling destruction.

Hybrid dominance: Fusing SFGD or Color features with CNN embeddings stabilizes the retrieval manifold, successfully separating confusing classes (e.g., distinguishing Airplanes from Trucks by validating edge structures alongside context).
