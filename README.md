# MNIST Neural Network Visualizer

A neural network interpretability toolkit for understanding how models learn to recognize handwritten digits from the MNIST dataset.

## Project Context

Developed as part of exploring neural network interpretability techniques. The goal is to make deep learning more transparent and understandable by visualizing what happens at each layer of the network.

## Project Overview

This project consists of two components:

### 1. Core Analysis Script (`neural_visualizer.py`)
Command-line tool for training and analyzing neural networks with different architectures.
- Run experiments with configurable network sizes via `--arch` flag
- Generates comprehensive visualizations automatically
- Designed for reproducible research and batch analysis
- Usage: `python neural_visualizer.py --arch deep`

### 2. Interactive Web App (`app.py`)
Streamlit dashboard that provides a user-friendly interface to the visualization techniques developed in the core script.
- Train models with real-time progress tracking
- Interactively explore individual neurons
- Adjust architecture and hyperparameters on the fly
- No coding required - accessible to anyone

## What It Does

**Training & Architecture:**
- Trains fully connected neural networks with customizable hidden layer sizes
- Supports multiple activation functions (ReLU, Tanh)
- Preset architectures: Small (128→32), Medium (256→64), Deep (512→128)

**Visualization Features:**
- **Weight Maps:** See what patterns each neuron responds to (for fc1, direct pixel weights; for fc2, projected back to input space)
- **Neuron Digit Profiles:** Bar charts showing which digits most strongly activate each neuron
- **Top Activating Images:** The 5 images that trigger each neuron the most
- **t-SNE/PCA Embeddings:** 2D visualization of how the network separates digits in activation space
- **Confusion Matrix:** See which digits the model confuses with each other
- **Noise Robustness Testing:** Check how model predictions change with added noise

## How It Works

**Technical Approach:**
- Built with PyTorch for neural network training
- Uses forward hooks to capture intermediate layer activations
- Applies t-SNE (with PCA fallback) for dimensionality reduction
- Streamlit for the interactive web interface

**Key Technique - Hooks:**
Hooks let us "spy" on what's happening inside the network during forward passes. Normally you only see final outputs, but hooks capture intermediate activations, which is crucial for understanding what individual neurons learn.

**Why This Matters:**
Neural networks are often treated as black boxes. This project opens them up to show:
- What features different layers learn
- How neurons specialize for specific digits
- Where the model struggles (confusion patterns)

## Current Status

**Working:**
- Core training and visualization pipeline
- Interactive Streamlit interface
- Multiple architecture presets
- Comprehensive neuron analysis tools

**In Progress:**
- CNN architectures for comparison
- Model save/load functionality
- Performance optimizations for larger networks
- Additional visualization techniques (grad-CAM, saliency maps)

## Running It

### Interactive App (Recommended)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Command-Line Script
```bash
pip install -r requirements.txt
python neural_visualizer.py --arch medium  # options: small, medium, tanh, deep
```

## Requirements
```
streamlit
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
```
