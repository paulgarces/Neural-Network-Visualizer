# MNIST Neural Network Visualizer

This project is an interactive Streamlit app that shows how a simple neural network learns to recognize handwritten digits from the MNIST dataset.  
It’s still a work in progress, but the main features are up and running.

## What It Does
- Trains a basic two-layer neural network on MNIST.  
- Lets you explore:
  - Weight maps (what each neuron is "looking for").  
  - Neuron activation profiles (which digits each neuron fires for).  
  - 2D visualizations using t-SNE or PCA.  
  - A confusion matrix of predictions.  
  - How the model handles noisy images.

## How It Works
- Built with PyTorch and Streamlit.  
- Hooks into layers to grab neuron activations for visualization.  
- Uses t-SNE and PCA to show how the model separates digits in lower dimensions.  
- Runs fully on your local machine.

## Current Status
The core training and visualization parts are working.  
Still planning to add a few things like:
- More network options (maybe a CNN).  
- Model saving and loading.  
- Cleaner layout and performance tweaks.

## Running It
```bash
pip install -r requirements.txt
streamlit run app.py

## Requirements
- streamlist
- torch
- torchvision
- streamlit
- numpy
- numpy
- matplotlib
- seaborn
- scikit-learn
