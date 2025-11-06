# neuron-by-neuron dnn visualizer for mnist - trying to figure out what each neuron is doing
# includes training, hooks for activations, and various visualizations
# IMPORTANT: running with different architectures will show how network size affects learning and neuron specialization

# setting thread limit to 1 to keep tsne stable and reproducible across runs
# without this, tsne can give slightly different results each time due to parallel processing randomness
import os
os.environ["OMP_NUM_THREADS"] = "1"
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import argparse

# preset architectures for quick testing
# "small" is faster, "deep" can learn more complex patterns
ARCHITECTURES = {
    "small":  {"hidden1": 128, "hidden2": 32,  "activation": "relu"},
    "medium": {"hidden1": 256, "hidden2": 64,  "activation": "relu"},
    "tanh":   {"hidden1": 256, "hidden2": 64,  "activation": "tanh"},
    "deep":   {"hidden1": 512, "hidden2": 128, "activation": "relu"},
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--arch",
    type=str,
    default="medium",
    choices=ARCHITECTURES.keys(),
    help="small | medium | tanh | deep"
)
args = parser.parse_args()
config = ARCHITECTURES[args.arch]
print(f"\nusing configuration: {args.arch} -> {config}\n")

# preparing mnist data: convert to tensors and flatten from 28x28 to 784-d vectors
# flattening is necessary because fully connected layers expect 1d input
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# batch_size=64 is a good balance of speed and memory
# shuffle=True for training helps model see varied examples
train_loader = DataLoader(train_data, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False, num_workers=0)

# simple feedforward network with two hidden layers
class mnist_visualizer(nn.Module):
    def __init__(self, cfg):
        super(mnist_visualizer, self).__init__()
        # relu(x) = max(0, x) helps with gradient flow, tanh(x) squashes to [-1, 1]
        act_fn = nn.ReLU if cfg["activation"] == "relu" else nn.Tanh
        
        # fc1: input layer takes 784 pixels and projects to first hidden layer
        # learns basic features like edges and curves
        self.fc1 = nn.Linear(784, cfg["hidden1"])
        
        # fc2: second hidden layer combines features from fc1 into more complex patterns
        # learns higher-level features like loops, lines, corners
        self.fc2 = nn.Linear(cfg["hidden1"], cfg["hidden2"])
        
        # fc3: output layer maps to 10 classes (digits 0-9)
        self.fc3 = nn.Linear(cfg["hidden2"], 10)
        
        self.act = act_fn()

    def forward(self, x):
        # data flows through network layer by layer with activations in between
        # activation functions add non-linearity so the network can learn complex patterns
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)  # final layer outputs raw scores (logits)

model = mnist_visualizer(config)

# adam optimizer adjusts weights using gradient descent with momentum
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# crossentropyloss combines softmax and negative log likelihood
loss_function = nn.CrossEntropyLoss()

train_losses, train_accuracies = [], []

print("\ntraining...")
# training for 5 epochs (one epoch = seeing all training data once)
for epoch in range(5):
    correct, total = 0, 0
    
    for X_batch, y_batch in train_loader:
        preds = model(X_batch)
        loss = loss_function(preds, y_batch)
        
        # backward pass: compute gradients and update weights
        optimizer.zero_grad()  # clear old gradients
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # calculating accuracy
        _, predicted = torch.max(preds, 1)
        correct += (predicted == y_batch).sum().item()
        total   += y_batch.size(0)
    
    acc = correct / total
    train_accuracies.append(acc)
    print(f"epoch {epoch+1}, loss: {loss.item():.4f}, acc: {acc:.4f}")

# plotting training progress
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="loss")
plt.plot(np.linspace(0, len(train_losses), len(train_accuracies)), train_accuracies, label="acc")
plt.legend()
plt.title("training progress")
plt.tight_layout()
plt.show()

# setting up "hooks" to spy on what's happening inside the network
# normally you only see final outputs, hooks let you capture intermediate activations
# crucial for understanding what each neuron is learning
activations = {}

def get_activation(name):
    # hook function saves activations when a layer is computed
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.fc1.register_forward_hook(get_activation("fc1"))
model.fc2.register_forward_hook(get_activation("fc2"))

# finds the top 5 images that most strongly activate each neuron
# helps answer "what does this neuron detect?"
def collect_top_images(layer_name, num_neurons):
    top = {i: [] for i in range(num_neurons)}
    
    model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            _ = model(X_batch)
            acts = activations[layer_name]
            
            for i in range(acts.shape[0]):
                img = X_batch[i].view(28, 28)
                for j in range(num_neurons):
                    top[j].append((acts[i, j].item(), img))
    
    # sorting by activation strength and keeping only top 5 for each neuron
    return {k: sorted(v, key=lambda x: x[0], reverse=True)[:5] for k, v in top.items()}

# computes average activation of each neuron for each digit class (0-9)
# helps identify if a neuron specializes in detecting a specific digit
def compute_avg_activations(model, layer_name, num_neurons):
    per_digit = {i: [[] for _ in range(num_neurons)] for i in range(10)}
    
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            _ = model(X_batch)
            acts = activations[layer_name]
            
            for i in range(len(y_batch)):
                d = y_batch[i].item()
                for j in range(num_neurons):
                    per_digit[d][j].append(acts[i, j].item())
    
    # computing averages for each digit-neuron pair
    avg = np.zeros((10, num_neurons))
    for d in range(10):
        for n in range(num_neurons):
            avg[d, n] = np.mean(per_digit[d][n]) if len(per_digit[d][n]) else 0.0
    
    return avg

def plot_digit_heatmap(avg_acts, layer):
    plt.figure(figsize=(12, 6))
    sns.heatmap(avg_acts, cmap="viridis", xticklabels=False, yticklabels=list(range(10)))
    plt.xlabel("neuron")
    plt.ylabel("digit")
    plt.title(f"avg activation per digit - {layer}")
    plt.tight_layout()
    plt.show()

print("computing avg activations for fc2...")
avg_fc2 = compute_avg_activations(model, "fc2", config["hidden2"])
plot_digit_heatmap(avg_fc2, "fc2")

def show_neuron_images(neuron_id, top_images, layer):
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, (act, img) in enumerate(top_images[neuron_id]):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"a:{act:.2f}")
        axes[i].axis("off")
    plt.suptitle(f"top 5 for neuron {neuron_id} ({layer})")
    plt.tight_layout()
    plt.show()

# visualizes the weights of a specific neuron
# for fc1: weights form a 28x28 pattern showing what image features the neuron looks for
# for fc2: weights connect to fc1 neurons (less spatially interpretable)
def show_weight_image(model, layer, neuron_id):
    if layer == "fc1":
        # fc1 weights can be reshaped to 28x28 to show what image pattern this neuron detects
        w = model.fc1.weight[neuron_id].detach().cpu().numpy().reshape(28, 28)
    elif layer == "fc2":
        # fc2 weights connect to fc1 neurons, not pixels directly
        # reshaping to a grid just for visualization
        w = model.fc2.weight[neuron_id].detach().cpu().numpy()
        side = int(np.ceil(np.sqrt(w.size)))
        pad = side * side - w.size
        w = np.pad(w, (0, pad)).reshape(side, side)
    else:
        raise ValueError("layer must be fc1 or fc2")
    
    plt.imshow(w, cmap="seismic")
    plt.title(f"weights - {layer} neuron {neuron_id}")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

show_weight_image(model, "fc1", 0)

print("\ncollecting activations...")
top_fc1 = collect_top_images("fc1", config["hidden1"])
show_neuron_images(0, top_fc1, "fc1")

top_fc2 = collect_top_images("fc2", config["hidden2"])
show_neuron_images(0, top_fc2, "fc2")

# creates tsne visualizations to see how the network organizes digit representations
# tsne reduces high-dimensional activations to 2d so we can plot them
# if digits cluster together by class, the layer has learned good representations
def plot_tsne_layers(model, layers=["fc1", "fc2"], num_samples=800):
    model.eval()
    with torch.no_grad():
        for layer in layers:
            acts_list, labels_list = [], []
            
            for X_batch, y_batch in test_loader:
                _ = model(X_batch)
                acts_list.append(activations[layer].cpu().numpy().astype(np.float32))
                labels_list.append(y_batch.cpu().numpy())
                if sum(a.shape[0] for a in acts_list) >= num_samples:
                    break
            
            X = np.concatenate(acts_list, axis=0)[:num_samples]
            y = np.concatenate(labels_list, axis=0)[:num_samples]
            
            # trying tsne with fallback to pca if tsne fails
            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=30,
                    init="pca",
                    learning_rate="auto",
                    n_iter=400,
                    random_state=42,
                    method="barnes_hut",
                    verbose=1,
                )
                X_2d = tsne.fit_transform(X)
                title = f"t-sne of {layer} activations"
            except Exception as e:
                from sklearn.decomposition import PCA
                X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
                title = f"pca fallback for {layer} activations (t-sne error: {e})"
            
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", alpha=0.7)
            plt.legend(*sc.legend_elements(), title="digits")
            plt.title(title)
            plt.tight_layout()
            plt.show()

print("\nrunning t-sne for fc1 and fc2...")
plot_tsne_layers(model, ["fc1", "fc2"])

# demonstrates a complete forward pass step-by-step for one image
def forward_pass_demo(model, image, label):
    with torch.no_grad():
        x  = image.view(-1, 784)
        a1 = F.relu(model.fc1(x))
        a2 = F.relu(model.fc2(a1))
        out = model.fc3(a2)
        probs = torch.softmax(out, dim=1)
    
    plt.imshow(image.view(28, 28), cmap="gray")
    plt.title(f"input (label: {label})")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,3))
    plt.bar(range(len(a1[0])), a1[0].numpy())
    plt.title("fc1 activations")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,3))
    plt.bar(range(len(a2[0])), a2[0].numpy())
    plt.title("fc2 activations")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6,3))
    plt.bar(range(10), probs[0].numpy())
    plt.title("output probabilities")
    plt.tight_layout()
    plt.show()

# finds and displays examples where the model made mistakes
def show_misclassified(model, num_samples=5):
    model.eval()
    wrong = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            _, p = torch.max(preds, 1)
            
            for img, t, pr in zip(X_batch, y_batch, p):
                if t != pr:
                    wrong.append((img, t.item(), pr.item()))
                if len(wrong) >= num_samples:
                    break
            if len(wrong) >= num_samples:
                break
    
    fig, axes = plt.subplots(1, len(wrong), figsize=(12, 3))
    if len(wrong) == 1:
        axes = [axes]
    for i, (img, t, pr) in enumerate(wrong):
        axes[i].imshow(img.view(28, 28), cmap="gray")
        axes[i].set_title(f"t:{t}, p:{pr}")
        axes[i].axis("off")
    plt.suptitle("misclassified samples")
    plt.tight_layout()
    plt.show()

# tests how robust the model is to noise
def noisy_prediction(model, image, noise_level=0.3):
    noisy = torch.clamp(image + noise_level * torch.randn_like(image), 0, 1)
    
    with torch.no_grad():
        probs = torch.softmax(model(noisy.view(-1, 784)), dim=1)
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(image.view(28, 28), cmap="gray")
    ax[0].set_title("original")
    ax[1].imshow(noisy.view(28, 28), cmap="gray")
    ax[1].set_title("noisy")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6,3))
    plt.bar(range(10), probs[0].numpy())
    plt.title("prediction on noisy image")
    plt.tight_layout()
    plt.show()

# demos
img, label = test_data[0]
forward_pass_demo(model, img, label)
show_misclassified(model, num_samples=5)
noisy_prediction(model, img, noise_level=0.5)