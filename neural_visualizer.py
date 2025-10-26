# neuron-by-neuron dnn visualizer for mnist - trying to figure out what each neuron is doing
# includes training, hooks for activations, and various visualizations

import os
os.environ["OMP_NUM_THREADS"] = "1" # limit threads for sklearn t-sne
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

# data 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# model definition
class mnist_visualizer(nn.Module):
    def __init__(self):
        super(mnist_visualizer, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = mnist_visualizer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

# training loop
train_losses, train_accuracies = [], []
print("\ntraining...")
for epoch in range(5):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        preds = model(X_batch)
        loss = loss_function(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        _, predicted = torch.max(preds, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    acc = correct / total
    train_accuracies.append(acc)
    print(f"epoch {epoch+1}, loss: {loss.item():.4f}, acc: {acc:.4f}")

plt.plot(train_losses, label="loss")
plt.plot(np.linspace(0, len(train_losses), len(train_accuracies)), train_accuracies, label="acc")
plt.legend()
plt.title("training progress")
plt.show()

# hooks to capture activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.fc1.register_forward_hook(get_activation("fc1"))
model.fc2.register_forward_hook(get_activation("fc2"))

# analysis functions 
def collect_top_images(layer_name, num_neurons):
    neuron_top_images = {i: [] for i in range(num_neurons)}
    model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            _ = model(X_batch)
            acts = activations[layer_name]
            for i in range(acts.shape[0]):
                img = X_batch[i].view(28, 28)
                for j in range(num_neurons):
                    neuron_top_images[j].append((acts[i, j].item(), img))
    return {k: sorted(v, key=lambda x: x[0], reverse=True)[:5] for k, v in neuron_top_images.items()}

def compute_avg_activations(model, layer_name, num_neurons):
    digit_acts = {i: [[] for _ in range(num_neurons)] for i in range(10)}
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            _ = model(X_batch)
            acts = activations[layer_name]
            for i in range(len(y_batch)):
                lbl = y_batch[i].item()
                for j in range(num_neurons):
                    digit_acts[lbl][j].append(acts[i, j].item())

    avg = np.zeros((10, num_neurons))
    for d in range(10):
        for n in range(num_neurons):
            avg[d, n] = np.mean(digit_acts[d][n])
    return avg

def plot_digit_heatmap(avg_acts, layer):
    plt.figure(figsize=(12, 6))
    sns.heatmap(avg_acts, cmap="viridis", xticklabels=False, yticklabels=list(range(10)))
    plt.xlabel("neuron")
    plt.ylabel("digit")
    plt.title(f"avg activation per digit - {layer}")
    plt.show()

print("computing avg activations for fc2...")
avg_fc2 = compute_avg_activations(model, "fc2", 64)
plot_digit_heatmap(avg_fc2, "fc2")

def show_neuron_images(neuron_id, top_images, layer):
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, (act, img) in enumerate(top_images[neuron_id]):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"a:{act:.2f}")
        axes[i].axis('off')
    plt.suptitle(f"top 5 for neuron {neuron_id} ({layer})")
    plt.tight_layout()
    plt.show()

def show_weight_image(model, layer, neuron_id):
    if layer == "fc1":
        w = model.fc1.weight[neuron_id].detach().cpu().numpy().reshape(28, 28)
    elif layer == "fc2":
        w = model.fc2.weight[neuron_id].detach().cpu().numpy().reshape(16, 4)
    else:
        raise ValueError("layer must be fc1 or fc2")
    plt.imshow(w, cmap='seismic')
    plt.title(f"weights - {layer} neuron {neuron_id}")
    plt.colorbar()
    plt.show()

# visualizations 
show_weight_image(model, "fc1", 0)
print("\ncollecting activations...")
top_fc1 = collect_top_images("fc1", 256)
show_neuron_images(0, top_fc1, "fc1")

top_fc2 = collect_top_images("fc2", 64)
show_neuron_images(0, top_fc2, "fc2")

def plot_tsne(model, layer, num_samples=1000):
    model.eval()
    acts_list, labels_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            _ = model(X_batch)
            acts_list.append(activations[layer].cpu().numpy().astype(np.float32))  # <-- numpy
            labels_list.append(y_batch.cpu().numpy())
            if sum(a.shape[0] for a in acts_list) >= num_samples:
                break

    X = np.concatenate(acts_list, axis=0)[:num_samples]
    y = np.concatenate(labels_list, axis=0)[:num_samples]

    # try tsne; if it crashes/errors, fall back to pca
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            n_iter=500,           # a bit smaller = stabler/faster
            random_state=42,
            method="barnes_hut",  # default; keeps memory lower
            verbose=1,
        )
        X_2d = tsne.fit_transform(X)
        title = f"t-sne of {layer} activations"
    except Exception as e:
        # fallback: pca (never crashes)
        from sklearn.decomposition import PCA
        X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
        title = f"pca fallback for {layer} activations (t-sne error: {e})"

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", alpha=0.7)
    plt.legend(*sc.legend_elements(), title="digits")
    plt.title(title)
    plt.show()

print("\nrunning t-sne for fc2...")
plot_tsne(model, "fc2")

# educational extras, going to try and add more later
def forward_pass_demo(model, image, label):
    with torch.no_grad():
        x = image.view(-1, 784)
        a1 = F.relu(model.fc1(x))
        a2 = F.relu(model.fc2(a1))
        out = model.fc3(a2)
        probs = torch.softmax(out, dim=1)

    plt.imshow(image.view(28,28), cmap="gray")
    plt.title(f"input (label: {label})")
    plt.show()

    plt.bar(range(len(a1[0])), a1[0].numpy())
    plt.title("fc1 activations")
    plt.show()

    plt.bar(range(len(a2[0])), a2[0].numpy())
    plt.title("fc2 activations")
    plt.show()

    plt.bar(range(10), probs[0].numpy())
    plt.title("output probabilities")
    plt.show()

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
                if len(wrong) >= num_samples: break
            if len(wrong) >= num_samples: break

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i, (img, t, pr) in enumerate(wrong):
        axes[i].imshow(img.view(28, 28), cmap="gray")
        axes[i].set_title(f"t:{t}, p:{pr}")
        axes[i].axis("off")
    plt.suptitle("misclassified samples")
    plt.show()

def noisy_prediction(model, image, noise_level=0.3):
    noisy = torch.clamp(image + noise_level * torch.randn_like(image), 0, 1)
    with torch.no_grad():
        probs = torch.softmax(model(noisy.view(-1, 784)), dim=1)

    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    ax[0].imshow(image.view(28,28), cmap="gray")
    ax[0].set_title("original")
    ax[1].imshow(noisy.view(28,28), cmap="gray")
    ax[1].set_title("noisy")
    plt.show()

    plt.bar(range(10), probs[0].numpy())
    plt.title("prediction on noisy image")
    plt.show()

# demo calls
img, label = test_data[0]
forward_pass_demo(model, img, label)
show_misclassified(model, num_samples=5)
noisy_prediction(model, img, noise_level=0.5)