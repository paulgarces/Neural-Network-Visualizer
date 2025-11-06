# Paul Garces
# streamlit app for interactive mnist neural network visualization
# lets users train different architectures and explore what individual neurons learn

# setting thread limit to keep tsne deterministic on macos
# without this, parallel processing in tsne can cause different results each run
import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random

st.set_page_config(page_title="MNIST Neural Network Visualizer", layout="wide")
st.title("MNIST Neural Network Visualizer")
st.write("see how different neural nets learn to recognize digits layer by layer")

# preset architectures with different sizes and activations
# smaller networks train faster but may not learn as well
# tanh is an older activation function, relu is more common now
ARCHITECTURES = {
    "Small (128→32, ReLU)":  {"hidden1": 128, "hidden2": 32,  "activation": "relu"},
    "Medium (256→64, ReLU)": {"hidden1": 256, "hidden2": 64,  "activation": "relu"},
    "Tanh (256→64, Tanh)":   {"hidden1": 256, "hidden2": 64,  "activation": "tanh"},
    "Deep (512→128, ReLU)":  {"hidden1": 512, "hidden2": 128, "activation": "relu"},
}

choice = st.selectbox("choose network architecture:", list(ARCHITECTURES.keys()))
config = ARCHITECTURES[choice]

# keeping epochs low for interactive use - don't want users waiting too long
epochs = st.slider("training epochs", 1, 6, 3, 1)
# larger batch sizes train faster but use more memory
batch_size = st.select_slider("batch size", options=[64, 128, 256, 512], value=256)
st.caption(f"using configuration: {config}")

# caching data loading so we don't re-download mnist every time user changes settings
@st.cache_resource(show_spinner=False)
def load_data():
    transform = transforms.ToTensor()  # converts images to tensors with values 0-1
    train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train, test

train_data, test_data = load_data()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # shuffle helps training
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)  # no need to shuffle test data

# simple 3-layer feedforward network
class SimpleNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        act = nn.ReLU if cfg["activation"] == "relu" else nn.Tanh
        # fc1 takes flattened 28x28 image (784 pixels) to first hidden layer
        self.fc1 = nn.Linear(784, cfg["hidden1"])
        # fc2 takes first hidden layer to second hidden layer
        self.fc2 = nn.Linear(cfg["hidden1"], cfg["hidden2"])
        # fc3 outputs logits for 10 digit classes
        self.fc3 = nn.Linear(cfg["hidden2"], 10)
        self.act = act()

    def forward(self, x):
        if x.dim() > 2: x = x.view(x.size(0), -1)  # flatten if image is 2d
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)  # raw scores, no softmax (handled by loss function)

def seed_all(seed=42):
    # seeding for reproducibility - same seed gives same results
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# streamlit session state persists data across reruns
# this is how we keep the trained model and training history
if "model" not in st.session_state: st.session_state.model = None
if "train_log" not in st.session_state: st.session_state.train_log = []
if "acts" not in st.session_state: st.session_state.acts = {}

# hooks let us capture intermediate layer activations during forward pass
# normally you only see final output, hooks give us access to hidden layers
def register_hooks(model, acts_dict):
    def grab(name):
        def hook(_m, _in, out): acts_dict[name] = out.detach().cpu()
        return hook
    h1 = model.fc1.register_forward_hook(grab("fc1"))
    h2 = model.fc2.register_forward_hook(grab("fc2"))
    return [h1, h2]  # return handles so we can remove them later

# computes average activation of a specific neuron for each digit (0-9)
# helps identify if a neuron specializes in detecting certain digits
def neuron_digit_profile(model, layer_name, neuron_id, loader, max_batches=60):
    model.eval()
    acts = {}
    handle = getattr(model, layer_name).register_forward_hook(lambda m,i,o: acts.setdefault("a", o.detach().cpu()))
    buckets = [[] for _ in range(10)]  # one list per digit class
    seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            _ = model(xb)
            vals = acts["a"][:, neuron_id].numpy()
            for v, y in zip(vals, yb.numpy()): buckets[int(y)].append(float(v))
            seen += 1
            if seen >= max_batches: break
    handle.remove()
    return np.array([np.mean(b) if b else 0.0 for b in buckets])

# finds the k images that most strongly activate a specific neuron
# shows what patterns/features the neuron has learned to detect
def top_activating_images(model, layer_name, neuron_id, loader, k=5, max_batches=80):
    model.eval()
    acts = {}
    handle = getattr(model, layer_name).register_forward_hook(lambda m,i,o: acts.setdefault("a", o.detach().cpu()))
    best = []
    with torch.no_grad():
        for xb, _ in loader:
            _ = model(xb)
            vals = acts["a"][:, neuron_id].numpy()
            for v, img in zip(vals, xb): best.append((float(v), img.view(28,28).numpy()))
            # keeping a sliding window of top-k to avoid memory issues
            if len(best) > 3000: best = sorted(best, key=lambda t: t[0], reverse=True)[:k]
    handle.remove()
    best = sorted(best, key=lambda t: t[0], reverse=True)[:k]
    return best

# tries tsne for visualization, falls back to pca if it fails
# tsne is better at showing clusters but can be unstable
def tsne_or_pca(X, y):
    try:
        X2 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                  n_iter=400, random_state=42, method="barnes_hut", verbose=0).fit_transform(X)
        title = "t-SNE"
    except Exception:
        X2 = PCA(n_components=2, random_state=42).fit_transform(X)  # fallback if tsne errors
        title = "PCA (t-SNE fallback)"
    return X2, title

def summarize_confusions(cm):
    # extracts the most common misclassifications from confusion matrix
    off = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0: off.append((i, j, int(cm[i, j])))
    return sorted(off, key=lambda x: x[2], reverse=True)

col_btn, _ = st.columns([1, 3])
with col_btn:
    if st.button("train / retrain model", use_container_width=True):
        seed_all(42)  # reproducible training
        model = SimpleNN(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        hooks = register_hooks(model, st.session_state.acts)
        st.session_state.train_log = []
        prog = st.progress(0, text="training...")

        for ep in range(epochs):
            correct, total, last_loss = 0, 0, 0.0
            for xb, yb in train_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                # standard gradient descent: zero grads, backward pass, optimizer step
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                last_loss = loss.item()
                correct += (pred.argmax(1) == yb).sum().item()
                total += yb.size(0)
            acc = correct / total
            st.session_state.train_log.append((ep+1, last_loss, acc))
            prog.progress((ep+1)/epochs, text=f"epoch {ep+1}/{epochs} • loss={last_loss:.4f} • acc={acc:.4f}")

        for h in hooks: h.remove()
        st.session_state.model = model
        st.success("training complete")

model = st.session_state.model
if model is not None:
    with st.expander("training progress", expanded=True):
        e, l, a = zip(*st.session_state.train_log)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(l, label="loss")
        ax2 = ax.twinx(); ax2.plot(a, label="acc", color="tab:orange")
        ax.set_title("training loss (line) and accuracy (orange)")
        ax.set_xlabel("epoch"); ax.set_xticks(range(len(e))); ax.set_xticklabels(e)
        st.pyplot(fig)
        st.caption("as loss drops and accuracy rises, the model is fitting the digits better")

    st.sidebar.header("neuron explorer")
    layer_opt = st.sidebar.selectbox("layer", ["fc1", "fc2"])
    hidden_size = config["hidden1"] if layer_opt == "fc1" else config["hidden2"]
    neuron_id = st.sidebar.slider("neuron id", 0, hidden_size-1, 0, 1)

    st.subheader("weight map")
    # for fc1, weights directly connect to pixels so we can visualize them as an image
    # for fc2, we project back through fc1 to see effective pixel-space pattern
    if layer_opt == "fc1":
        w = model.fc1.weight[neuron_id].detach().cpu().numpy()
        Wimg = w.reshape(28,28)
    else:
        # fc2 neuron weights times fc1 weights gives effective input-space weights
        w2 = model.fc2.weight[neuron_id].detach().cpu().numpy()
        W1 = model.fc1.weight.detach().cpu().numpy()
        w_eff = w2 @ W1  # matrix multiplication projects to input space
        Wimg = w_eff.reshape(28,28)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(Wimg, cmap="seismic", ax=ax, cbar=True)
    ax.set_title(f"weights of {layer_opt} neuron {neuron_id}")
    st.pyplot(fig)
    st.caption("red pixels boost activation; blue reduce it. this shows the pattern the neuron responds to.")

    st.subheader("neuron digit profile")
    profile = neuron_digit_profile(model, layer_opt, neuron_id, test_loader)
    st.bar_chart(profile)
    top3 = np.argsort(profile)[::-1][:3].tolist()
    st.caption(f"higher bars mean stronger response. neuron {neuron_id} fires most for digits {top3}")

    st.subheader("top activating images")
    best = top_activating_images(model, layer_opt, neuron_id, test_loader, k=5)
    fig, axs = plt.subplots(1, len(best), figsize=(10,2))
    for i,(act,img) in enumerate(best):
        axs[i].imshow(img, cmap="gray"); axs[i].axis("off"); axs[i].set_title(f"a:{act:.2f}")
    st.pyplot(fig)
    st.caption("these are the examples that push this neuron the hardest")

    st.subheader("layer embedding (t-SNE / PCA)")
    # collecting activations from test set to visualize in 2d
    acts_list, labels = [], []
    with torch.no_grad():
        handles = register_hooks(model, st.session_state.acts)
        cnt = 0
        for xb, yb in test_loader:
            _ = model(xb)
            A = st.session_state.acts[layer_opt].numpy().astype(np.float32)  # get current batch activations
            acts_list.append(A); labels.append(yb.numpy())
            cnt += len(yb)
            if cnt >= 800: break  # limit for interactive performance
        for h in handles: h.remove()
    X = np.concatenate(acts_list, axis=0)[:800]; y = np.concatenate(labels, axis=0)[:800]
    X2, title = tsne_or_pca(X, y)
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(X2[:,0], X2[:,1], c=y, cmap="tab10", s=12, alpha=0.8)
    legend = ax.legend(*sc.legend_elements(num=10), title="digits", bbox_to_anchor=(1.02,1), loc="upper left")
    ax.add_artist(legend); ax.set_title(f"{title} of {layer_opt} activations")
    st.pyplot(fig)
    st.caption("each dot is an image; tighter clusters mean the layer separates those digits better")

    st.subheader("confusion matrix")
    # shows which digits the model confuses with each other
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(yb.cpu().numpy().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"accuracy = {acc:.3f}")
    st.pyplot(fig)
    confs = summarize_confusions(cm)
    st.caption(f"most common mix-ups: {confs[:3]}")

    st.subheader("noise robustness test")
    # tests how well model handles corrupted inputs
    idx = st.slider("test image index", 0, len(test_data)-1, 0)
    img, label = test_data[idx]
    noise = st.slider("noise level", 0.0, 0.8, 0.4, 0.05)
    noisy = torch.clamp(img + noise * torch.randn_like(img), 0, 1)  # clip to valid range
    with torch.no_grad():
        probs = F.softmax(model(noisy), dim=1)[0].numpy()
    c1, c2 = st.columns(2)
    with c1:
        st.image(img.view(28,28).numpy(), caption=f"original (label {label})", use_column_width=True)
        st.image(noisy.view(28,28).numpy(), caption="noisy version", use_column_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(range(10), probs); ax.set_title("predictions on noisy image"); ax.set_xticks(range(10))
        st.pyplot(fig)
        st.caption("if one bar still dominates, the model is robust at this noise level")
else:
    st.info("choose an architecture and click train to start")