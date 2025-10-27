import os
os.environ["OMP_NUM_THREADS"] = "1"  # keep tsne stable on macos

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

st.set_page_config(page_title="MNIST Neural Network Visualizer", layout="wide")  # basic page setup
st.title("MNIST Neural Network Visualizer")  # title only
st.write("see how different neural nets learn to recognize digits layer by layer")  # quick context

ARCHITECTURES = {  # presets kept simple
    "Small (128→32, ReLU)":  {"hidden1": 128, "hidden2": 32,  "activation": "relu"},
    "Medium (256→64, ReLU)": {"hidden1": 256, "hidden2": 64,  "activation": "relu"},
    "Tanh (256→64, Tanh)":   {"hidden1": 256, "hidden2": 64,  "activation": "tanh"},
    "Deep (512→128, ReLU)":  {"hidden1": 512, "hidden2": 128, "activation": "relu"},
}
choice = st.selectbox("choose network architecture:", list(ARCHITECTURES.keys()))  # user picks config
config = ARCHITECTURES[choice]  # grab dict

epochs = st.slider("training epochs", 1, 6, 3, 1)  # small range = quick runs
batch_size = st.select_slider("batch size", options=[64, 128, 256, 512], value=256)  # common sizes
st.caption(f"using configuration: {config}")  # show current setup

@st.cache_resource(show_spinner=False)
def load_data():
    transform = transforms.ToTensor()  # just tensor; we flatten inside forward
    train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)  # train set
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)  # test set
    return train, test

train_data, test_data = load_data()  # load once (cached)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # shuffle for training
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)  # bigger batch ok for eval

class SimpleNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        act = nn.ReLU if cfg["activation"] == "relu" else nn.Tanh  # pick activation
        self.fc1 = nn.Linear(784, cfg["hidden1"])  # 28*28 -> hidden1
        self.fc2 = nn.Linear(cfg["hidden1"], cfg["hidden2"])  # hidden1 -> hidden2
        self.fc3 = nn.Linear(cfg["hidden2"], 10)  # hidden2 -> logits
        self.act = act()  # instantiate once

    def forward(self, x):
        if x.dim() > 2: x = x.view(x.size(0), -1)  # flatten nhwc to (n, 784)
        x = self.act(self.fc1(x))  # layer 1 + nonlinearity
        x = self.act(self.fc2(x))  # layer 2 + nonlinearity
        return self.fc3(x)  # raw scores for 10 classes

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)  # make runs repeatable

# simple state buckets
if "model" not in st.session_state: st.session_state.model = None  # hold current model
if "train_log" not in st.session_state: st.session_state.train_log = []  # (epoch, loss, acc)
if "acts" not in st.session_state: st.session_state.acts = {}  # latest activations by layer name

def register_hooks(model, acts_dict):
    def grab(name):
        def hook(_m, _in, out): acts_dict[name] = out.detach().cpu()  # store activation tensor
        return hook
    h1 = model.fc1.register_forward_hook(grab("fc1"))  # tap after fc1
    h2 = model.fc2.register_forward_hook(grab("fc2"))  # tap after fc2
    return [h1, h2]  # so we can remove later

def neuron_digit_profile(model, layer_name, neuron_id, loader, max_batches=60):
    model.eval()  # eval mode
    acts = {}
    handle = getattr(model, layer_name).register_forward_hook(lambda m,i,o: acts.setdefault("a", o.detach().cpu()))  # capture
    buckets = [[] for _ in range(10)]  # one list per digit
    seen = 0
    with torch.no_grad():  # no gradients needed here
        for xb, yb in loader:
            _ = model(xb)  # forward to fill acts["a"]
            vals = acts["a"][:, neuron_id].numpy()  # grab this neuron's activations
            for v, y in zip(vals, yb.numpy()): buckets[int(y)].append(float(v))  # bin by true label
            seen += 1
            if seen >= max_batches: break  # cap work
    handle.remove()  # clean up
    return np.array([np.mean(b) if b else 0.0 for b in buckets])  # average per digit

def top_activating_images(model, layer_name, neuron_id, loader, k=5, max_batches=80):
    model.eval()
    acts = {}
    handle = getattr(model, layer_name).register_forward_hook(lambda m,i,o: acts.setdefault("a", o.detach().cpu()))  # capture
    best = []
    with torch.no_grad():
        for xb, _ in loader:
            _ = model(xb)  # fill acts
            vals = acts["a"][:, neuron_id].numpy()  # activations for this neuron
            for v, img in zip(vals, xb): best.append((float(v), img.view(28,28).numpy()))  # keep score+image
            if len(best) > 3000: best = sorted(best, key=lambda t: t[0], reverse=True)[:k]  # keep top-k window
    handle.remove()
    best = sorted(best, key=lambda t: t[0], reverse=True)[:k]  # final top-k
    return best

def tsne_or_pca(X, y):
    try:
        X2 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                  n_iter=400, random_state=42, method="barnes_hut", verbose=0).fit_transform(X)  # try tsne
        title = "t-SNE"  # label
    except Exception:
        X2 = PCA(n_components=2, random_state=42).fit_transform(X)  # fallback if tsne errors
        title = "PCA (t-SNE fallback)"  # label
    return X2, title

def summarize_confusions(cm):
    off = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0: off.append((i, j, int(cm[i, j])))  # collect off-diagonal counts
    return sorted(off, key=lambda x: x[2], reverse=True)  # most frequent errors first

col_btn, _ = st.columns([1, 3])  # small left column for actions
with col_btn:
    if st.button("train / retrain model", use_container_width=True):  # fire training
        seed_all(42)  # reproducible run
        model = SimpleNN(config)  # new model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # standard adam
        loss_fn = nn.CrossEntropyLoss()  # multiclass loss

        hooks = register_hooks(model, st.session_state.acts)  # capture layer outputs during training
        st.session_state.train_log = []  # reset log
        prog = st.progress(0, text="training...")  # simple progress bar

        for ep in range(epochs):
            correct, total, last_loss = 0, 0, 0.0  # track stats
            for xb, yb in train_loader:
                pred = model(xb)  # forward pass
                loss = loss_fn(pred, yb)  # compute loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()  # one opt step
                last_loss = loss.item()  # remember last batch loss
                correct += (pred.argmax(1) == yb).sum().item()  # count hits
                total += yb.size(0)  # count samples
            acc = correct / total  # epoch accuracy
            st.session_state.train_log.append((ep+1, last_loss, acc))  # store epoch summary
            prog.progress((ep+1)/epochs, text=f"epoch {ep+1}/{epochs} • loss={last_loss:.4f} • acc={acc:.4f}")  # update bar

        for h in hooks: h.remove()  # cleanup hooks
        st.session_state.model = model  # persist trained model
        st.success("training complete")  # done

model = st.session_state.model  # pull model
if model is not None:
    with st.expander("training progress", expanded=True):
        e, l, a = zip(*st.session_state.train_log)  # unpack logs
        fig, ax = plt.subplots(figsize=(6,3))  # compact plot
        ax.plot(l, label="loss")  # loss curve
        ax2 = ax.twinx(); ax2.plot(a, label="acc", color="tab:orange")  # accuracy on twin axis
        ax.set_title("training loss (line) and accuracy (orange)")  # combined view
        ax.set_xlabel("epoch"); ax.set_xticks(range(len(e))); ax.set_xticklabels(e)  # tidy axis
        st.pyplot(fig)
        st.caption("as loss drops and accuracy rises, the model is fitting the digits better")  # quick read

    st.sidebar.header("neuron explorer")  # side controls
    layer_opt = st.sidebar.selectbox("layer", ["fc1", "fc2"])  # choose layer
    hidden_size = config["hidden1"] if layer_opt == "fc1" else config["hidden2"]  # get size
    neuron_id = st.sidebar.slider("neuron id", 0, hidden_size-1, 0, 1)  # pick neuron

    st.subheader("weight map")  # pixel-space template
    if layer_opt == "fc1":
        w = model.fc1.weight[neuron_id].detach().cpu().numpy()  # direct weights to pixels
        Wimg = w.reshape(28,28)  # reshape to image
    else:
        w2 = model.fc2.weight[neuron_id].detach().cpu().numpy()  # fc2 weights over fc1 units
        W1 = model.fc1.weight.detach().cpu().numpy()  # fc1 maps units to pixels
        w_eff = w2 @ W1  # project fc2 neuron back to input pixels
        Wimg = w_eff.reshape(28,28)  # image view
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(Wimg, cmap="seismic", ax=ax, cbar=True)  # red = increases activation, blue = decreases
    ax.set_title(f"weights of {layer_opt} neuron {neuron_id}")  # simple title
    st.pyplot(fig)
    st.caption("red pixels boost activation; blue reduce it. this shows the pattern the neuron responds to.")

    st.subheader("neuron digit profile")  # average activation per label
    profile = neuron_digit_profile(model, layer_opt, neuron_id, test_loader)  # means over digits
    st.bar_chart(profile)  # quick bar chart
    top3 = np.argsort(profile)[::-1][:3].tolist()  # strongest digits
    st.caption(f"higher bars mean stronger response. neuron {neuron_id} fires most for digits {top3}")  # takeaway

    st.subheader("top activating images")  # most triggering examples
    best = top_activating_images(model, layer_opt, neuron_id, test_loader, k=5)  # top k images
    fig, axs = plt.subplots(1, len(best), figsize=(10,2))
    for i,(act,img) in enumerate(best):
        axs[i].imshow(img, cmap="gray"); axs[i].axis("off"); axs[i].set_title(f"a:{act:.2f}")  # show score
    st.pyplot(fig)
    st.caption("these are the examples that push this neuron the hardest")  # quick read

    st.subheader("layer embedding (t-SNE / PCA)")  # 2d view of activations
    acts_list, labels = [], []  # collect features and labels
    with torch.no_grad():
        handles = register_hooks(model, st.session_state.acts)  # latch layer outputs
        cnt = 0
        for xb, yb in test_loader:
            _ = model(xb)  # fill acts
            A = st.session_state.acts[layer_opt].numpy().astype(np.float32)  # get current batch activations
            acts_list.append(A); labels.append(yb.numpy())  # store
            cnt += len(yb)
            if cnt >= 800: break  # cap compute
        for h in handles: h.remove()  # cleanup
    X = np.concatenate(acts_list, axis=0)[:800]; y = np.concatenate(labels, axis=0)[:800]  # stack arrays
    X2, title = tsne_or_pca(X, y)  # 2d projection
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(X2[:,0], X2[:,1], c=y, cmap="tab10", s=12, alpha=0.8)  # colored by label
    legend = ax.legend(*sc.legend_elements(num=10), title="digits", bbox_to_anchor=(1.02,1), loc="upper left")  # quick legend
    ax.add_artist(legend); ax.set_title(f"{title} of {layer_opt} activations")  # label plot
    st.pyplot(fig)
    st.caption("each dot is an image; tighter clusters mean the layer separates those digits better")  # interpretation

    st.subheader("confusion matrix")  # where predictions go wrong
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)  # forward
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())  # predicted labels
            y_true.extend(yb.cpu().numpy().tolist())  # true labels
    y_true, y_pred = np.array(y_true), np.array(y_pred)  # arrays for metrics
    acc = (y_true == y_pred).mean()  # overall accuracy
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))  # counts table
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax)  # darker = more examples
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"accuracy = {acc:.3f}")  # tidy labels
    st.pyplot(fig)
    confs = summarize_confusions(cm)  # sort top errors
    st.caption(f"most common mix-ups: {confs[:3]}")  # quick scan of mistakes

    st.subheader("noise robustness test")  # simple stress test
    idx = st.slider("test image index", 0, len(test_data)-1, 0)  # pick a sample
    img, label = test_data[idx]  # get item
    noise = st.slider("noise level", 0.0, 0.8, 0.4, 0.05)  # additive gaussian
    noisy = torch.clamp(img + noise * torch.randn_like(img), 0, 1)  # clip to valid range
    with torch.no_grad():
        probs = F.softmax(model(noisy), dim=1)[0].numpy()  # forward (model flattens)
    c1, c2 = st.columns(2)  # side-by-side
    with c1:
        st.image(img.view(28,28).numpy(), caption=f"original (label {label})", use_column_width=True)
        st.image(noisy.view(28,28).numpy(), caption="noisy version", use_column_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(range(10), probs); ax.set_title("predictions on noisy image"); ax.set_xticks(range(10))  # bar chart
        st.pyplot(fig)
        st.caption("if one bar still dominates, the model is robust at this noise level")  # rule of thumb
else:
    st.info("choose an architecture and click train to start")  # initial hint