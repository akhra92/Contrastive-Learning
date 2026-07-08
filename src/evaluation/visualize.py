"""
Visualisation utilities for the contrastive learning project.

Four key visualisations:
  1. t-SNE of encoder embeddings — verify representation quality.
  2. Per-class ROC curves — evaluate classification performance.
  3. GradCAM saliency maps — interpretability for clinical inspection.
  4. Training loss / metric curves — monitor pre-training and fine-tuning.
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import ALL_CLASSES


# ---------------------------------------------------------------------------
# 1. t-SNE embeddings
# ---------------------------------------------------------------------------

def plot_tsne(
    encoder,
    loader,
    device,
    save_path: str = "logs/tsne.png",
    n_samples: int = 3000,
    perplexity: int = 30,
):
    """
    Extract encoder features and visualise with t-SNE.
    Colour each point by its primary (first-listed) pathology label.
    """
    from sklearn.manifold import TSNE

    encoder.eval()
    features, primary_labels = [], []

    collected = 0
    with torch.no_grad():
        for images, labels in loader:
            if collected >= n_samples:
                break
            images = images.to(device)
            h = encoder(images)
            features.append(h.cpu().numpy())
            # Use argmax as a proxy "primary" label for colouring
            primary_labels.append(labels.numpy().argmax(axis=1))
            collected += images.size(0)

    features = np.concatenate(features, axis=0)[:n_samples]
    primary_labels = np.concatenate(primary_labels, axis=0)[:n_samples]

    print(f"Running t-SNE on {len(features)} samples …")
    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
    ).fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=primary_labels, cmap="tab20", s=4, alpha=0.7)
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(len(ALL_CLASSES)))
    cbar.ax.set_yticklabels(ALL_CLASSES, fontsize=7)
    ax.set_title("t-SNE of Encoder Embeddings", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE saved to {save_path}")


# ---------------------------------------------------------------------------
# 2. ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    save_path: str = "logs/roc_curves.png",
    top_k: int = 8,
):
    """Plot per-class ROC curves for the top-k most prevalent classes."""
    from sklearn.metrics import roc_curve, auc

    y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred_logits, -88, 88)))

    # Select top-k classes by number of positive examples
    pos_counts = y_true.sum(axis=0)
    top_idx = np.argsort(pos_counts)[::-1][:top_k]

    n_cols = 4
    n_rows = (top_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for plot_i, cls_idx in enumerate(top_idx):
        cls_name = ALL_CLASSES[cls_idx]
        if y_true[:, cls_idx].sum() == 0:
            axes[plot_i].axis("off")
            continue
        fpr, tpr, _ = roc_curve(y_true[:, cls_idx], y_pred_prob[:, cls_idx])
        roc_auc = auc(fpr, tpr)
        ax = axes[plot_i]
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(cls_name, fontsize=10)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    for j in range(plot_i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-class ROC Curves", fontsize=14)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curves saved to {save_path}")


# ---------------------------------------------------------------------------
# 3. GradCAM saliency maps
# ---------------------------------------------------------------------------

def plot_gradcam(
    model,
    images: torch.Tensor,
    target_class_idx: int,
    device,
    save_path: str = "logs/gradcam.png",
    n_images: int = 4,
):
    """
    Overlay GradCAM heatmaps on the original X-ray images.

    Requires `grad-cam` package: pip install grad-cam

    Args:
        model            : ChestXrayClassifier in eval mode.
        images           : (B, 1, H, W) tensor (un-normalised for display).
        target_class_idx : index into ALL_CLASSES for the target pathology.
        device           : torch device.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("GradCAM unavailable. Install with: pip install grad-cam")
        return

    from src.models.encoder import get_gradcam_target_layer

    model.eval()
    target_layers = get_gradcam_target_layer(model.encoder)

    cam = GradCAM(model=model, target_layers=target_layers)

    imgs_to_show = images[:n_images].to(device)
    # One target per image: pytorch_grad_cam zips targets with the batch, so a
    # single-element list would only produce a real CAM for the first image and
    # leave the rest near-zero (blank overlays).
    targets = [ClassifierOutputTarget(target_class_idx)] * len(imgs_to_show)
    grayscale_cam = cam(input_tensor=imgs_to_show, targets=targets)

    fig, axes = plt.subplots(2, n_images, figsize=(4 * n_images, 8))
    for i in range(min(n_images, len(imgs_to_show))):
        raw = imgs_to_show[i, 0].cpu().numpy()
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        rgb = np.stack([raw, raw, raw], axis=-1)

        axes[0, i].imshow(raw, cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        cam_image = show_cam_on_image(rgb.astype(np.float32), grayscale_cam[i], use_rgb=True)
        axes[1, i].imshow(cam_image)
        axes[1, i].set_title(f"GradCAM: {ALL_CLASSES[target_class_idx]}")
        axes[1, i].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"GradCAM saved to {save_path}")


# ---------------------------------------------------------------------------
# 4. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    log_dir: str = "logs",
    save_path: str = "logs/loss_curves.png",
):
    """Plot pre-training loss and fine-tuning train/val losses from log files."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pre-training loss
    pretrain_log = os.path.join(log_dir, "pretrain_loss.txt")
    if os.path.isfile(pretrain_log):
        data = np.loadtxt(pretrain_log)
        axes[0].plot(data[:, 0], data[:, 1], lw=2, color="steelblue")
        axes[0].set_title("SimCLR Pre-training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("NT-Xent Loss")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No pretrain log found", ha="center", va="center", transform=axes[0].transAxes)

    # Fine-tuning losses — overlay all modes
    colors = {"full_finetune": "steelblue", "linear_probe": "darkorange", "imagenet_baseline": "green"}
    found_any = False
    for mode, color in colors.items():
        ft_log = os.path.join(log_dir, f"finetune_{mode}_loss.txt")
        if os.path.isfile(ft_log):
            data = np.loadtxt(ft_log, skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            axes[1].plot(data[:, 0], data[:, 1], lw=2, linestyle="-", color=color, label=f"{mode} train")
            axes[1].plot(data[:, 0], data[:, 2], lw=2, linestyle="--", color=color, label=f"{mode} val")
            found_any = True

    if found_any:
        axes[1].set_title("Fine-tuning Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("BCE Loss")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No finetune log found", ha="center", va="center", transform=axes[1].transAxes)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curves saved to {save_path}")
