"""
Shared training utilities: device selection, checkpointing, LR scheduling,
metric tracking, reproducibility, and W&B integration.
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device string to a torch.device.

    'auto' preference order: MPS (Apple Silicon) > CUDA > CPU.
    """
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Data path helpers
# ---------------------------------------------------------------------------

def find_image_dir(raw_dir: str) -> str:
    """
    Locate the directory containing X-ray .png images.

    Checks, in order:
      1. raw_dir/images/          (flat layout from download script)
      2. raw_dir/images_001/ …    (split-archive layout)
      3. raw_dir itself           (only if it contains .png files)

    Raises FileNotFoundError with an actionable message if nothing is found.
    """
    # 1. Flat layout
    flat = os.path.join(raw_dir, "images")
    if os.path.isdir(flat):
        return flat

    # 2. Split-archive layout (images_001, images_002, …)
    import glob as _glob
    split_dirs = sorted(_glob.glob(os.path.join(raw_dir, "images_*")))
    if split_dirs and os.path.isdir(split_dirs[0]):
        return raw_dir

    # 3. Images directly in raw_dir
    sample = _glob.glob(os.path.join(raw_dir, "*.png"))
    if sample:
        return raw_dir

    raise FileNotFoundError(
        f"No X-ray images found under '{raw_dir}'. Expected one of:\n"
        f"  • {flat}/  (flat layout)\n"
        f"  • {raw_dir}/images_001/ … (split archives)\n"
        f"  • .png files directly in {raw_dir}/\n"
        "Run scripts/download_data.sh to download and organise the dataset."
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def save_training_state(path: str, **kwargs):
    """Save full training state for resuming (models, optimizer, scheduler, epoch, etc.)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kwargs, path)


def load_training_state(path: str, device: torch.device) -> dict:
    """Load a full training state checkpoint for resuming."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    print(f"Resuming from {path} (epoch {state.get('epoch', '?')})")
    return state


def load_encoder_weights(encoder: nn.Module, checkpoint_path: str, device: torch.device):
    """Load only encoder weights from a pre-training checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Support both bare state_dict and wrapped {'encoder': state_dict}
    state_dict = ckpt.get("encoder", ckpt)
    encoder.load_state_dict(state_dict)
    print(f"Loaded encoder weights from {checkpoint_path}")
    return encoder


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Linear warm-up for `warmup_epochs`, then cosine annealing to 0.
    Returns a LambdaLR scheduler.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # epoch is 0-based: first epoch runs at 1/warmup_epochs, reaching
            # the full LR on the last warmup epoch (not one epoch late, and
            # never at the near-zero factor a bare epoch/warmup_epochs gives).
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Class-imbalance weighting (multi-label)
# ---------------------------------------------------------------------------

def compute_pos_weight(label_matrix, device: torch.device) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.

    pos_weight[c] = (# negative examples for class c) / (# positive examples for class c)

    This compensates for the severe class imbalance in NIH Chest X-ray14,
    where some pathologies appear in < 0.5 % of images.
    """
    pos_counts = label_matrix.sum(axis=0).astype(float)
    neg_counts = (len(label_matrix) - pos_counts).astype(float)
    pos_weight = neg_counts / (pos_counts + 1e-8)
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Update with latest metric. Returns True if training should stop."""
        if metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Simple metric accumulator
# ---------------------------------------------------------------------------

class AverageMeter:
    """Track a running mean of a scalar (e.g., loss)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# Weights & Biases helpers
# ---------------------------------------------------------------------------

def wandb_available() -> bool:
    return _wandb is not None


def init_wandb(config: dict, project: str, run_name: str | None = None):
    """Initialise a W&B run. Returns the run object or None if disabled/missing."""
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    if _wandb is None:
        print("Warning: wandb not installed — skipping W&B logging. pip install wandb")
        return None
    return _wandb.init(
        project=wandb_cfg.get("project", project),
        name=run_name or wandb_cfg.get("run_name"),
        config=config,
        tags=wandb_cfg.get("tags", []),
    )


def wandb_log(metrics: dict, step: int | None = None):
    """Log metrics if a W&B run is active."""
    if _wandb is not None and _wandb.run is not None:
        _wandb.log(metrics, step=step)


def wandb_finish():
    """Finish the active W&B run if one exists."""
    if _wandb is not None and _wandb.run is not None:
        _wandb.finish()
