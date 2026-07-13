"""
Supervised fine-tuning of the SimCLR pre-trained encoder on NIH Chest X-ray14.

Three experimental modes (set via config['training']['mode']):
  full_finetune      — load SimCLR encoder, train all layers end-to-end.
  linear_probe       — load SimCLR encoder, freeze backbone, train classifier only.
  imagenet_baseline  — load ImageNet-pretrained ResNet50, full fine-tune.

All modes use multi-label BCEWithLogitsLoss with per-class positive weighting
to compensate for the severe class imbalance in this dataset.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.augmentations import FinetuneAugmentation
from src.data.dataset import ChestXrayDataset, build_label_matrix
from src.models.classifier import ChestXrayClassifier
from src.models.encoder import SimCLREncoder
from src.training.utils import (
    AverageMeter,
    EarlyStopping,
    cosine_schedule_with_warmup,
    compute_pos_weight,
    find_image_dir,
    get_device,
    init_wandb,
    load_encoder_weights,
    load_training_state,
    save_checkpoint,
    save_training_state,
    set_seed,
    wandb_finish,
    wandb_log,
)


def _build_model(config: dict, device: torch.device) -> ChestXrayClassifier:
    model_cfg = config["model"]
    train_cfg = config["training"]
    mode = train_cfg["mode"]

    if mode == "imagenet_baseline":
        encoder = SimCLREncoder(backbone=model_cfg["backbone"], pretrained_imagenet=True)
        freeze = False
    else:
        encoder = SimCLREncoder(backbone=model_cfg["backbone"], pretrained_imagenet=False)
        checkpoint_path = model_cfg["pretrained_checkpoint"]
        load_encoder_weights(encoder, checkpoint_path, device)
        freeze = (mode == "linear_probe")

    model = ChestXrayClassifier(
        encoder=encoder,
        num_classes=model_cfg["num_classes"],
        freeze_backbone=freeze,
        hidden_dim=model_cfg["classifier_hidden_dim"],
        dropout=model_cfg["dropout"],
    )
    return model.to(device)


def _build_optimizer(model: ChestXrayClassifier, config: dict):
    train_cfg = config["training"]
    lr = train_cfg["lr"]
    bb_lr = lr * train_cfg["backbone_lr_multiplier"]

    # Differential learning rates: lower LR for backbone to preserve pre-trained features
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": bb_lr},
            {"params": model.classifier.parameters(), "lr": lr},
        ],
        weight_decay=train_cfg["weight_decay"],
    )


def finetune(config: dict, resume_from: str | None = None):
    train_cfg = config["training"]
    data_cfg = config["data"]
    ckpt_cfg = config["checkpointing"]
    log_cfg = config["logging"]
    mode = train_cfg["mode"]

    set_seed(train_cfg.get("seed", 42))

    device = get_device(train_cfg["device"])
    print(f"Device: {device} | Mode: {mode}")

    run = init_wandb(config, project="simclr-finetune", run_name=f"finetune-{mode}")

    # ------------------------------------------------------------------ #
    # Datasets                                                             #
    # ------------------------------------------------------------------ #
    processed_dir = data_cfg["processed_dir"]
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))

    # Locate image directory (handles both flat and split archive structures)
    raw_dir = data_cfg["raw_dir"]
    image_dir = find_image_dir(raw_dir)

    train_aug = FinetuneAugmentation(config, train=True)
    val_aug = FinetuneAugmentation(config, train=False)

    train_ds = ChestXrayDataset(train_df, image_dir, transform=train_aug)
    val_ds = ChestXrayDataset(val_df, image_dir, transform=val_aug)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"] and device.type != "mps",
        persistent_workers=data_cfg["num_workers"] > 0,
        prefetch_factor=4 if data_cfg["num_workers"] > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"] and device.type != "mps",
        persistent_workers=data_cfg["num_workers"] > 0,
        prefetch_factor=4 if data_cfg["num_workers"] > 0 else None,
    )

    # ------------------------------------------------------------------ #
    # Model, loss, optimiser                                               #
    # ------------------------------------------------------------------ #
    model = _build_model(config, device)

    # Per-class positive weight for imbalanced multi-label BCE
    if train_cfg["use_class_weights"]:
        label_matrix = build_label_matrix(train_df)
        pos_weight = compute_pos_weight(label_matrix, device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = _build_optimizer(model, config)
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=train_cfg["warmup_epochs"],
        total_epochs=train_cfg["epochs"],
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ------------------------------------------------------------------ #
    # Resume from checkpoint (if requested)                                #
    # ------------------------------------------------------------------ #
    start_epoch = 1
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    early_stopping = EarlyStopping(
        patience=train_cfg.get("early_stopping_patience", 0),
        min_delta=train_cfg.get("early_stopping_min_delta", 1e-4),
    )
    use_early_stopping = train_cfg.get("early_stopping_patience", 0) > 0

    if resume_from:
        state = load_training_state(resume_from, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_val_loss = state.get("best_val_loss", float("inf"))
        history = state.get("history", {"train_loss": [], "val_loss": []})
        if "early_stopping" in state:
            early_stopping.best = state["early_stopping"]["best"]
            early_stopping.counter = state["early_stopping"]["counter"]
            early_stopping.should_stop = state["early_stopping"]["should_stop"]

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    os.makedirs(ckpt_cfg["save_dir"], exist_ok=True)
    os.makedirs(log_cfg["log_dir"], exist_ok=True)

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        # ---- Train --------------------------------------------------- #
        model.train()
        train_meter = AverageMeter("train_loss")
        t0 = time.time()

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_meter.update(loss.item(), n=images.size(0))

            if step % log_cfg["log_every"] == 0:
                print(
                    f"  Epoch {epoch:03d} | Step {step:04d}/{len(train_loader):04d}"
                    f" | TrainLoss {train_meter.avg:.4f}"
                )

        # ---- Validate ------------------------------------------------ #
        model.eval()
        val_meter = AverageMeter("val_loss")
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
                val_meter.update(loss.item(), n=images.size(0))

        scheduler.step()
        elapsed = time.time() - t0

        train_loss = train_meter.avg
        val_loss = val_meter.avg
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{train_cfg['epochs']:03d}"
            f" | TrainLoss {train_loss:.4f}"
            f" | ValLoss {val_loss:.4f}"
            f" | LR {optimizer.param_groups[-1]['lr']:.2e}"
            f" | {elapsed:.1f}s"
        )

        wandb_log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[-1]["lr"],
        }, step=epoch)

        # ---- Checkpointing ------------------------------------------- #
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if epoch % ckpt_cfg["save_every"] == 0 or is_best:
            ckpt_path = os.path.join(ckpt_cfg["save_dir"], f"model_{mode}_ep{epoch:03d}.pth")
            save_checkpoint(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "mode": mode},
                ckpt_path,
            )
            if is_best:
                best_path = os.path.join(ckpt_cfg["save_dir"], f"best_model_{mode}.pth")
                save_checkpoint(
                    {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "mode": mode},
                    best_path,
                )
                print(f"  [Best] Saved model to {best_path}")

        # Save full training state for resume
        resume_path = os.path.join(ckpt_cfg["save_dir"], f"latest_finetune_{mode}.pth")
        save_training_state(
            resume_path,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            epoch=epoch,
            best_val_loss=best_val_loss,
            history=history,
            mode=mode,
            early_stopping={
                "best": early_stopping.best,
                "counter": early_stopping.counter,
                "should_stop": early_stopping.should_stop,
            },
        )

        if use_early_stopping and early_stopping.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch} (patience={early_stopping.patience})")
            break

    # Save loss history
    log_path = os.path.join(log_cfg["log_dir"], f"finetune_{mode}_loss.txt")
    with open(log_path, "w") as f:
        f.write("epoch\ttrain_loss\tval_loss\n")
        for ep, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"]), start=1):
            f.write(f"{ep}\t{tl:.6f}\t{vl:.6f}\n")

    wandb_finish()
    print(f"\nFine-tuning complete. Best val loss: {best_val_loss:.4f}")
