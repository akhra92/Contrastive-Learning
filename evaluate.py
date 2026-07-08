"""
Full evaluation suite on the NIH Chest X-ray14 test set.

Computes per-class and macro-averaged AUC-ROC, Average Precision, and F1.
Generates: ROC curves, t-SNE embeddings, GradCAM saliency maps, loss curves.
Optionally exports the model to ONNX and/or TorchScript.

Usage:
    python evaluate.py --checkpoint checkpoints/finetune/best_model_full_finetune.pth
    python evaluate.py --checkpoint <path> --mode full_finetune --no_gradcam
    python evaluate.py --checkpoint <path> --export onnx torchscript
"""

import argparse
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.augmentations import FinetuneAugmentation
from src.data.dataset import ALL_CLASSES, ChestXrayDataset
from src.evaluation.metrics import (
    collect_predictions,
    evaluate_multilabel,
    print_metrics,
    tune_thresholds,
)
from src.evaluation.visualize import plot_gradcam, plot_loss_curves, plot_roc_curves, plot_tsne
from src.models.classifier import ChestXrayClassifier
from src.models.encoder import SimCLREncoder
from src.training.utils import find_image_dir, get_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned classifier")
    p.add_argument("--config", default="configs/finetune_config.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to fine-tuned model checkpoint")
    p.add_argument(
        "--mode",
        choices=["full_finetune", "linear_probe", "imagenet_baseline"],
        default=None,
    )
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=None)
    p.add_argument("--no_tsne", action="store_true", help="Skip t-SNE (slow for large datasets)")
    p.add_argument("--no_gradcam", action="store_true", help="Skip GradCAM generation")
    p.add_argument("--output_dir", default="logs", help="Directory for saved figures")
    p.add_argument(
        "--export",
        nargs="+",
        choices=["onnx", "torchscript"],
        default=None,
        help="Export model to ONNX and/or TorchScript after evaluation",
    )
    p.add_argument("--export_dir", default="exports", help="Directory for exported models")
    return p.parse_args()


def select_gradcam_images(model, loader, target_class_idx, device, n_images=8, threshold=0.5):
    """
    Select test images the model *correctly identifies* as having the target
    pathology: ground-truth positive AND predicted positive (prob >= threshold),
    ranked by the model's predicted probability (most confident first).

    Grad-CAM only explains a prediction meaningfully when that prediction is both
    correct and confident, so restricting to true positives keeps the saliency
    maps interpretable. Returns a (n, 1, H, W) tensor, or None if no correctly
    identified positive image exists.
    """
    model.eval()
    pos_images, pos_scores = [], []

    with torch.no_grad():
        for images, labels in loader:
            mask = labels[:, target_class_idx] == 1
            if not mask.any():
                continue
            sel = images[mask]
            logits = model(sel.to(device))
            scores = torch.sigmoid(logits[:, target_class_idx]).cpu()
            correct = scores >= threshold  # keep only true positives
            if not correct.any():
                continue
            pos_images.append(sel[correct])
            pos_scores.append(scores[correct])

    if not pos_images:
        return None

    pos_images = torch.cat(pos_images, dim=0)
    pos_scores = torch.cat(pos_scores, dim=0)
    order = torch.argsort(pos_scores, descending=True)
    top = order[: min(n_images, len(order))]
    return pos_images[top]


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode:
        config["training"]["mode"] = args.mode
    if args.device:
        config["training"]["device"] = args.device

    device = get_device(config["training"]["device"])
    model_cfg = config["model"]
    data_cfg = config["data"]

    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")

    # ------------------------------------------------------------------ #
    # Load model                                                           #
    # ------------------------------------------------------------------ #
    encoder = SimCLREncoder(backbone=model_cfg["backbone"])
    model = ChestXrayClassifier(
        encoder=encoder,
        num_classes=model_cfg["num_classes"],
        hidden_dim=model_cfg["classifier_hidden_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # ------------------------------------------------------------------ #
    # Test dataset                                                         #
    # ------------------------------------------------------------------ #
    test_df = pd.read_csv(os.path.join(data_cfg["processed_dir"], "test.csv"))
    image_dir = find_image_dir(data_cfg["raw_dir"])
    test_aug = FinetuneAugmentation(config, train=False)
    test_ds = ChestXrayDataset(test_df, image_dir, transform=test_aug)
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"] and device.type != "mps",
    )

    # ------------------------------------------------------------------ #
    # Compute metrics                                                       #
    # ------------------------------------------------------------------ #
    # Tune per-class F1 thresholds on the VALIDATION set (never the test set),
    # then apply them to the test set. Training uses BCE with pos_weight, so
    # logits aren't calibrated around 0.5 and a fixed 0.5 threshold gives poor
    # F1 despite good ranking (AUC/AP). AUC and AP are threshold-free.
    print("\nTuning F1 thresholds on the validation set …")
    val_df = pd.read_csv(os.path.join(data_cfg["processed_dir"], "val.csv"))
    val_ds = ChestXrayDataset(val_df, image_dir, transform=test_aug)
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"] and device.type != "mps",
    )
    y_true_val, y_logits_val = collect_predictions(model, val_loader, device)
    thresholds = tune_thresholds(y_true_val, y_logits_val)

    print("\nRunning inference on test set …")
    y_true, y_pred_logits = collect_predictions(model, test_loader, device)

    # Baseline (fixed 0.5) vs tuned per-class thresholds — reported side by side.
    metrics_default = evaluate_multilabel(y_true, y_pred_logits, threshold=0.5)
    metrics = evaluate_multilabel(y_true, y_pred_logits, threshold=thresholds)
    print_metrics(metrics)
    print(
        f"Macro F1 @0.5 (fixed) : {metrics_default['macro_f1']:.4f}"
        f"   →   Macro F1 (tuned): {metrics['macro_f1']:.4f}"
    )

    # Save metrics to file
    os.makedirs(args.output_dir, exist_ok=True)
    mode_str = config["training"]["mode"]
    metrics_path = os.path.join(args.output_dir, f"metrics_{mode_str}.txt")
    with open(metrics_path, "w") as mf:
        mf.write("F1 uses per-class thresholds tuned on the validation set.\n")
        mf.write("AUC-ROC and Avg-Prec are threshold-free.\n\n")
        mf.write(f"{'Class':<25} {'AUC-ROC':>8} {'Avg-Prec':>10} {'F1':>8} {'Thresh':>8}\n")
        for cls_name in ALL_CLASSES:
            if cls_name in metrics:
                m = metrics[cls_name]
                mf.write(
                    f"{cls_name:<25} {m['auc_roc']:>8.4f} {m['avg_precision']:>10.4f}"
                    f" {m['f1']:>8.4f} {m['threshold']:>8.3f}\n"
                )
        mf.write(f"\nMacro AUC-ROC     : {metrics['macro_auc_roc']:.4f}\n")
        mf.write(f"Macro Avg-Prec    : {metrics['macro_ap']:.4f}\n")
        mf.write(f"Macro F1 (tuned)  : {metrics['macro_f1']:.4f}\n")
        mf.write(f"Macro F1 (@0.5)   : {metrics_default['macro_f1']:.4f}\n")
    print(f"Metrics saved to {metrics_path}")

    # ------------------------------------------------------------------ #
    # Visualisations                                                        #
    # ------------------------------------------------------------------ #
    plot_roc_curves(y_true, y_pred_logits, save_path=os.path.join(args.output_dir, "roc_curves.png"))
    plot_loss_curves(log_dir=args.output_dir, save_path=os.path.join(args.output_dir, "loss_curves.png"))

    if not args.no_tsne:
        # Shuffled loader so the class-balanced t-SNE sample is drawn from across
        # the whole test set, not just the first images.
        tsne_loader = DataLoader(
            test_ds,
            batch_size=config["training"]["batch_size"] * 2,
            shuffle=True,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"] and device.type != "mps",
        )
        plot_tsne(
            encoder=model.encoder,
            loader=tsne_loader,
            device=device,
            save_path=os.path.join(args.output_dir, "tsne.png"),
        )

    if not args.no_gradcam:
        # GradCAM only makes sense on images that actually contain the target
        # pathology. Select test images that are ground-truth positive for the
        # target class, ranked by the model's own confidence, so the saliency
        # map explains a correct, confident prediction (not a random negative).
        target_class_idx = 2  # Cardiomegaly
        gradcam_images = select_gradcam_images(
            model, test_loader, target_class_idx, device, n_images=8
        )
        if gradcam_images is None:
            print(
                f"No positive '{ALL_CLASSES[target_class_idx]}' images found in the "
                "test set; skipping GradCAM."
            )
        else:
            plot_gradcam(
                model=model,
                images=gradcam_images,
                target_class_idx=target_class_idx,
                device=device,
                save_path=os.path.join(args.output_dir, "gradcam_cardiomegaly.png"),
            )

    # ------------------------------------------------------------------ #
    # Model export                                                          #
    # ------------------------------------------------------------------ #
    if args.export:
        from export_model import export_onnx, export_torchscript

        image_size = data_cfg["image_size"]
        # Export on CPU for compatibility
        export_device = torch.device("cpu")
        export_model = model.to(export_device)
        export_model.eval()
        dummy_input = torch.randn(1, 1, image_size, image_size, device=export_device)
        base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

        if "onnx" in args.export:
            export_onnx(export_model, dummy_input, os.path.join(args.export_dir, f"{base_name}.onnx"))
        if "torchscript" in args.export:
            export_torchscript(export_model, dummy_input, os.path.join(args.export_dir, f"{base_name}.pt"))

    print("\nEvaluation complete. Outputs saved to:", args.output_dir)



if __name__ == "__main__":
    main()
