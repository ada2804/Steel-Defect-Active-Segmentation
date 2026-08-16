"""
Supervised Training Loop for DinoUNetDecoder.

Trains the progressive convolutional decoder on top of frozen DINOv2 ViT-B/14
features using BCEWithLogitsLoss and monitors Validation Dice Scores.
Pre-caches frozen features in memory for lightning-fast training throughput.
Saves the best decoder weights to results/best_unet_decoder.pth.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.build_index import extract_patch_embeddings
from src.dataset import SeverstalUNetDataset
from src.model import DinoUNetDecoder


class CachedFeatureDataset(Dataset):
    """Memory-resident dataset holding pre-extracted DINOv2 patch tokens and masks."""

    def __init__(self, features: torch.Tensor, masks: torch.Tensor, image_ids: List[str]):
        self.features = features  # [N, 256, 768]
        self.masks = masks        # [N, 4, 224, 224]
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.features[idx], self.masks[idx], self.image_ids[idx]


def compute_batch_dice(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes per-class and mean Dice score for a batch of predictions.
    """
    probs = torch.sigmoid(pred_logits)
    preds = (probs >= threshold).float()

    dices = []
    class_dices = {}

    for c in range(4):
        p_c = preds[:, c].reshape(-1)
        t_c = targets[:, c].reshape(-1)

        intersection = torch.sum(p_c * t_c).item()
        total = torch.sum(p_c).item() + torch.sum(t_c).item()

        if total == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection + smooth) / (total + smooth)

        class_dices[f"dice_class_{c+1}"] = float(dice)
        dices.append(dice)

    class_dices["mean_dice"] = float(np.mean(dices))
    return class_dices


def extract_and_cache_dataset(
    model: DinoUNetDecoder,
    dataset: SeverstalUNetDataset,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> CachedFeatureDataset:
    """
    Passes the dataset through the frozen DINOv2 backbone once and caches [N, 256, 768] tokens.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_features = []
    all_masks = []
    all_ids = []

    print(f"[Caching] Extracting frozen DINOv2 representations for {len(dataset)} images...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Caching DINOv2 Embeddings"):
            imgs = batch["image"].to(device)
            masks = batch["mask"]  # [B, 4, 224, 224]
            img_ids = batch["image_id"]

            patch_tokens = extract_patch_embeddings(model.encoder, imgs, device)  # [B, 256, 768]
            all_features.append(patch_tokens.cpu())
            all_masks.append(masks.cpu())
            all_ids.extend(img_ids)

    cached_features = torch.cat(all_features, dim=0)
    cached_masks = torch.cat(all_masks, dim=0)

    print(f"[Caching] Caching completed: Features {cached_features.shape}, Masks {cached_masks.shape}")
    return CachedFeatureDataset(cached_features, cached_masks, all_ids)


def train_one_epoch(
    model: DinoUNetDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs one training epoch over cached features."""
    model.decoder.train()

    total_loss = 0.0
    all_dices = []

    for features, targets, _ in loader:
        features = features.to(device)  # [B, 256, 768]
        targets = targets.to(device)    # [B, 4, 224, 224]

        optimizer.zero_grad()
        logits = model.forward_from_features(features)  # [B, 4, 224, 224]
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        batch_dice = compute_batch_dice(logits.detach(), targets)
        all_dices.append(batch_dice["mean_dice"])

    epoch_loss = total_loss / len(loader.dataset)
    epoch_dice = float(np.mean(all_dices))
    return epoch_loss, epoch_dice


@torch.no_grad()
def evaluate_validation(
    model: DinoUNetDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluates validation loss and per-class Dice scores from cached features."""
    model.decoder.eval()

    total_loss = 0.0
    class_metric_sums = {f"dice_class_{c}": 0.0 for c in range(1, 5)}
    class_metric_sums["mean_dice"] = 0.0
    num_batches = 0

    for features, targets, _ in loader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model.forward_from_features(features)
        loss = criterion(logits, targets)

        total_loss += loss.item() * features.size(0)
        batch_dices = compute_batch_dice(logits, targets)

        for k in class_metric_sums:
            class_metric_sums[k] += batch_dices[k]
        num_batches += 1

    val_loss = total_loss / len(loader.dataset)
    val_metrics = {k: v / max(1, num_batches) for k, v in class_metric_sums.items()}
    return val_loss, val_metrics


def train_decoder(
    img_dir: str = "data/severstal/train_images",
    csv_path: str = "data/severstal/train.csv",
    output_model_path: str = "results/best_unet_decoder.pth",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_split: float = 0.20,
    subset_fraction: Optional[float] = None,
    device_str: Optional[str] = None,
) -> DinoUNetDecoder:
    """
    Complete training pipeline for the U-Net Decoder.
    """
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Training] Using Device: {device}")
    print(f"[Training] Loading Dataset from {img_dir}...")

    full_raw_dataset = SeverstalUNetDataset(
        img_dir=img_dir,
        csv_path=csv_path,
        subset_fraction=subset_fraction,
    )

    # Initialize Hybrid Model
    print("[Training] Initializing DinoUNetDecoder...")
    model = DinoUNetDecoder(num_classes=4, device=device)

    # Pre-extract representations (runs once!)
    cached_dataset = extract_and_cache_dataset(model, full_raw_dataset, batch_size=batch_size, device=device)

    # 80/20 train/validation split
    total_samples = len(cached_dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(cached_dataset, [train_size, val_size])

    print(f"[Training] Dataset Split: {train_size} Train Samples | {val_size} Validation Samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer strictly for decoder parameters
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_mean_dice": [],
        "val_class_1_dice": [],
        "val_class_2_dice": [],
        "val_class_3_dice": [],
        "val_class_4_dice": [],
    }

    best_val_dice = -1.0
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)

    print("\n" + "=" * 70)
    print(f"STARTING U-NET DECODER SUPERVISED TRAINING ({epochs} EPOCHS)")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_validation(model, val_loader, criterion, device)

        val_mean_dice = val_metrics["mean_dice"]
        epoch_time = time.time() - epoch_start

        # Record metrics
        history["train_loss"].append(float(train_loss))
        history["train_dice"].append(float(train_dice))
        history["val_loss"].append(float(val_loss))
        history["val_mean_dice"].append(float(val_mean_dice))
        for c in range(1, 5):
            history[f"val_class_{c}_dice"].append(float(val_metrics[f"dice_class_{c}"]))

        print(f"  [Epoch {epoch}/{epochs}] Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_mean_dice:.4f} ({epoch_time:.2f}s)", flush=True)
        print(f"    Class 1: {val_metrics['dice_class_1']:.4f} | Class 2: {val_metrics['dice_class_2']:.4f} | Class 3: {val_metrics['dice_class_3']:.4f} | Class 4: {val_metrics['dice_class_4']:.4f}", flush=True)

        # Checkpointing
        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            print(f"  --> Saved new BEST decoder checkpoint to {output_model_path} (Val Dice: {val_mean_dice:.4f})", flush=True)
            torch.save(
                {
                    "epoch": epoch,
                    "decoder_state_dict": model.decoder.state_dict(),
                    "best_val_dice": best_val_dice,
                    "val_metrics": val_metrics,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                output_model_path,
            )

    total_training_time = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETE in {total_training_time:.2f}s! Best Validation Mean Dice: {best_val_dice:.4f}", flush=True)
    print("=" * 70, flush=True)

    # Save training history JSON
    history_json_path = os.path.join(os.path.dirname(output_model_path), "training_history.json")
    with open(history_json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_json_path}", flush=True)

    # Plot training curves
    _plot_curves(history, os.path.join(os.path.dirname(output_model_path), "training_curves.png"))

    return model


def _plot_curves(history: Dict[str, List[float]], output_path: str) -> None:
    """Plots training and validation loss and Dice curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Loss Plot
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-s", label="Val Loss")
    axes[0].set_title("BCE Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Dice Score Plot
    axes[1].plot(epochs, history["train_dice"], "b-o", label="Train Mean Dice")
    axes[1].plot(epochs, history["val_mean_dice"], "r-s", label="Val Mean Dice")
    for c in range(1, 5):
        axes[1].plot(epochs, history[f"val_class_{c}_dice"], "--", label=f"Val Class {c}")
    axes[1].set_title("Dice Overlap Progression", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curve figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Progressive U-Net Decoder on Frozen DINOv2 Backbone.")
    parser.add_argument("--img_dir", type=str, default="data/severstal/train_images", help="Path to train images directory")
    parser.add_argument("--csv_path", type=str, default="data/severstal/train.csv", help="Path to train.csv")
    parser.add_argument("--output_model", type=str, default="results/best_unet_decoder.pth", help="Path to save best weights")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--val_split", type=float, default=0.20, help="Validation set split fraction")
    parser.add_argument("--subset_fraction", type=float, default=None, help="Fraction of dataset to use for quick experiments")

    args = parser.parse_args()

    train_decoder(
        img_dir=args.img_dir,
        csv_path=args.csv_path,
        output_model_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        subset_fraction=args.subset_fraction,
    )


if __name__ == "__main__":
    main()
