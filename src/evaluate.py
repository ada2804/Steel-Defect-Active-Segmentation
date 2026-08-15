"""
Evaluation and Segmentation Pipeline for Severstal Steel Defect Detection.

Performs zero-shot anomaly segmentation using DINOv2 patch features and a
quantized FAISS IVFPQ memory bank with k-NN majority voting.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import faiss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from src.dataset import DINOV2_MEAN, DINOV2_STD, SeverstalDataset
from src.build_index import extract_patch_embeddings, load_dinov2_model


# Color palette for defect classes (RGBA format for overlay)
CLASS_COLORS = {
    0: (0, 0, 0, 0),          # Background: Transparent
    1: (0, 220, 100, 180),    # Class 1: Vivid Emerald Green
    2: (255, 200, 0, 180),    # Class 2: Bright Amber Yellow
    3: (230, 40, 40, 180),    # Class 3: Crimson Red
    4: (30, 140, 255, 180),   # Class 4: Deep Azure Blue
}

CLASS_NAMES = {
    0: "Normal",
    1: "Defect Class 1 (Pitted/Inclusion)",
    2: "Defect Class 2 (Edge Imperfection)",
    3: "Defect Class 3 (Surface Scratch/Gouge)",
    4: "Defect Class 4 (Patch Defect)",
}


class DefectInspector:
    """
    Inference and visualization inspector for steel defect segmentation.
    """

    def __init__(
        self,
        index_path: str = "data/severstal_ivfpq.index",
        labels_path: str = "data/severstal_labels.npy",
        k: int = 5,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the inspector with DINOv2 backbone and FAISS IVFPQ memory bank.

        Args:
            index_path: Path to serialized FAISS .index file.
            labels_path: Path to serialized numpy patch labels array.
            k: Number of nearest neighbors for majority voting (default: 5).
            device: torch device (cuda or cpu).
        """
        self.k = k
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Loading DINOv2 model on {self.device}...")
        self.model = load_dinov2_model(self.device)

        print(f"Loading FAISS IVFPQ index from {index_path}...")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run build_index.py first.")
        self.index = faiss.read_index(index_path)

        # Set nprobe for IVFPQ search (number of Voronoi centroids to inspect)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(16, getattr(self.index, "nlist", 100))

        print(f"Loading patch labels from {labels_path}...")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels array not found at {labels_path}. Please run build_index.py first.")
        self.labels = np.load(labels_path)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
            ]
        )

    def predict_patch_grid(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patch embeddings and predict 16x16 patch defect classes via k-NN majority voting.

        Args:
            image_tensor: Normalized tensor of shape [1, 3, 224, 224] or [3, 224, 224].

        Returns:
            grid_preds: 2D numpy array of shape (16, 16) with predicted class IDs (0..4).
            confidence_grid: 2D numpy array of shape (16, 16) with defect confidence (0.0 to 1.0).
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Extract [1, 256, 768] patch features
        patch_embeds = extract_patch_embeddings(self.model, image_tensor, self.device)
        patch_embeds_np = patch_embeds.squeeze(0).numpy().astype(np.float32)  # [256, 768]

        # Query FAISS index for k nearest neighbors
        distances, indices = self.index.search(patch_embeds_np, self.k)

        # Retrieve neighbor labels: shape [256, k]
        neighbor_labels = self.labels[indices]

        grid_preds = np.zeros(256, dtype=np.int32)
        confidence = np.zeros(256, dtype=np.float32)

        for i in range(256):
            nn_labels = neighbor_labels[i]
            # Majority vote
            unique_vals, counts = np.unique(nn_labels, return_counts=True)
            majority_label = unique_vals[np.argmax(counts)]
            grid_preds[i] = majority_label
            # Confidence is the fraction of defect votes (classes 1..4)
            defect_votes = np.sum(nn_labels > 0)
            confidence[i] = defect_votes / self.k

        return grid_preds.reshape((16, 16)), confidence.reshape((16, 16))

    def evaluate_image(
        self,
        image_path_or_tensor: Union[str, torch.Tensor],
        ground_truth_mask: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """
        Perform end-to-end defect inspection, full-resolution upsampling, and visual rendering.

        Args:
            image_path_or_tensor: Path to image file, or preprocessed [3, 224, 224] tensor.
            ground_truth_mask: Optional [4, 256, 1600] binary array of true defects.
            output_path: Path to save visualization PNG.
            image_id: Optional identifier for display title.

        Returns:
            result_dict: Dictionary containing predicted masks, metrics, and visualization paths.
        """
        # Load original image for visualization
        if isinstance(image_path_or_tensor, str):
            orig_pil = Image.open(image_path_or_tensor).convert("RGB")
            orig_np = np.array(orig_pil)
            img_tensor = self.transform(orig_pil).unsqueeze(0)
            if image_id is None:
                image_id = os.path.basename(image_path_or_tensor)
        else:
            img_tensor = image_path_or_tensor
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            # Inverse normalize for display
            unnorm = img_tensor.squeeze(0).cpu().clone()
            for t, m, s in zip(unnorm, DINOV2_MEAN, DINOV2_STD):
                t.mul_(s).add_(m)
            orig_np = (torch.clamp(unnorm, 0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # Resize display placeholder to 256x1600
            orig_np = cv2.resize(orig_np, (1600, 256), interpolation=cv2.INTER_CUBIC)
            if image_id is None:
                image_id = "sample_image"

        orig_h, orig_w = orig_np.shape[:2]

        # Predict 16x16 patch classes and defect confidence
        patch_preds_16x16, conf_16x16 = self.predict_patch_grid(img_tensor)

        # Upsample 16x16 patch predictions to full image dimensions (1600x256)
        full_pred_mask = cv2.resize(
            patch_preds_16x16.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        full_conf_map = cv2.resize(
            conf_16x16,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Separate into 4 binary defect masks [4, H, W]
        pred_4ch = np.zeros((4, orig_h, orig_w), dtype=np.uint8)
        for c in range(1, 5):
            pred_4ch[c - 1] = (full_pred_mask == c).astype(np.uint8)

        # Compute metrics if ground truth is available
        metrics: Dict[str, float] = {}
        if ground_truth_mask is not None:
            metrics = compute_segmentation_metrics(pred_4ch, ground_truth_mask)

        # Generate side-by-side visualization
        if output_path is not None:
            self._render_figure(
                orig_np=orig_np,
                pred_mask=full_pred_mask,
                conf_map=full_conf_map,
                gt_mask=ground_truth_mask,
                metrics=metrics,
                image_id=image_id,
                output_path=output_path,
            )

        return {
            "predicted_mask_4ch": pred_4ch,
            "predicted_class_map": full_pred_mask,
            "confidence_map": full_conf_map,
            "patch_preds_16x16": patch_preds_16x16,
            "metrics": metrics,
        }

    def _render_figure(
        self,
        orig_np: np.ndarray,
        pred_mask: np.ndarray,
        conf_map: np.ndarray,
        gt_mask: Optional[np.ndarray],
        metrics: Dict[str, float],
        image_id: str,
        output_path: str,
    ) -> None:
        """Render high-contrast, multi-panel inspection figure."""
        has_gt = gt_mask is not None
        num_rows = 3 if has_gt else 2

        fig, axes = plt.subplots(num_rows, 1, figsize=(16, 3.2 * num_rows), dpi=150)
        if num_rows == 1:
            axes = [axes]

        # Panel 1: Raw Steel Strip
        axes[0].imshow(orig_np)
        axes[0].set_title(f"Inspection Target: {image_id} (Raw Steel Surface)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Predicted Defect Segmentation Overlay
        overlay = orig_np.copy()
        for c in range(1, 5):
            mask_c = pred_mask == c
            if np.any(mask_c):
                color = CLASS_COLORS[c][:3]
                alpha = CLASS_COLORS[c][3] / 255.0
                colored_layer = np.zeros_like(orig_np)
                colored_layer[mask_c] = color
                overlay = np.where(
                    mask_c[..., None],
                    (overlay * (1 - alpha) + colored_layer * alpha).astype(np.uint8),
                    overlay,
                )

        axes[1].imshow(overlay)
        pred_classes_found = [c for c in range(1, 5) if np.any(pred_mask == c)]
        found_str = ", ".join([f"Class {c}" for c in pred_classes_found]) if pred_classes_found else "No Defects Detected (Normal)"
        axes[1].set_title(f"Zero-Shot IVFPQ Segmentation Prediction: {found_str}", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Ground Truth (if provided)
        if has_gt:
            gt_overlay = orig_np.copy()
            for c in range(1, 5):
                mask_c = gt_mask[c - 1] > 0
                if np.any(mask_c):
                    color = CLASS_COLORS[c][:3]
                    alpha = CLASS_COLORS[c][3] / 255.0
                    colored_layer = np.zeros_like(orig_np)
                    colored_layer[mask_c] = color
                    gt_overlay = np.where(
                        mask_c[..., None],
                        (gt_overlay * (1 - alpha) + colored_layer * alpha).astype(np.uint8),
                        gt_overlay,
                    )
            dice_info = f" | Mean Dice: {metrics.get('mean_dice', 0.0):.3f}" if "mean_dice" in metrics else ""
            axes[2].imshow(gt_overlay)
            axes[2].set_title(f"Ground Truth Defect Mask{dice_info}", fontsize=11, fontweight="bold")
            axes[2].axis("off")

        # Add legend
        legend_patches = [
            mpatches.Patch(color=np.array(CLASS_COLORS[1][:3]) / 255.0, label="Class 1 (Pitted)"),
            mpatches.Patch(color=np.array(CLASS_COLORS[2][:3]) / 255.0, label="Class 2 (Edge)"),
            mpatches.Patch(color=np.array(CLASS_COLORS[3][:3]) / 255.0, label="Class 3 (Scratch)"),
            mpatches.Patch(color=np.array(CLASS_COLORS[4][:3]) / 255.0, label="Class 4 (Patch)"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=True, fontsize=10)

        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved inspection result to {output_path}")


def compute_segmentation_metrics(
    pred_mask_4ch: np.ndarray,
    gt_mask_4ch: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes per-class and mean Dice and IoU coefficients between prediction and ground truth.
    """
    metrics = {}
    dices = []
    ious = []

    for c in range(4):
        p = (pred_mask_4ch[c] > 0).astype(np.float32)
        g = (gt_mask_4ch[c] > 0).astype(np.float32)

        intersection = np.sum(p * g)
        total = np.sum(p) + np.sum(g)
        union = np.sum((p + g) > 0)

        if np.sum(g) == 0 and np.sum(p) == 0:
            dice = 1.0
            iou = 1.0
        else:
            dice = (2.0 * intersection + smooth) / (total + smooth)
            iou = (intersection + smooth) / (union + smooth)

        metrics[f"dice_class_{c+1}"] = float(dice)
        metrics[f"iou_class_{c+1}"] = float(iou)
        dices.append(dice)
        ious.append(iou)

    metrics["mean_dice"] = float(np.mean(dices))
    metrics["mean_iou"] = float(np.mean(ious))
    return metrics


def evaluate_pipeline(
    img_dir: str = "data/severstal/train_images",
    csv_path: str = "data/severstal/train.csv",
    index_path: str = "data/severstal_ivfpq.index",
    labels_path: str = "data/severstal_labels.npy",
    output_dir: str = "results",
    num_samples: int = 20,
    k: int = 5,
) -> Dict[str, Union[float, List[Dict]]]:
    """
    Runs full evaluation on a subset of annotated images and produces metric summaries,
    structured JSON/CSV reports, and high-resolution visual plots.
    """
    import json
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    inspector = DefectInspector(index_path=index_path, labels_path=labels_path, k=k)

    dataset = SeverstalDataset(img_dir=img_dir, csv_path=csv_path)
    print(f"Total dataset images: {len(dataset)}. Evaluating {num_samples} sample images...")

    # Select both defect and normal samples for balanced evaluation
    defect_indices = [i for i, img_id in enumerate(dataset.image_ids) if dataset.image_to_rle.get(img_id, {})]
    normal_indices = [i for i, img_id in enumerate(dataset.image_ids) if not dataset.image_to_rle.get(img_id, {})]

    selected_indices = []
    if defect_indices:
        selected_indices.extend(defect_indices[: min(len(defect_indices), num_samples * 3 // 4)])
    if normal_indices:
        remaining = num_samples - len(selected_indices)
        selected_indices.extend(normal_indices[: min(len(normal_indices), remaining)])

    per_image_results = []
    total_pixels_correct = 0
    total_pixels_count = 0
    total_patches_correct = 0
    total_patches_count = 0
    correct_binary_image_preds = 0

    for i, idx in enumerate(tqdm(selected_indices, desc="Evaluating Inspection Samples")):
        sample = dataset[idx]
        image_id = sample["image_id"]
        image_path = os.path.join(img_dir, image_id)
        gt_mask_4ch = sample["full_mask"].numpy()  # [4, 256, 1600]

        # Ground truth full class map
        gt_class_map = np.zeros((256, 1600), dtype=np.uint8)
        for c in range(1, 5):
            gt_class_map[gt_mask_4ch[c - 1] > 0] = c

        # Ground truth patch grid
        gt_patch_grid = np.zeros((16, 16), dtype=np.uint8)
        for c in range(1, 5):
            gt_patch_grid[sample["mask_16x16"][c - 1].numpy() > 0] = c

        output_plot_path = os.path.join(output_dir, f"inspection_{i+1:02d}_{image_id.split('.')[0]}.png")
        res = inspector.evaluate_image(
            image_path_or_tensor=image_path if os.path.exists(image_path) else sample["image"],
            ground_truth_mask=gt_mask_4ch,
            output_path=output_plot_path,
            image_id=image_id,
        )

        pred_class_map = res["predicted_class_map"]
        patch_preds = res["patch_preds_16x16"]

        # Pixel accuracy
        pix_corr = int(np.sum(pred_class_map == gt_class_map))
        pix_tot = gt_class_map.size
        total_pixels_correct += pix_corr
        total_pixels_count += pix_tot
        pix_acc = pix_corr / pix_tot

        # Patch accuracy
        patch_corr = int(np.sum(patch_preds == gt_patch_grid))
        patch_tot = gt_patch_grid.size
        total_patches_correct += patch_corr
        total_patches_count += patch_tot
        patch_acc = patch_corr / patch_tot

        # Image-level binary detection
        gt_has_defect = bool(sample["has_defect"])
        pred_has_defect = bool(np.any(pred_class_map > 0))
        if gt_has_defect == pred_has_defect:
            correct_binary_image_preds += 1

        img_record = {
            "image_id": image_id,
            "has_defect_gt": gt_has_defect,
            "has_defect_pred": pred_has_defect,
            "pixel_accuracy": float(pix_acc),
            "patch_accuracy": float(patch_acc),
            "mean_dice": float(res["metrics"]["mean_dice"]),
            "mean_iou": float(res["metrics"]["mean_iou"]),
            "dice_class_1": float(res["metrics"]["dice_class_1"]),
            "dice_class_2": float(res["metrics"]["dice_class_2"]),
            "dice_class_3": float(res["metrics"]["dice_class_3"]),
            "dice_class_4": float(res["metrics"]["dice_class_4"]),
            "iou_class_1": float(res["metrics"]["iou_class_1"]),
            "iou_class_2": float(res["metrics"]["iou_class_2"]),
            "iou_class_3": float(res["metrics"]["iou_class_3"]),
            "iou_class_4": float(res["metrics"]["iou_class_4"]),
            "plot_path": output_plot_path,
        }
        per_image_results.append(img_record)

    # Aggregate summaries
    overall_pixel_acc = total_pixels_correct / max(1, total_pixels_count)
    overall_patch_acc = total_patches_correct / max(1, total_patches_count)
    overall_binary_acc = correct_binary_image_preds / max(1, len(selected_indices))
    avg_dice = float(np.mean([m["mean_dice"] for m in per_image_results]))
    avg_iou = float(np.mean([m["mean_iou"] for m in per_image_results]))

    summary = {
        "num_evaluated_images": len(selected_indices),
        "image_level_binary_accuracy": float(overall_binary_acc),
        "patch_level_accuracy": float(overall_patch_acc),
        "pixel_level_accuracy": float(overall_pixel_acc),
        "average_mean_dice": float(avg_dice),
        "average_mean_iou": float(avg_iou),
        "per_class_metrics": {
            f"class_{c}": {
                "name": CLASS_NAMES[c],
                "avg_dice": float(np.mean([m[f"dice_class_{c}"] for m in per_image_results])),
                "avg_iou": float(np.mean([m[f"iou_class_{c}"] for m in per_image_results])),
            }
            for c in range(1, 5)
        },
        "per_image_details": per_image_results,
    }

    # Save to JSON
    json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved quantitative JSON summary to: {json_path}")

    # Save to CSV
    csv_out_path = os.path.join(output_dir, "metrics_summary.csv")
    df_metrics = pd.DataFrame(per_image_results)
    df_metrics.to_csv(csv_out_path, index=False)
    print(f"Saved quantitative CSV table to: {csv_out_path}")

    # Print Console Summary
    print("\n" + "=" * 60)
    print("QUANTITATIVE EVALUATION SUMMARY:")
    print(f"  Evaluated Samples:                {len(selected_indices)}")
    print(f"  Image-Level Defect Accuracy:      {overall_binary_acc * 100:.2f}%")
    print(f"  Patch-Level Grid Accuracy:        {overall_patch_acc * 100:.2f}%")
    print(f"  Pixel-Level Exact Accuracy:       {overall_pixel_acc * 100:.2f}%")
    print(f"  Mean Dice Score (All Classes):    {avg_dice:.4f}")
    print(f"  Mean IoU Score (All Classes):     {avg_iou:.4f}")
    for c in range(1, 5):
        c_dice = summary["per_class_metrics"][f"class_{c}"]["avg_dice"]
        c_iou = summary["per_class_metrics"][f"class_{c}"]["avg_iou"]
        print(f"  {CLASS_NAMES[c]}: Dice={c_dice:.4f}, IoU={c_iou:.4f}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Severstal Steel Defect Detection Pipeline.")
    parser.add_argument("--img_dir", type=str, default="data/severstal/train_images", help="Path to train_images directory")
    parser.add_argument("--csv_path", type=str, default="data/severstal/train.csv", help="Path to train.csv")
    parser.add_argument("--index_path", type=str, default="data/severstal_ivfpq.index", help="Path to FAISS index")
    parser.add_argument("--labels_path", type=str, default="data/severstal_labels.npy", help="Path to labels array")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output visualizations")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of sample images to evaluate")
    parser.add_argument("--k", type=int, default=5, help="k-NN neighbors for majority voting")

    args = parser.parse_args()
    evaluate_pipeline(
        img_dir=args.img_dir,
        csv_path=args.csv_path,
        index_path=args.index_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        k=args.k,
    )


if __name__ == "__main__":
    main()
