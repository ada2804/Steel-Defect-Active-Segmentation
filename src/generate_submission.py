"""
Kaggle Submission Generator for Severstal Steel Defect Detection.

Runs the supervised DINOv2 + U-Net Decoder model on the official test set
(data/severstal/test_images/), converts multi-class predictions into Kaggle-compliant
RLE strings, and exports submission.csv.
"""

import argparse
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.dataset import DINOV2_MEAN, DINOV2_STD
from src.model import DinoUNetDecoder
from src.rle_utils import mask_to_rle


class TestImageDataset(Dataset):
    """Dataset loader for unlabelled test images."""

    def __init__(self, test_dir: str, image_size: Tuple[int, int] = (224, 224), max_images: Optional[int] = None):
        self.test_dir = test_dir
        self.image_size = image_size
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_ids = [
            f for f in sorted(os.listdir(test_dir)) if f.lower().endswith(valid_exts)
        ]
        if max_images is not None and max_images < len(self.image_ids):
            self.image_ids = self.image_ids[:max_images]

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.test_dir, image_id)
        pil_img = Image.open(image_path).convert("RGB")
        return self.transform(pil_img), image_id


def generate_kaggle_submission(
    test_dir: str = "data/severstal/test_images",
    model_weights_path: str = "results/best_unet_decoder.pth",
    output_csv_path: str = "submission.csv",
    batch_size: int = 16,
    threshold: float = 0.5,
    min_pixels_threshold: int = 10,
    max_images: Optional[int] = None,
    device_str: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates Kaggle-compliant submission.csv using trained DINOv2-UNet model.
    """
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Kaggle Evaluator] Running on device: {device}")
    print(f"[Kaggle Evaluator] Loading model architecture...")
    model = DinoUNetDecoder(num_classes=4, device=device)

    if os.path.exists(model_weights_path):
        print(f"[Kaggle Evaluator] Loading trained weights from {model_weights_path}...")
        checkpoint = torch.load(model_weights_path, map_location=device, weights_only=False)
        if "decoder_state_dict" in checkpoint:
            model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        elif isinstance(checkpoint, dict):
            model.decoder.load_state_dict(checkpoint)
        print("Successfully loaded trained decoder weights.")
    else:
        print(f"[Kaggle Evaluator] WARNING: Checkpoint {model_weights_path} not found. Running with initialized decoder.")

    model.eval()

    print(f"[Kaggle Evaluator] Preparing test dataset from {test_dir}...")
    test_dataset = TestImageDataset(test_dir=test_dir, max_images=max_images)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[Kaggle Evaluator] Total test images to process: {len(test_dataset)}")

    submission_rows = []
    total_defects_found = 0

    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Generating Kaggle Predictions"):
            images = images.to(device)  # [B, 3, 224, 224]

            # Forward pass -> [B, 4, 224, 224]
            logits = model(images)
            probs = torch.sigmoid(logits)  # [B, 4, 224, 224]
            binary_masks = (probs >= threshold).float().cpu().numpy()  # [B, 4, 224, 224]

            batch_size_curr = images.size(0)

            for b in range(batch_size_curr):
                img_id = image_ids[b]

                for class_idx in range(4):
                    class_id = class_idx + 1
                    key = f"{img_id}_{class_id}"

                    mask_224 = binary_masks[b, class_idx]  # [224, 224]

                    # Resize prediction back to native resolution [256, 1600]
                    mask_native = cv2.resize(
                        mask_224.astype(np.uint8),
                        (1600, 256),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    # Post-processing: minimum pixel threshold to filter spurious 1-pixel noise
                    if np.sum(mask_native) >= min_pixels_threshold:
                        rle_str = mask_to_rle(mask_native)
                        total_defects_found += 1
                    else:
                        rle_str = ""

                    submission_rows.append({"ImageId_ClassId": key, "EncodedPixels": rle_str})

    df_sub = pd.DataFrame(submission_rows)

    # Save primary submission.csv
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or ".", exist_ok=True)
    df_sub.to_csv(output_csv_path, index=False)
    print(f"\n[Kaggle Evaluator] Saved final Kaggle submission to: {output_csv_path}")

    # Also save a copy to results/submission.csv
    results_sub_path = os.path.join("results", "submission.csv")
    os.makedirs("results", exist_ok=True)
    df_sub.to_csv(results_sub_path, index=False)
    print(f"[Kaggle Evaluator] Saved duplicate copy to: {results_sub_path}")

    # Summary
    print("\n" + "=" * 60)
    print("KAGGLE SUBMISSION SUMMARY:")
    print(f"  Total Submission Rows:       {len(df_sub)}")
    print(f"  Total Unique Images:         {len(df_sub) // 4}")
    print(f"  Non-Empty Defect Preds:      {total_defects_found} ({total_defects_found / len(df_sub) * 100:.2f}%)")
    print(f"  Submission Head:\n{df_sub.head(8)}")
    print("=" * 60)

    return df_sub


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission.csv using DinoUNetDecoder.")
    parser.add_argument("--test_dir", type=str, default="data/severstal/test_images", help="Path to test images directory")
    parser.add_argument(
        "--model_path", "--weights",
        type=str,
        default="results/best_unet_decoder.pth",
        help="Path to trained decoder checkpoint (.pth)",
    )
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Path to output submission.csv")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for test inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    parser.add_argument("--min_pixels", type=int, default=10, help="Minimum positive pixels to retain defect prediction")
    parser.add_argument("--max_images", type=int, default=None, help="Max test images to process (or None for all 5506)")

    args = parser.parse_args()

    generate_kaggle_submission(
        test_dir=args.test_dir,
        model_weights_path=args.model_path,
        output_csv_path=args.output_csv,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_pixels_threshold=args.min_pixels,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
