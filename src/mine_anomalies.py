"""
The Data Engine: Zero-Shot FAISS IVFPQ Anomaly Mining Module.

Simulates a Day-1 factory deployment active learning / Human-in-the-Loop (HITL) workflow:
1. Builds a quantized FAISS IVFPQ normal memory bank using pure normal (defect-free) steel strips.
2. Mines an unannotated stream of steel strips by querying patch-level distance to normal clusters.
3. Flags candidate defective images exceeding an anomaly distance threshold for human expert review.
4. Saves flagged images to data/flagged_for_human_review.csv.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.build_index import extract_patch_embeddings, load_dinov2_model
from src.dataset import DINOV2_MEAN, DINOV2_STD


class SimpleImageDataset(Dataset):
    """Simple image loader for anomaly mining."""

    def __init__(self, img_dir: str, image_ids: List[str], image_size: Tuple[int, int] = (224, 224)):
        self.img_dir = img_dir
        self.image_ids = image_ids
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.img_dir, image_id)
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
        else:
            img = Image.new("RGB", (1600, 256), color=(128, 128, 128))
        return self.transform(img), image_id


def build_normal_memory_bank(
    model: nn.Module,
    img_dir: str,
    normal_image_ids: List[str],
    device: torch.device,
    batch_size: int = 16,
    nlist: int = 100,
    m: int = 16,
    nbits: int = 8,
) -> faiss.IndexIVFPQ:
    """
    Extracts patch features from pure normal images and trains a quantized FAISS IVFPQ index.
    """
    print(f"\n[Data Engine] Building Normal Memory Bank from {len(normal_image_ids)} normal images...")
    dataset = SimpleImageDataset(img_dir, normal_image_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_normal_patches = []
    for images, _ in tqdm(loader, desc="Extracting Normal Patches"):
        patch_embeds = extract_patch_embeddings(model, images, device).numpy()  # [B, 256, 768]
        batch_b = patch_embeds.shape[0]
        for b in range(batch_b):
            all_normal_patches.append(patch_embeds[b])

    embeddings_np = np.ascontiguousarray(np.vstack(all_normal_patches), dtype=np.float32)
    num_patches, dim = embeddings_np.shape
    print(f"[Data Engine] Collected {num_patches} normal reference patches (dim: {dim}).")

    actual_nlist = min(nlist, max(1, num_patches // 39))
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, actual_nlist, m, nbits)

    print("[Data Engine] Training and populating normal IVFPQ index...")
    index.train(embeddings_np)
    index.add(embeddings_np)
    if hasattr(index, "nprobe"):
        index.nprobe = min(16, actual_nlist)

    print(f"[Data Engine] Normal Memory Bank ready with {index.ntotal} quantized patches.")
    return index


def mine_candidate_defects(
    model: nn.Module,
    normal_index: faiss.IndexIVFPQ,
    img_dir: str,
    unlabeled_image_ids: List[str],
    device: torch.device,
    threshold: float = 120.0,
    batch_size: int = 16,
    ground_truth_lookup: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """
    Queries unlabeled images against the normal memory bank and flags candidate anomalies.
    """
    print(f"\n[Data Engine] Mining anomalies across {len(unlabeled_image_ids)} unannotated images (Threshold: {threshold})...")
    dataset = SimpleImageDataset(img_dir, unlabeled_image_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    records = []

    for images, img_ids in tqdm(loader, desc="Mining Candidate Defects"):
        patch_embeds = extract_patch_embeddings(model, images, device).numpy()  # [B, 256, 768]
        batch_b = patch_embeds.shape[0]

        for b in range(batch_b):
            image_id = img_ids[b]
            img_patches = np.ascontiguousarray(patch_embeds[b], dtype=np.float32)  # [256, 768]

            # Query 1-NN L2 distance to closest normal patch
            distances, _ = normal_index.search(img_patches, k=1)
            dist_1d = distances.flatten()

            max_dist = float(np.max(dist_1d))
            mean_dist = float(np.mean(dist_1d))
            p95_dist = float(np.percentile(dist_1d, 95))

            # Flag if max anomaly distance exceeds threshold
            is_flagged = bool(max_dist >= threshold)

            record = {
                "ImageId": image_id,
                "MaxAnomalyDistance": max_dist,
                "MeanAnomalyDistance": mean_dist,
                "P95AnomalyDistance": p95_dist,
                "FlaggedForReview": is_flagged,
            }

            if ground_truth_lookup is not None:
                record["ActualGroundTruthHasDefect"] = ground_truth_lookup.get(image_id, False)

            records.append(record)

    df_results = pd.DataFrame(records)
    return df_results


def run_data_engine(
    img_dir: str = "data/severstal/train_images",
    csv_path: str = "data/severstal/train.csv",
    output_csv: str = "data/flagged_for_human_review.csv",
    num_normal: int = 200,
    max_unlabeled: Optional[int] = None,
    threshold: float = 120.0,
    batch_size: int = 16,
    device_str: Optional[str] = None,
) -> pd.DataFrame:
    """
    Executes the full Human-in-the-Loop active learning mining simulation.
    """
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Data Engine] Initializing on device: {device}")
    model = load_dinov2_model(device)

    # 1. Parse train.csv to identify known defect annotations
    print(f"[Data Engine] Parsing {csv_path}...")
    df = pd.read_csv(csv_path)
    if "ImageId_ClassId" in df.columns:
        df["ImageId"] = df["ImageId_ClassId"].str.rsplit("_", n=1, expand=True)[0]

    defect_images_set = set(df[df["EncodedPixels"].notna() & (df["EncodedPixels"].str.strip() != "")]["ImageId"].unique())

    all_available_images = [
        f for f in sorted(os.listdir(img_dir)) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    normal_images = [img for img in all_available_images if img not in defect_images_set]
    defective_images = [img for img in all_available_images if img in defect_images_set]

    print(f"[Data Engine] Total Dataset Images: {len(all_available_images)} ({len(normal_images)} Normal, {len(defective_images)} Defective)")

    # 2. Select normal subset for training FAISS memory bank
    np.random.seed(42)
    selected_normal = list(np.random.choice(normal_images, size=min(num_normal, len(normal_images)), replace=False))
    remaining_normal = [img for img in normal_images if img not in set(selected_normal)]

    # 3. Build Normal FAISS Memory Bank
    normal_index = build_normal_memory_bank(
        model=model,
        img_dir=img_dir,
        normal_image_ids=selected_normal,
        device=device,
        batch_size=batch_size,
    )

    # 4. Construct Unlabeled Stream
    unlabeled_stream = defective_images + remaining_normal
    np.random.seed(123)
    np.random.shuffle(unlabeled_stream)

    if max_unlabeled is not None:
        unlabeled_stream = unlabeled_stream[:max_unlabeled]

    gt_lookup = {img: (img in defect_images_set) for img in unlabeled_stream}

    # 5. Mining Loop
    df_mined = mine_candidate_defects(
        model=model,
        normal_index=normal_index,
        img_dir=img_dir,
        unlabeled_image_ids=unlabeled_stream,
        device=device,
        threshold=threshold,
        batch_size=batch_size,
        ground_truth_lookup=gt_lookup,
    )

    # 6. Save flagged CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_mined.to_csv(output_csv, index=False)
    print(f"\n[Data Engine] Successfully saved flagged candidates to: {output_csv}")

    # Summary Statistics
    flagged_total = df_mined["FlaggedForReview"].sum()
    print("\n" + "=" * 60)
    print("DATA ENGINE MINING RESULTS (HITL SIMULATION):")
    print(f"  Total Unlabeled Stream Evaluated: {len(df_mined)}")
    print(f"  Flagged for Human Review:         {flagged_total} ({flagged_total / len(df_mined) * 100:.1f}%)")
    if "ActualGroundTruthHasDefect" in df_mined.columns:
        true_positives = len(df_mined[df_mined["FlaggedForReview"] & df_mined["ActualGroundTruthHasDefect"]])
        actual_defects = len(df_mined[df_mined["ActualGroundTruthHasDefect"]])
        precision = true_positives / max(1, flagged_total)
        recall = true_positives / max(1, actual_defects)
        print(f"  Actual Defective in Stream:       {actual_defects}")
        print(f"  Defects Successfully Captured:    {true_positives} / {actual_defects}")
        print(f"  Mining Candidate Precision:       {precision * 100:.1f}%")
        print(f"  Mining Defect Capture Recall:     {recall * 100:.1f}%")
    print("=" * 60)

    return df_mined


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot FAISS IVFPQ Anomaly Mining Data Engine.")
    parser.add_argument("--img_dir", type=str, default="data/severstal/train_images", help="Path to images directory")
    parser.add_argument("--csv_path", type=str, default="data/severstal/train.csv", help="Path to train.csv")
    parser.add_argument("--output_csv", type=str, default="data/flagged_for_human_review.csv", help="Path to save flagged CSV")
    parser.add_argument("--num_normal", type=int, default=200, help="Number of normal images to build memory bank")
    parser.add_argument("--max_unlabeled", type=int, default=500, help="Number of unannotated images to mine (or None for all)")
    parser.add_argument("--threshold", type=float, default=120.0, help="Distance threshold to flag defective candidate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")

    args = parser.parse_args()

    run_data_engine(
        img_dir=args.img_dir,
        csv_path=args.csv_path,
        output_csv=args.output_csv,
        num_normal=args.num_normal,
        max_unlabeled=args.max_unlabeled,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
