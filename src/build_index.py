"""
Module to extract DINOv2 patch features and construct a quantized FAISS IVFPQ memory bank.

This script extracts patch embeddings (256 tokens per 224x224 image) using a frozen
DINOv2 ViT-B/14 backbone and indexes them with FAISS IndexIVFPQ for fast, ultra-compact
nearest-neighbor anomaly segmentation.
"""

import argparse
import os
import sys
from typing import Optional, Tuple
import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SeverstalDataset


def load_dinov2_model(device: torch.device) -> nn.Module:
    """
    Load frozen DINOv2 ViT-B/14 model.

    Tries PyTorch Hub first, falling back to HuggingFace Transformers if needed.

    Args:
        device: torch device (cuda or cpu).

    Returns:
        model: Frozen PyTorch model in eval mode.
    """
    print(f"Loading DINOv2 ViT-B/14 on {device}...")
    try:
        # Load from PyTorch Hub
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("Successfully loaded DINOv2 from torch.hub.")
        return model
    except Exception as hub_err:
        print(f"torch.hub loading encountered: {hub_err}. Falling back to HuggingFace transformers...")
        try:
            from transformers import AutoModel

            hf_model = AutoModel.from_pretrained("facebook/dinov2-base")
            hf_model.to(device)
            hf_model.eval()
            for param in hf_model.parameters():
                param.requires_grad = False
            print("Successfully loaded DINOv2 from transformers.")
            return hf_model
        except Exception as hf_err:
            raise RuntimeError(f"Failed to load DINOv2 from both torch.hub and transformers: {hf_err}")


def extract_patch_embeddings(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract [B, 256, 768] patch embeddings from a batch of [B, 3, 224, 224] images.

    Args:
        model: DINOv2 model.
        images: Batch of preprocessed images [B, 3, 224, 224].
        device: Execution device.

    Returns:
        patch_embeddings: Tensor of shape [B, 256, 768].
    """
    images = images.to(device)
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            features_dict = model.forward_features(images)
            # DINOv2 torch hub returns 'x_norm_patchtokens' of shape [B, 256, 768]
            if "x_norm_patchtokens" in features_dict:
                return features_dict["x_norm_patchtokens"].cpu()
            elif "x_prenorm" in features_dict:
                return features_dict["x_prenorm"][:, 1:, :].cpu()
            else:
                tokens = model.get_intermediate_layers(images, n=1, return_class_token=False)[0]
                return tokens.cpu()
        elif hasattr(model, "get_intermediate_layers"):
            tokens = model.get_intermediate_layers(images, n=1, return_class_token=False)[0]
            return tokens.cpu()
        else:
            # Hugging Face transformers DINOv2 model
            outputs = model(pixel_values=images)
            # outputs.last_hidden_state is [B, 257, 768] (index 0 is CLS token)
            return outputs.last_hidden_state[:, 1:, :].cpu()


def build_faiss_ivfpq_index(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_index_path: str,
    output_labels_path: str,
    nlist: int = 100,
    m: int = 16,
    nbits: int = 8,
) -> Tuple[faiss.IndexIVFPQ, np.ndarray]:
    """
    Build, train, and save a compressed FAISS IndexIVFPQ memory bank.

    Args:
        embeddings: Float32 array of shape [N, 768].
        labels: Int64/Int32 array of shape [N] with values 0 (normal) or 1..4 (defects).
        output_index_path: Target path to save .index file.
        output_labels_path: Target path to save .npy labels file.
        nlist: Number of Voronoi cells / coarse quantizer clusters.
        m: Number of sub-vector quantizers (768 must be divisible by m).
        nbits: Bits per sub-vector quantization code (8 bits = 256 centroids per sub-quantizer).

    Returns:
        index: Trained and populated FAISS IndexIVFPQ.
        labels: Saved labels array.
    """
    num_samples, dim = embeddings.shape
    print(f"Building FAISS IndexIVFPQ for {num_samples} patch embeddings (dimension: {dim})...")

    # Adapt nlist if sample size is small for training stability
    actual_nlist = min(nlist, max(1, num_samples // 39))
    if actual_nlist != nlist:
        print(f"Adjusted nlist from {nlist} to {actual_nlist} based on sample size ({num_samples}).")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, actual_nlist, m, nbits)

    print("Training FAISS IVFPQ index...")
    index.train(embeddings)

    print("Adding embeddings to index...")
    index.add(embeddings)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_index_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_labels_path)), exist_ok=True)

    print(f"Saving FAISS index to {output_index_path}...")
    faiss.write_index(index, output_index_path)

    print(f"Saving labels to {output_labels_path}...")
    np.save(output_labels_path, labels)

    index_size_mb = os.path.getsize(output_index_path) / (1024 * 1024)
    raw_size_mb = (embeddings.nbytes) / (1024 * 1024)
    print(f"Done! Raw memory: {raw_size_mb:.2f} MB -> Quantized Index: {index_size_mb:.2f} MB")
    print(f"Compression ratio: {raw_size_mb / max(0.001, index_size_mb):.1f}x")

    return index, labels


def extract_and_index_dataset(
    img_dir: str,
    csv_path: str,
    output_index_path: str,
    output_labels_path: str,
    batch_size: int = 16,
    subsample_normal_rate: float = 0.05,
    max_images: Optional[int] = None,
    subset_fraction: Optional[float] = None,
    nlist: int = 100,
    m: int = 16,
    nbits: int = 8,
    device_str: Optional[str] = None,
) -> Tuple[faiss.IndexIVFPQ, np.ndarray]:
    """
    Iterates over the dataset, extracts DINOv2 patch features, and builds the IVFPQ index.
    """
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    model = load_dinov2_model(device)

    dataset = SeverstalDataset(
        img_dir=img_dir,
        csv_path=csv_path,
        subset_fraction=subset_fraction,
    )

    if max_images is not None and max_images < len(dataset):
        dataset.image_ids = dataset.image_ids[:max_images]

    print(f"Total images to process for memory bank: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 0 for safe Windows multiprocessing
    )

    all_embeddings_list = []
    all_labels_list = []

    np.random.seed(42)

    for batch in tqdm(loader, desc="Extracting DINOv2 Patch Features"):
        images = batch["image"]  # [B, 3, 224, 224]
        masks_16x16 = batch["mask_16x16"]  # [B, 4, 16, 16]

        # Extract [B, 256, 768]
        patch_embeds = extract_patch_embeddings(model, images, device).numpy()  # [B, 256, 768]
        batch_b = patch_embeds.shape[0]

        # Reshape masks to [B, 4, 256]
        masks_flat = masks_16x16.view(batch_b, 4, 256).numpy()

        for b in range(batch_b):
            for patch_idx in range(256):
                patch_mask = masks_flat[b, :, patch_idx]  # [4]
                # Check for defect classes 1, 2, 3, 4
                positive_classes = np.where(patch_mask > 0.5)[0]

                if len(positive_classes) > 0:
                    # Defect patch: assign label (1-4)
                    label = int(positive_classes[0] + 1)
                    all_embeddings_list.append(patch_embeds[b, patch_idx])
                    all_labels_list.append(label)
                else:
                    # Normal patch: subsample at specified rate
                    if np.random.rand() < subsample_normal_rate:
                        all_embeddings_list.append(patch_embeds[b, patch_idx])
                        all_labels_list.append(0)

    embeddings_np = np.ascontiguousarray(np.vstack(all_embeddings_list), dtype=np.float32)
    labels_np = np.asarray(all_labels_list, dtype=np.int64)

    # Class breakdown
    unique, counts = np.unique(labels_np, return_counts=True)
    print("\nPatch Memory Bank Class Distribution:")
    class_names = {0: "Normal (Class 0)", 1: "Defect Class 1", 2: "Defect Class 2", 3: "Defect Class 3", 4: "Defect Class 4"}
    for u, c in zip(unique, counts):
        print(f"  {class_names.get(u, f'Class {u}')}: {c} patches ({c / len(labels_np) * 100:.1f}%)")

    index, saved_labels = build_faiss_ivfpq_index(
        embeddings=embeddings_np,
        labels=labels_np,
        output_index_path=output_index_path,
        output_labels_path=output_labels_path,
        nlist=nlist,
        m=m,
        nbits=nbits,
    )

    return index, saved_labels


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 patch features and build FAISS IVFPQ index.")
    parser.add_argument("--img_dir", type=str, default="data/severstal/train_images", help="Path to train images directory")
    parser.add_argument("--csv_path", type=str, default="data/severstal/train.csv", help="Path to train.csv")
    parser.add_argument("--output_index", type=str, default="data/severstal_ivfpq.index", help="Output path for FAISS index")
    parser.add_argument("--output_labels", type=str, default="data/severstal_labels.npy", help="Output path for labels array")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--subsample_normal", type=float, default=0.05, help="Subsampling fraction for normal patches (0.0 to 1.0)")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to index")
    parser.add_argument("--subset_fraction", type=float, default=None, help="Fraction of dataset to use")
    parser.add_argument("--nlist", type=int, default=100, help="FAISS IVFPQ nlist (number of centroids)")
    parser.add_argument("--m", type=int, default=16, help="FAISS IVFPQ sub-vector quantizers")
    parser.add_argument("--nbits", type=int, default=8, help="FAISS IVFPQ bits per sub-vector")

    args = parser.parse_args()

    extract_and_index_dataset(
        img_dir=args.img_dir,
        csv_path=args.csv_path,
        output_index_path=args.output_index,
        output_labels_path=args.output_labels,
        batch_size=args.batch_size,
        subsample_normal_rate=args.subsample_normal,
        max_images=args.max_images,
        subset_fraction=args.subset_fraction,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
    )


if __name__ == "__main__":
    main()
