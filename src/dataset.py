"""
Severstal Steel Defect PyTorch Dataset Module.

Handles image loading, multi-class RLE mask decoding, DINOv2 preprocessing,
and 16x16 patch-level mask grid generation for ViT-B/14 (224x224 input).
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

from src.rle_utils import rle_to_mask


# Standard ImageNet normalization used by DINOv2
DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD = [0.229, 0.224, 0.225]


class SeverstalDataset(Dataset):
    """
    PyTorch Dataset for Severstal Steel Defect Detection.

    Provides multi-class segmentation masks, 224x224 image resizing, and
    16x16 patch-level ground truth masks mapped to DINOv2 ViT-B/14 patch grid.
    """

    def __init__(
        self,
        img_dir: str,
        csv_path: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224),
        patch_grid_size: Tuple[int, int] = (16, 16),
        transform: Optional[transforms.Compose] = None,
        is_test: bool = False,
        subset_fraction: Optional[float] = None,
        random_seed: int = 42,
    ):
        """
        Initialize the Severstal Dataset.

        Args:
            img_dir: Path to directory containing images (e.g. data/severstal/train_images).
            csv_path: Path to train.csv containing ImageId, ClassId, and EncodedPixels.
            image_size: Target image dimensions (height, width) for DINOv2 (default: 224x224).
            patch_grid_size: Patch grid resolution (height, width) (default: 16x16).
            transform: Optional custom torchvision transform. If None, default DINOv2 transform is used.
            is_test: Set True for unlabelled test sets without annotations.
            subset_fraction: Optional float (0.0 < fraction <= 1.0) to subsample the dataset.
            random_seed: Random seed for reproducible subsampling.
        """
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.image_size = image_size
        self.patch_grid_size = patch_grid_size
        self.is_test = is_test

        # Set up image transformations
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
                ]
            )

        # Build annotations lookup
        self.image_to_rle: Dict[str, Dict[int, str]] = {}
        self.image_ids: List[str] = []

        if not self.is_test and self.csv_path and os.path.exists(self.csv_path):
            self._load_annotations()
        elif os.path.exists(self.img_dir):
            # If no CSV or test mode, load all images from directory
            valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
            self.image_ids = [
                f for f in sorted(os.listdir(self.img_dir)) if f.lower().endswith(valid_exts)
            ]
        else:
            self.image_ids = []

        # Optional subsampling
        if subset_fraction is not None and 0.0 < subset_fraction < 1.0:
            np.random.seed(random_seed)
            sample_size = max(1, int(len(self.image_ids) * subset_fraction))
            self.image_ids = list(np.random.choice(self.image_ids, size=sample_size, replace=False))

    def _load_annotations(self) -> None:
        """Parse train.csv supporting both legacy and modern Severstal CSV schemas."""
        df = pd.read_csv(self.csv_path)

        # Handle legacy schema: ImageId_ClassId, EncodedPixels
        if "ImageId_ClassId" in df.columns:
            split_data = df["ImageId_ClassId"].str.rsplit("_", n=1, expand=True)
            df["ImageId"] = split_data[0]
            df["ClassId"] = split_data[1].astype(int)

        # Ensure types
        df["ImageId"] = df["ImageId"].astype(str)
        df["ClassId"] = df["ClassId"].astype(int)

        # Group by ImageId
        grouped = df.groupby("ImageId")
        for image_id, group in grouped:
            class_dict: Dict[int, str] = {}
            for _, row in group.iterrows():
                class_id = int(row["ClassId"])
                rle = row["EncodedPixels"]
                if pd.notna(rle) and str(rle).strip():
                    class_dict[class_id] = str(rle).strip()
            self.image_to_rle[image_id] = class_dict

        # Filter image IDs to those that exist in the directory if directory is available
        if os.path.exists(self.img_dir):
            available_files = set(os.listdir(self.img_dir))
            self.image_ids = [img_id for img_id in self.image_to_rle.keys() if img_id in available_files]
            # Also include any images in folder not in CSV if needed
            if not self.image_ids:
                self.image_ids = sorted(list(self.image_to_rle.keys()))
        else:
            self.image_ids = sorted(list(self.image_to_rle.keys()))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """
        Load image and compute corresponding 4-channel mask and 16x16 patch grid mask.

        Returns:
            Dict containing:
                - 'image': Tensor of shape [3, 224, 224] (normalized)
                - 'mask_16x16': Tensor of shape [4, 16, 16] (binary defect indicators)
                - 'full_mask': Tensor of shape [4, 256, 1600] (original resolution masks)
                - 'image_id': String filename of the image
                - 'has_defect': Boolean indicating if any defect is present
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.img_dir, image_id)

        # Load image via PIL and convert to RGB
        if os.path.exists(image_path):
            pil_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = pil_img.size
        else:
            # Fallback placeholder if image file missing during testing
            orig_w, orig_h = 1600, 256
            pil_img = Image.new("RGB", (orig_w, orig_h), color=(128, 128, 128))

        # Apply DINOv2 transformation
        img_tensor = self.transform(pil_img)

        # Build 4-channel full-resolution mask [4, 256, 1600]
        # Class indices: 1, 2, 3, 4 -> Channel indices: 0, 1, 2, 3
        full_mask_np = np.zeros((4, orig_h, orig_w), dtype=np.uint8)
        rle_dict = self.image_to_rle.get(image_id, {})
        has_defect = False

        for class_id in range(1, 5):
            if class_id in rle_dict:
                rle_str = rle_dict[class_id]
                cls_mask = rle_to_mask(rle_str, width=orig_w, height=orig_h)
                full_mask_np[class_id - 1] = cls_mask
                if np.any(cls_mask):
                    has_defect = True

        full_mask_tensor = torch.from_numpy(full_mask_np).float()  # [4, H, W]

        # Downsample full mask to patch_grid_size (16x16)
        # We first resize the mask to image_size (224x224) using nearest neighbor,
        # then apply max_pooling with kernel_size=(14, 14) so that any defect pixel
        # inside a 14x14 patch marks that patch as defective.
        mask_4d = full_mask_tensor.unsqueeze(0)  # [1, 4, H, W]
        mask_224 = F.interpolate(mask_4d, size=self.image_size, mode="nearest")  # [1, 4, 224, 224]
        
        # 224 / 16 = 14 pixel patch size
        patch_h = self.image_size[0] // self.patch_grid_size[0]
        patch_w = self.image_size[1] // self.patch_grid_size[1]
        
        mask_16x16 = F.max_pool2d(mask_224, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        mask_16x16 = (mask_16x16.squeeze(0) > 0.5).float()  # [4, 16, 16]

        return {
            "image": img_tensor,
            "mask_16x16": mask_16x16,
            "full_mask": full_mask_tensor,
            "image_id": image_id,
            "has_defect": has_defect,
        }


class SeverstalUNetDataset(Dataset):
    """
    Supervised PyTorch Dataset for DINOv2-UNet Decoder training and evaluation.

    Loads 1600x256 images, resizes both images and 4-class multi-label ground truth
    masks to 224x224 (nearest-neighbor for masks to preserve crisp defect boundaries),
    and normalizes images for DINOv2.
    """

    def __init__(
        self,
        img_dir: str,
        csv_path: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[transforms.Compose] = None,
        is_test: bool = False,
        subset_fraction: Optional[float] = None,
        random_seed: int = 42,
    ):
        """
        Initialize the dataset.

        Args:
            img_dir: Path to directory containing images (train_images or test_images).
            csv_path: Path to train.csv containing ImageId, ClassId, and EncodedPixels.
            image_size: Target dimensions (height, width) (default: 224x224).
            transform: Optional custom torchvision transform.
            is_test: Set True for unlabelled test sets without annotations.
            subset_fraction: Optional float (0.0 < fraction <= 1.0) to subsample the dataset.
            random_seed: Random seed for reproducible subsampling.
        """
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.image_size = image_size
        self.is_test = is_test

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
                ]
            )

        self.image_to_rle: Dict[str, Dict[int, str]] = {}
        self.image_ids: List[str] = []

        if not self.is_test and self.csv_path and os.path.exists(self.csv_path):
            self._load_annotations()
        elif os.path.exists(self.img_dir):
            valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
            self.image_ids = [
                f for f in sorted(os.listdir(self.img_dir)) if f.lower().endswith(valid_exts)
            ]
        else:
            self.image_ids = []

        if subset_fraction is not None and 0.0 < subset_fraction < 1.0:
            np.random.seed(random_seed)
            sample_size = max(1, int(len(self.image_ids) * subset_fraction))
            self.image_ids = list(np.random.choice(self.image_ids, size=sample_size, replace=False))

    def _load_annotations(self) -> None:
        """Parse train.csv supporting both legacy and modern Severstal CSV schemas."""
        df = pd.read_csv(self.csv_path)

        if "ImageId_ClassId" in df.columns:
            split_data = df["ImageId_ClassId"].str.rsplit("_", n=1, expand=True)
            df["ImageId"] = split_data[0]
            df["ClassId"] = split_data[1].astype(int)

        df["ImageId"] = df["ImageId"].astype(str)
        df["ClassId"] = df["ClassId"].astype(int)

        grouped = df.groupby("ImageId")
        for image_id, group in grouped:
            class_dict: Dict[int, str] = {}
            for _, row in group.iterrows():
                class_id = int(row["ClassId"])
                rle = row["EncodedPixels"]
                if pd.notna(rle) and str(rle).strip():
                    class_dict[class_id] = str(rle).strip()
            self.image_to_rle[image_id] = class_dict

        if os.path.exists(self.img_dir):
            available_files = set(os.listdir(self.img_dir))
            self.image_ids = [img_id for img_id in self.image_to_rle.keys() if img_id in available_files]
            if not self.image_ids:
                self.image_ids = sorted(list(self.image_to_rle.keys()))
        else:
            self.image_ids = sorted(list(self.image_to_rle.keys()))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """
        Load image and generate [4, 224, 224] target segmentation mask.

        Returns:
            Dict containing:
                - 'image': Tensor of shape [3, 224, 224] (normalized for DINOv2)
                - 'mask': Tensor of shape [4, 224, 224] (binary target for U-Net loss)
                - 'image_id': String filename
                - 'has_defect': Boolean flag
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.img_dir, image_id)

        if os.path.exists(image_path):
            pil_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = pil_img.size
        else:
            orig_w, orig_h = 1600, 256
            pil_img = Image.new("RGB", (orig_w, orig_h), color=(128, 128, 128))

        img_tensor = self.transform(pil_img)

        # Build full mask [4, orig_h, orig_w]
        full_mask_np = np.zeros((4, orig_h, orig_w), dtype=np.uint8)
        rle_dict = self.image_to_rle.get(image_id, {})
        has_defect = False

        for class_id in range(1, 5):
            if class_id in rle_dict:
                rle_str = rle_dict[class_id]
                cls_mask = rle_to_mask(rle_str, width=orig_w, height=orig_h)
                full_mask_np[class_id - 1] = cls_mask
                if np.any(cls_mask):
                    has_defect = True

        full_mask_tensor = torch.from_numpy(full_mask_np).float()  # [4, H, W]

        # Crucial Step: Resize the mask tensor to [4, 224, 224] using nearest neighbor
        mask_4d = full_mask_tensor.unsqueeze(0)  # [1, 4, H, W]
        mask_224 = F.interpolate(mask_4d, size=self.image_size, mode="nearest").squeeze(0)  # [4, 224, 224]
        mask_224 = (mask_224 > 0.5).float()

        return {
            "image": img_tensor,
            "mask": mask_224,
            "image_id": image_id,
            "has_defect": has_defect,
        }

