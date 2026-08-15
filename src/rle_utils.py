"""
Run-Length Encoding (RLE) Utility Module for Severstal Steel Defect Detection.

This module provides high-performance encoding and decoding between binary 2D
segmentation masks and Kaggle-compliant Run-Length Encoded (RLE) string representations.
The Kaggle Severstal format operates in column-major (Fortran) order with 1-based indexing.
"""

from typing import Optional, Union
import numpy as np


def rle_to_mask(
    rle_string: Optional[Union[str, float]],
    width: int = 1600,
    height: int = 256,
) -> np.ndarray:
    """
    Decodes a Kaggle Severstal RLE string into a 2D binary segmentation mask.

    Kaggle Severstal encoding format:
    - Pairs of [start_pixel, run_length]
    - 1-indexed (the first pixel is 1)
    - Column-major / Fortran order (pixels numbered top-to-bottom, then left-to-right)

    Args:
        rle_string: Run-Length Encoded string (e.g. "1 3 10 5"), or NaN/None/empty for no defect.
        width: Image width in pixels (default: 1600).
        height: Image height in pixels (default: 256).

    Returns:
        mask: 2D numpy array of shape (height, width) with dtype np.uint8 (0 or 1).
    """
    # Initialize blank mask of specified dimensions
    mask = np.zeros(height * width, dtype=np.uint8)

    # Handle missing, null, or empty RLE representations
    if rle_string is None or (isinstance(rle_string, float) and np.isnan(rle_string)):
        return mask.reshape((height, width), order="F")

    rle_str = str(rle_string).strip()
    if not rle_str or rle_str.lower() == "nan":
        return mask.reshape((height, width), order="F")

    # Parse paired integers: start positions and run lengths
    elements = rle_str.split()
    if len(elements) % 2 != 0:
        raise ValueError(f"Invalid RLE string length (must be even number of values): '{rle_str}'")

    starts = np.asarray(elements[0::2], dtype=int) - 1  # Convert 1-based to 0-based indexing
    lengths = np.asarray(elements[1::2], dtype=int)
    ends = starts + lengths

    # Populate 1D flattened column-major array
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    # Reshape in Fortran (column-major) order to (height, width)
    return mask.reshape((height, width), order="F")


def mask_to_rle(mask: np.ndarray) -> str:
    """
    Encodes a 2D binary segmentation mask into a Kaggle-compliant RLE string.

    Args:
        mask: 2D binary numpy array of shape (height, width) where non-zero indicates defect.

    Returns:
        rle_string: Space-delimited string of pairs [start_pixel, length] in 1-based,
                    column-major order. Returns an empty string if no pixels are set.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D numpy array, got shape {mask.shape}")

    # Binarize and flatten in Fortran (column-major) order
    dots = (mask > 0).astype(np.uint8).flatten(order="F")

    # If completely empty, return empty string
    if not np.any(dots):
        return ""

    # Detect boundary transitions
    runs = np.where(dots[1:] != dots[:-1])[0] + 2

    # Account for starting and ending points
    runs = np.concatenate(
        [
            [1] if dots[0] else [],
            runs,
            [len(dots) + 1] if dots[-1] else [],
        ]
    )

    # Convert [start, end] pairs into [start, length]
    runs[1::2] -= runs[0::2]

    return " ".join(str(int(x)) for x in runs)
