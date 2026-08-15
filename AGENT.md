# Severstal Steel Defect Detection (Zero-Shot IVFPQ Pipeline)

## Project Context
**Objective:** Build a scalable, multi-class anomaly detection and segmentation pipeline for the Severstal Steel dataset.  
**Architecture:** Frozen DINOv2 feature extraction paired with a highly compressed FAISS `IndexIVFPQ` memory bank.  
**Requirements:** Code must be modular, heavily commented, and structured as a professional GitHub repository.

---

## Phase 1: Repository Initialization
**Agent Task:** Create the project structure and establish the Python environment.

1. Create a new root directory named `Steel-Defect-Quantized-Inspector`.
2. Initialize a blank Git repository in this folder.
3. Create the following folder structure:
   * `data/severstal/`
   * `src/`
   * `notebooks/`
4. Create a `.gitignore` file. Ensure it ignores `data/`, `.venv`, `__pycache__/`, and `.index` files.
5. Create a `requirements.txt` file containing: `torch`, `torchvision`, `faiss-cpu`, `numpy`, `pandas`, `opencv-python`, `Pillow`, `matplotlib`, `scikit-learn`.

---

## Phase 2: Data Acquisition
**Agent Task:** Download the dataset using the Kaggle API.

1. Navigate to `data/severstal/` via the terminal.
2. Run the command: `kaggle competitions download -c severstal-steel-defect-detection`
3. Unzip the downloaded file into the same directory and delete the original `.zip` file.
4. Verify that `train_images/`, `test_images/`, and `train.csv` are present.

---

## Phase 3: RLE Utilities (`src/rle_utils.py`)
**Agent Task:** Write the utility functions to decode Kaggle's Run-Length Encoding.

1. Create `src/rle_utils.py`.
2. Write a function `rle_to_mask(rle_string, width=1600, height=256)`.
   * It must parse the RLE string.
   * It must return a binary 2D numpy array of the specified shape.
3. Write a function `mask_to_rle(mask)`.
   * It must convert a 2D binary numpy array back into a Kaggle-compliant RLE string.

---

## Phase 4: Dataset Module (`src/dataset.py`)
**Agent Task:** Build the PyTorch Dataset class to handle multi-class masks and patch mapping.

1. Create `src/dataset.py`.
2. Create a class `SeverstalDataset(Dataset)`.
3. In `__init__`, load `train.csv`. Group the dataframe by `ImageId` so each image has access to all 4 potential defect classes.
4. In `__getitem__`:
   * Load the image using PIL and convert to RGB.
   * Check the dataframe for any RLE strings for the 4 classes.
   * If defects exist, use `rle_to_mask` to generate a `[4, Height, Width]` mask tensor. 
   * If no defects exist, generate a zero-tensor of the same shape.
   * Apply necessary PyTorch transformations (Resize to 224x224, ToTensor, Normalize for DINOv2). 
   * Also resize the mask tensor to **16x16** using nearest-neighbor interpolation. (Note: DINOv2 ViT-B/14 on a 224x224 image creates a 16x16 grid of 14-pixel patches, yielding 256 patches total).
   * Return a dictionary: `{'image': tensor, 'mask_16x16': mask_tensor, 'image_id': string}`.

---

## Phase 5: Build Compressed Index (`src/build_index.py`)
**Agent Task:** Extract features and build the FAISS IVFPQ memory bank.

1. Create `src/build_index.py`.
2. Load the frozen `dinov2_vitb14` model to the GPU (or CPU if unavailable).
3. Initialize an empty list for patch embeddings and a corresponding list for patch labels (0 for Normal, 1-4 for Defects).
4. Iterate through the `SeverstalDataset`.
5. For each image, extract the `[256, 768]` patch embeddings.
6. Check the `mask_16x16`. 
   * Flatten the mask to match the 256 patches.
   * If a patch overlaps with a defect, append its embedding and label (1-4) to your lists.
   * If a patch is normal, append it with label 0 (implement a 5% random subsampling for normal patches to save memory).
7. Convert lists to `float32` numpy arrays.
8. Initialize `faiss.IndexIVFPQ(faiss.IndexFlatL2(768), 768, nlist=100, m=16, bits=8)`.
9. Train the index on the embeddings array: `index.train(embeddings)`.
10. Add the embeddings: `index.add(embeddings)`.
11. Save the index to `data/severstal_ivfpq.index`. Save the label array to `data/severstal_labels.npy`.

---

## Phase 6: Segmentation Pipeline (`src/evaluate.py`)
**Agent Task:** Write the inference and visualization script.

1. Create `src/evaluate.py`.
2. Load the frozen DINOv2 model, the saved FAISS index, and the saved labels array.
3. Write an `evaluate_image(image_path)` function:
   * Load and transform the test image.
   * Extract the 256 patch embeddings via DINOv2.
   * Query the FAISS index: `distances, indices = index.search(patch_embeddings, k=5)`.
   * Determine the class of each patch using a majority vote from the top 5 nearest neighbors' labels.
   * Reshape the 256 predicted labels back into a 16x16 grid.
   * Use OpenCV to resize/interpolate the 16x16 prediction grid back to the original image dimensions (1600x256).
   * Save a matplotlib figure side-by-side: Original Image, Predicted Defect Overlay (color-coded by class).
4. Save outputs to a new `results/` directory.