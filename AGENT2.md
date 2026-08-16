# AGENT2.md: Human-in-the-Loop Defect Segmentation (FAISS Data Engine + U-Net Decoder)

## Project Context
**Objective:** Build an end-to-end industrial anomaly detection pipeline demonstrating both Zero-Shot Data Mining and High-Precision Supervised Segmentation.  
**Architecture Strategy:** 
1. **The Data Engine (Real-World Simulation):** Use a zero-shot FAISS IVFPQ memory bank trained only on "Normal" steel to mine the dataset for anomalies, simulating a Human-in-the-Loop (HITL) active learning workflow.
2. **The Segmentation Head (Kaggle Scorer):** Train a lightweight U-Net Decoder on top of a frozen DINOv2 ViT-B/14 backbone using the "human-labeled" dataset to output pixel-perfect masks. 
**Evaluation Rules:** Only the predictions from the DINOv2-UNet architecture will be formatted into the final `submission.csv` for the Kaggle evaluator.

---

## Phase 1: Repository Initialization & Dependencies
**Agent Task:** Create the dual-pipeline project structure.

1. Create a root directory named `Steel-Defect-Active-Learning`.
2. Initialize a blank Git repository.
3. Create the following structure:
   * `data/severstal/`
   * `src/`
   * `notebooks/`
   * `results/`
4. Create a `.gitignore` ignoring `data/`, `.venv`, and `__pycache__/`.
5. Create `requirements.txt` containing: `torch`, `torchvision`, `faiss-cpu`, `numpy`, `pandas`, `opencv-python`, `Pillow`, `tqdm`.

---

## Phase 2: RLE Utilities (`src/rle_utils.py`)
**Agent Task:** Write utilities to decode and encode Kaggle's RLE format.

1. Create `src/rle_utils.py`.
2. Implement `rle_to_mask(rle_string, width=1600, height=256)` to return a 2D numpy binary mask.
3. Implement `mask_to_rle(mask)` to convert a predicted 2D binary mask back into a Kaggle-compliant RLE string.

---

## Phase 3: The Data Engine (FAISS Anomaly Mining)
**Agent Task:** Build the script simulating the Day-1 unannotated factory environment (`src/mine_anomalies.py`).

1. Load `train.csv`. Identify images that have NO defects (Normal). 
2. Take a subset of 200 Normal images. Pass them through frozen `dinov2_vitb14` (resize to 224x224) to extract 256 patches per image.
3. Build and train a `faiss.IndexIVFPQ` on these pure normal patches.
4. **The Mining Loop:** Iterate through the remaining images in the dataset (pretending they are unlabeled). 
   * Extract patches and query the FAISS index for L2 distance.
   * If the distance of any patch exceeds a set threshold, flag the image as "Defective Candidate".
5. Save a CSV called `data/flagged_for_human_review.csv` containing the flagged image IDs. *(Note: We state in the README that these flagged images were then passed to a human annotator, yielding our Kaggle ground truth masks).*

---

## Phase 4: Supervised Dataset Module (`src/dataset.py`)
**Agent Task:** Build the PyTorch Dataset for U-Net training.

1. Create `src/dataset.py`.
2. Create `SeverstalUNetDataset(Dataset)`.
3. In `__getitem__`:
   * Load the image and apply basic transforms (Resize to 224x224, ToTensor, Normalize).
   * Check `train.csv` for the 4 defect classes.
   * Generate a `[4, Height, Width]` binary mask tensor using `rle_to_mask`.
   * **Crucial Step:** Resize the mask tensor to `[4, 224, 224]` using nearest-neighbor interpolation to match the image dimensions for the U-Net loss calculation.
   * Return `{'image': tensor, 'mask': mask_tensor, 'image_id': string}`.

---

## Phase 5: The Hybrid Architecture (`src/model.py`)
**Agent Task:** Stitch a trainable CNN decoder onto the frozen ViT.

1. Create `src/model.py`.
2. Define a class `DinoUNetDecoder(nn.Module)`.
3. **The Encoder (Frozen):** Initialize `dinov2_vitb14`. Explicitly set `requires_grad = False` for all its parameters.
4. **The Decoder (Trainable):** Build a lightweight upsampling block.
   * Input: The `[Batch, 256, 768]` DINOv2 output.
   * Reshape: Mathematically fold the 256 patches into a `[Batch, 768, 16, 16]` spatial grid.
   * Upsampling Layers: Use a sequence of `nn.ConvTranspose2d` and `nn.Conv2d` layers with `ReLU` activations to progressively scale the `16x16` feature map up to `224x224`.
   * Final Layer: `nn.Conv2d(in_channels, 4, kernel_size=1)` to output exactly 4 channels (one for each defect class). No Sigmoid here (handle it in the loss function).

---

## Phase 6: Supervised Training Loop (`src/train_decoder.py`)
**Agent Task:** Train the U-Net's "hands" to draw pixel-perfect boundaries.

1. Create `src/train_decoder.py`.
2. Instantiate `SeverstalUNetDataset` and a DataLoader (split 80/20 for train/val).
3. Instantiate `DinoUNetDecoder` and move it to the GPU.
4. Define the Optimizer (AdamW, lr=1e-4) specifically strictly for `model.decoder.parameters()`.
5. Define the Loss Function: `BCEWithLogitsLoss`. 
6. Write a standard PyTorch training loop for 5 epochs. Calculate Validation Dice Score after each epoch to monitor improvement. 
7. Save the best weights to `results/best_unet_decoder.pth`.

---

## Phase 7: Kaggle Evaluator Pipeline (`src/generate_submission.py`)
**Agent Task:** Run the official test set through the U-Net and format the submission. *(The FAISS engine is completely ignored here).*

1. Create `src/generate_submission.py`.
2. Load the frozen DINOv2 model and the trained U-Net decoder weights.
3. Iterate through all images in `data/severstal/test_images/`.
4. For each image:
   * Resize to 224x224 and pass through `DinoUNetDecoder`.
   * Apply a Sigmoid activation to the output and threshold at `0.5` to get a binary mask of shape `[4, 224, 224]`.
   * Resize the 4 prediction masks back to the native `[256, 1600]` resolution using `cv2.resize` (nearest neighbor).
   * For each class (1 to 4):
     * If the mask has positive pixels, convert it to an RLE string using `mask_to_rle`.
     * Append to a list in the format `ImageId_ClassId, EncodedPixels`.
5. Convert the list to a Pandas DataFrame and save as `submission.csv`. This is the single file evaluated by Kaggle.