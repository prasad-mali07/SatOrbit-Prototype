import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

PATCH_DIR = r"C:\deep_learning_model\data\OSCD\mumbai_patches"

BATCH_SIZE = 4
RANDOM_STATE = 42


# ============================================================
# Find patches
# ============================================================

image_files = sorted([
    f for f in os.listdir(PATCH_DIR)
    if f.startswith("image_") and f.endswith(".npy")
])

mask_files = sorted([
    f for f in os.listdir(PATCH_DIR)
    if f.startswith("mask_") and f.endswith(".npy")
])


if len(image_files) != len(mask_files):
    raise ValueError(
        "Number of images and masks is different!"
    )


# ============================================================
# Train / Validation split
# ============================================================

indices = np.arange(len(image_files))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=RANDOM_STATE,
    shuffle=True
)


print("========================================")
print("DATASET SPLIT")
print("========================================")

print("Total patches     :", len(indices))
print("Training patches  :", len(train_indices))
print("Validation patches:", len(val_indices))


# ============================================================
# Dataset class
# ============================================================

class OSCDDataset(Dataset):

    def __init__(self, indices):

        self.indices = indices

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        patch_index = self.indices[idx]

        image_path = os.path.join(
            PATCH_DIR,
            image_files[patch_index]
        )

        mask_path = os.path.join(
            PATCH_DIR,
            mask_files[patch_index]
        )

        image = np.load(image_path).astype(
            np.float32
        )

        mask = np.load(mask_path).astype(
            np.float32
        )

        # Convert numpy → PyTorch
        image = torch.from_numpy(image)

        mask = torch.from_numpy(mask)

        return image, mask


# ============================================================
# Create datasets
# ============================================================

train_dataset = OSCDDataset(
    train_indices
)

val_dataset = OSCDDataset(
    val_indices
)


# ============================================================
# Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Test DataLoader
# ============================================================

print("\n========================================")
print("TESTING DATALOADER")
print("========================================")

images, masks = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Mask batch shape :", masks.shape)

print("Image dtype:", images.dtype)
print("Mask dtype :", masks.dtype)

print("Mask values:", torch.unique(masks))

print("\nDataset is ready!")