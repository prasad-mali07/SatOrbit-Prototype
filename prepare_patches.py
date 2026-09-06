import os
import numpy as np
import rasterio

# ============================================================
# OSCD - Mumbai
# Prepare paired Sentinel-2 patches for deep learning
# ============================================================

BASE = r"C:\deep_learning_model\data\OSCD"

IMAGE_BASE = os.path.join(
    BASE,
    "Onera Satellite Change Detection dataset - Images",
    "Onera Satellite Change Detection dataset - Images",
    "mumbai"
)

LABEL_BASE = os.path.join(
    BASE,
    "Onera Satellite Change Detection dataset - Train Labels",
    "Onera Satellite Change Detection dataset - Train Labels",
    "mumbai",
    "cm"
)

OUTPUT_BASE = os.path.join(
    BASE,
    "mumbai_patches"
)

os.makedirs(OUTPUT_BASE, exist_ok=True)

# ============================================================
# Settings
# ============================================================

PATCH_SIZE = 128
STRIDE = 64

BANDS = [
    "B02.tif",
    "B03.tif",
    "B04.tif",
    "B08.tif"
]


# ============================================================
# Function to load bands
# ============================================================

def load_image(folder):

    bands = []

    for band in BANDS:

        path = os.path.join(folder, band)

        with rasterio.open(path) as src:
            image = src.read(1).astype(np.float32)

        bands.append(image)

    # Shape:
    # (4, height, width)

    return np.stack(bands, axis=0)


# ============================================================
# Function to normalize each band
# ============================================================

def normalize_image(image):

    normalized = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):

        band = image[i]

        low = np.percentile(band, 2)
        high = np.percentile(band, 98)

        band = np.clip(band, low, high)

        if high > low:
            normalized[i] = (band - low) / (high - low)
        else:
            normalized[i] = 0

    return normalized


# ============================================================
# Load Date 1
# ============================================================

print("Loading Date 1...")

img1_folder = os.path.join(
    IMAGE_BASE,
    "imgs_1_rect"
)

img1 = load_image(img1_folder)

print("Date 1 shape:", img1.shape)


# ============================================================
# Load Date 2
# ============================================================

print("\nLoading Date 2...")

img2_folder = os.path.join(
    IMAGE_BASE,
    "imgs_2_rect"
)

img2 = load_image(img2_folder)

print("Date 2 shape:", img2.shape)


# ============================================================
# Check dimensions
# ============================================================

if img1.shape != img2.shape:

    raise ValueError(
        f"Date 1 and Date 2 dimensions do not match: "
        f"{img1.shape} vs {img2.shape}"
    )

print("\nBoth dates have matching dimensions.")


# ============================================================
# Normalize images
# ============================================================

print("\nNormalizing images...")

img1 = normalize_image(img1)
img2 = normalize_image(img2)


# ============================================================
# Load ground truth
# ============================================================

mask_path = os.path.join(
    LABEL_BASE,
    "mumbai-cm.tif"
)

print("\nLoading ground-truth mask...")

with rasterio.open(mask_path) as src:

    mask = src.read(1)

print("Original mask shape:", mask.shape)
print("Original mask values:", np.unique(mask))


# ============================================================
# Convert OSCD mask to binary
# ============================================================
# Original mask:
# 1 = No Change
# 2 = Change
#
# Converted mask:
# 0 = No Change
# 1 = Change

mask = (mask == 2).astype(np.uint8)

print("Binary mask values:", np.unique(mask))
# ============================================================
# Check image and mask dimensions
# ============================================================

if img1.shape[1:] != mask.shape:

    raise ValueError(
        f"Image and mask dimensions do not match: "
        f"{img1.shape[1:]} vs {mask.shape}"
    )

print("Image and mask dimensions match.")


# ============================================================
# Create 8-channel image
# ============================================================

# Date 1:
# B02 B03 B04 B08
#
# Date 2:
# B02 B03 B04 B08
#
# Final shape:
# (8, height, width)

combined = np.concatenate(
    [img1, img2],
    axis=0
)

print("\nCombined image shape:", combined.shape)


# ============================================================
# Create patches
# ============================================================

height = combined.shape[1]
width = combined.shape[2]

patch_count = 0

print("\nCreating patches...")

for y in range(0, height - PATCH_SIZE + 1, STRIDE):

    for x in range(0, width - PATCH_SIZE + 1, STRIDE):

        # ----------------------------------------------------
        # Extract image patch
        # ----------------------------------------------------

        image_patch = combined[
            :,
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        # ----------------------------------------------------
        # Extract mask patch
        # ----------------------------------------------------

        mask_patch = mask[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        image_file = os.path.join(
            OUTPUT_BASE,
            f"image_{patch_count:04d}.npy"
        )

        mask_file = os.path.join(
            OUTPUT_BASE,
            f"mask_{patch_count:04d}.npy"
        )

        np.save(
            image_file,
            image_patch.astype(np.float32)
        )

        np.save(
            mask_file,
            mask_patch.astype(np.uint8)
        )

        patch_count += 1


# ============================================================
# Summary
# ============================================================

print("\n========================================")
print("PATCH GENERATION COMPLETE")
print("========================================")

print("Image dimensions :", height, "x", width)
print("Patch size       :", PATCH_SIZE, "x", PATCH_SIZE)
print("Stride           :", STRIDE)
print("Channels         :", combined.shape[0])
print("Total patches    :", patch_count)

print("\nSaved to:")
print(OUTPUT_BASE)