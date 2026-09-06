import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rasterio

# ============================================================
# OSCD - Mumbai Dataset
# Verify images and ground-truth mask
# ============================================================

BASE = r"C:\deep_learning_model\data\OSCD"

PROCESSED = os.path.join(
    BASE,
    "processed_mumbai"
)

LABEL_BASE = os.path.join(
    BASE,
    "Onera Satellite Change Detection dataset - Train Labels",
    "Onera Satellite Change Detection dataset - Train Labels",
    "mumbai",
    "cm"
)

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

rgb1_path = os.path.join(
    PROCESSED,
    "imgs_1_rect_RGB.png"
)

rgb2_path = os.path.join(
    PROCESSED,
    "imgs_2_rect_RGB.png"
)

ndvi1_path = os.path.join(
    PROCESSED,
    "imgs_1_rect_NDVI.png"
)

ndvi2_path = os.path.join(
    PROCESSED,
    "imgs_2_rect_NDVI.png"
)

mask_path = os.path.join(
    LABEL_BASE,
    "mumbai-cm.tif"
)

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

print("Checking files...")

files = [
    rgb1_path,
    rgb2_path,
    ndvi1_path,
    ndvi2_path,
    mask_path
]

for path in files:

    if os.path.exists(path):
        print("FOUND:", path)
    else:
        print("NOT FOUND:", path)
        raise FileNotFoundError(path)

print("\nAll files found!")

# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------

rgb1 = np.array(Image.open(rgb1_path).convert("RGB"))
rgb2 = np.array(Image.open(rgb2_path).convert("RGB"))

ndvi1 = np.array(Image.open(ndvi1_path))
ndvi2 = np.array(Image.open(ndvi2_path))

# ------------------------------------------------------------
# Load ground-truth mask
# ------------------------------------------------------------

with rasterio.open(mask_path) as src:
    mask = src.read(1)

# ------------------------------------------------------------
# Print dimensions
# ------------------------------------------------------------

print("\n==============================")
print("IMAGE INFORMATION")
print("==============================")

print("Date 1 RGB shape :", rgb1.shape)
print("Date 2 RGB shape :", rgb2.shape)
print("Date 1 NDVI shape:", ndvi1.shape)
print("Date 2 NDVI shape:", ndvi2.shape)

print("\nMask shape        :", mask.shape)
print("Mask values      :", np.unique(mask))

# ------------------------------------------------------------
# Check spatial dimensions
# ------------------------------------------------------------

print("\n==============================")
print("DIMENSION CHECK")
print("==============================")

if rgb1.shape[:2] == rgb2.shape[:2]:
    print("RGB images: OK")
else:
    print("RGB images: SIZE MISMATCH")

if rgb1.shape[:2] == mask.shape:
    print("RGB and mask: OK")
else:
    print("RGB and mask: SIZE MISMATCH")

# ------------------------------------------------------------
# Calculate changed pixels
# ------------------------------------------------------------

unique, counts = np.unique(mask, return_counts=True)

print("\n==============================")
print("GROUND TRUTH INFORMATION")
print("==============================")

for value, count in zip(unique, counts):

    percentage = (count / mask.size) * 100

    print(
        f"Class {value}: "
        f"{count} pixels "
        f"({percentage:.2f}%)"
    )

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

plt.figure(figsize=(15, 8))

# Date 1
plt.subplot(2, 3, 1)
plt.imshow(rgb1)
plt.title("Mumbai - Date 1")
plt.axis("off")

# Date 2
plt.subplot(2, 3, 2)
plt.imshow(rgb2)
plt.title("Mumbai - Date 2")
plt.axis("off")

# Ground truth
plt.subplot(2, 3, 3)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Change")
plt.axis("off")

# NDVI Date 1
plt.subplot(2, 3, 4)
plt.imshow(ndvi1)
plt.title("NDVI - Date 1")
plt.axis("off")

# NDVI Date 2
plt.subplot(2, 3, 5)
plt.imshow(ndvi2)
plt.title("NDVI - Date 2")
plt.axis("off")

# Difference between dates
plt.subplot(2, 3, 6)

difference = np.abs(
    rgb2.astype(np.float32) -
    rgb1.astype(np.float32)
)

difference = difference / 255.0

plt.imshow(difference)
plt.title("Image Difference")
plt.axis("off")

plt.tight_layout()

# ------------------------------------------------------------
# Save result
# ------------------------------------------------------------

output_path = os.path.join(
    BASE,
    "mumbai_verification.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

print("\nVerification image saved at:")
print(output_path)

plt.show()