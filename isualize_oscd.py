import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from PIL import Image

# ============================================================
# OSCD - Mumbai Dataset
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

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

img1_path = os.path.join(IMAGE_BASE, "pair", "img1.png")
img2_path = os.path.join(IMAGE_BASE, "pair", "img2.png")
mask_path = os.path.join(LABEL_BASE, "mumbai-cm.tif")

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

print("Checking files...")

for path in [img1_path, img2_path, mask_path]:
    if not os.path.exists(path):
        print("FILE NOT FOUND:")
        print(path)
        exit()

print("All files found!")
print()

# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------

img1 = Image.open(img1_path).convert("RGB")
img2 = Image.open(img2_path).convert("RGB")

# Convert mask
with rasterio.open(mask_path) as src:
    mask = src.read(1)

print("Image 1 size :", img1.size)
print("Image 2 size :", img2.size)
print("Mask shape   :", mask.shape)
print("Mask values  :", np.unique(mask))

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))

# First date
plt.subplot(1, 3, 1)
plt.imshow(img1)
plt.title("Mumbai - Date 1")
plt.axis("off")

# Second date
plt.subplot(1, 3, 2)
plt.imshow(img2)
plt.title("Mumbai - Date 2")
plt.axis("off")

# Ground truth
plt.subplot(1, 3, 3)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Change Mask")
plt.axis("off")

plt.tight_layout()

# ------------------------------------------------------------
# Save visualization
# ------------------------------------------------------------

output_path = os.path.join(BASE, "mumbai_visualization.png")

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print()
print("Visualization saved at:")
print(output_path)

plt.show()