import os
import numpy as np

# ============================================================
# Check generated Mumbai patches
# ============================================================

PATCH_DIR = r"C:\deep_learning_model\data\OSCD\mumbai_patches"

print("Checking patches...")
print()

image_files = sorted([
    f for f in os.listdir(PATCH_DIR)
    if f.startswith("image_") and f.endswith(".npy")
])

mask_files = sorted([
    f for f in os.listdir(PATCH_DIR)
    if f.startswith("mask_") and f.endswith(".npy")
])

print("Number of image patches:", len(image_files))
print("Number of mask patches :", len(mask_files))

if len(image_files) != len(mask_files):
    raise ValueError("Image and mask patch counts do not match!")

print("\nChecking first few patches...\n")

for i in range(min(5, len(image_files))):

    image_path = os.path.join(
        PATCH_DIR,
        image_files[i]
    )

    mask_path = os.path.join(
        PATCH_DIR,
        mask_files[i]
    )

    image = np.load(image_path)
    mask = np.load(mask_path)

    print(
        f"Patch {i}: "
        f"image={image.shape}, "
        f"mask={mask.shape}, "
        f"mask_values={np.unique(mask)}"
    )


# ============================================================
# Check all mask pixels
# ============================================================

total_pixels = 0
changed_pixels = 0
unchanged_pixels = 0

patches_with_change = 0
patches_without_change = 0

for mask_file in mask_files:

    mask_path = os.path.join(
        PATCH_DIR,
        mask_file
    )

    mask = np.load(mask_path)

    values = np.unique(mask)

    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(
            f"Invalid mask values in {mask_file}: {values}"
        )

    total_pixels += mask.size

    changed = np.sum(mask == 1)
    unchanged = np.sum(mask == 0)

    changed_pixels += changed
    unchanged_pixels += unchanged

    if changed > 0:
        patches_with_change += 1
    else:
        patches_without_change += 1


# ============================================================
# Results
# ============================================================

print("\n========================================")
print("PATCH CHECK COMPLETE")
print("========================================")

print("Total patches       :", len(image_files))
print("Total pixels        :", total_pixels)

print("Unchanged pixels    :", unchanged_pixels)
print("Changed pixels      :", changed_pixels)

print(
    "Changed percentage  :",
    f"{(changed_pixels / total_pixels) * 100:.2f}%"
)

print()
print("Patches with change :", patches_with_change)
print("Patches without change:", patches_without_change)

print("\nAll patches are valid.")