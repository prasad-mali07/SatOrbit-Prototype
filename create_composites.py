import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# ============================================================
# OSCD - Mumbai Dataset
# Create RGB, False Color and NDVI images
# ============================================================

BASE = r"C:\deep_learning_model\data\OSCD"

IMAGE_BASE = os.path.join(
    BASE,
    "Onera Satellite Change Detection dataset - Images",
    "Onera Satellite Change Detection dataset - Images",
    "mumbai"
)

# Output folder
OUTPUT_BASE = os.path.join(BASE, "processed_mumbai")
os.makedirs(OUTPUT_BASE, exist_ok=True)


# ============================================================
# Function: Normalize image
# ============================================================

def normalize(image):
    """
    Normalize Sentinel-2 uint16 values to 0-1
    using 2nd and 98th percentile.
    """

    low = np.percentile(image, 2)
    high = np.percentile(image, 98)

    image = np.clip(image, low, high)

    if high == low:
        return np.zeros_like(image, dtype=np.float32)

    return (image - low) / (high - low)


# ============================================================
# Process both dates
# ============================================================

for date_folder in ["imgs_1_rect", "imgs_2_rect"]:

    print("\n========================================")
    print("Processing:", date_folder)
    print("========================================")

    folder = os.path.join(IMAGE_BASE, date_folder)

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    blue_path = os.path.join(folder, "B02.tif")
    green_path = os.path.join(folder, "B03.tif")
    red_path = os.path.join(folder, "B04.tif")
    nir_path = os.path.join(folder, "B08.tif")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [blue_path, green_path, red_path, nir_path]:

        if not os.path.exists(path):
            print("FILE NOT FOUND:")
            print(path)
            raise FileNotFoundError(path)

    # --------------------------------------------------------
    # Read bands
    # --------------------------------------------------------

    with rasterio.open(blue_path) as src:
        blue = src.read(1).astype(np.float32)

    with rasterio.open(green_path) as src:
        green = src.read(1).astype(np.float32)

    with rasterio.open(red_path) as src:
        red = src.read(1).astype(np.float32)

    with rasterio.open(nir_path) as src:
        nir = src.read(1).astype(np.float32)

    print("Band shape:", blue.shape)

    # --------------------------------------------------------
    # Normalize bands
    # --------------------------------------------------------

    blue_n = normalize(blue)
    green_n = normalize(green)
    red_n = normalize(red)
    nir_n = normalize(nir)

    # ========================================================
    # 1. RGB IMAGE
    # R = B04
    # G = B03
    # B = B02
    # ========================================================

    rgb = np.dstack([
        red_n,
        green_n,
        blue_n
    ])

    rgb_path = os.path.join(
        OUTPUT_BASE,
        f"{date_folder}_RGB.png"
    )

    plt.imsave(
        rgb_path,
        rgb
    )

    print("RGB saved:", rgb_path)

    # ========================================================
    # 2. FALSE COLOR IMAGE
    # R = B08 (NIR)
    # G = B04 (Red)
    # B = B03 (Green)
    # ========================================================

    false_color = np.dstack([
        nir_n,
        red_n,
        green_n
    ])

    false_color_path = os.path.join(
        OUTPUT_BASE,
        f"{date_folder}_FalseColor.png"
    )

    plt.imsave(
        false_color_path,
        false_color
    )

    print("False Color saved:", false_color_path)

    # ========================================================
    # 3. NDVI
    #
    # NDVI = (NIR - RED) / (NIR + RED)
    #
    # NIR = B08
    # RED = B04
    # ========================================================

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir, dtype=np.float32),
        where=denominator != 0
    )

    # Limit values to -1 to +1
    ndvi = np.clip(ndvi, -1, 1)

    ndvi_path = os.path.join(
        OUTPUT_BASE,
        f"{date_folder}_NDVI.png"
    )

    plt.imsave(
        ndvi_path,
        ndvi,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1
    )

    print("NDVI saved:", ndvi_path)


# ============================================================
# Complete
# ============================================================

print("\n========================================")
print("Composite generation complete!")
print("========================================")

print("\nOutput folder:")
print(OUTPUT_BASE)