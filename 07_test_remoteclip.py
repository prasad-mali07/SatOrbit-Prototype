import os
import sys
from typing import Any

import torch
import open_clip
import rasterio
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "ViT-B-32"

CHECKPOINT = (
    r"C:\database\RemoteCLIP\checkpoints"
    r"\models--chendelong--RemoteCLIP"
    r"\snapshots\bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    r"\RemoteCLIP-ViT-B-32.pt"
)

SCENE_DIR = r"C:\SATORBITDATA\scenes\sentinel-2\2026"

B02 = os.path.join(SCENE_DIR, "B02_10m.jp2")
B03 = os.path.join(SCENE_DIR, "B03_10m.jp2")
B04 = os.path.join(SCENE_DIR, "B04_10m.jp2")

DEVICE = "cpu"

TILE_SIZE = 256


# ============================================================
# START
# ============================================================

print("=" * 60)
print("SatOrbit - RemoteCLIP ViT-B/32 Test")
print("=" * 60)

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", DEVICE)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\n" + "=" * 60)
print("CHECKING FILES")
print("=" * 60)

files_to_check = [
    ("RemoteCLIP checkpoint", CHECKPOINT),
    ("B02 Blue band", B02),
    ("B03 Green band", B03),
    ("B04 Red band", B04),
]

for name, path in files_to_check:

    print(f"\n{name}:")
    print(path)

    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  OK ({size_mb:.2f} MB)")
    else:
        print("  ERROR: File not found")
        sys.exit(1)


# ============================================================
# LOAD REMOTECLIP
# ============================================================

print("\n" + "=" * 60)
print("LOADING REMOTECLIP")
print("=" * 60)

print("\nCreating OpenCLIP ViT-B/32 model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=None,
)

# OpenCLIP's type information can confuse Pylance.
# Treat these objects as dynamic objects.
model: Any = model
preprocess: Any = preprocess

print("OpenCLIP model created.")


# ============================================================
# LOAD REMOTECLIP CHECKPOINT
# ============================================================

print("\nLoading RemoteCLIP checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

print("Checkpoint file loaded.")

# The downloaded RemoteCLIP checkpoint is expected
# to contain the OpenCLIP model state dictionary.

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

    print("Checkpoint contains 'state_dict'.")

    state_dict = checkpoint["state_dict"]

else:

    print("Checkpoint is a direct state dictionary.")

    state_dict = checkpoint


# Remove "module." prefix if present.
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


print("Loading weights into model...")

load_result = model.load_state_dict(
    clean_state_dict,
    strict=False,
)

print("\nCheckpoint loaded.")

print(
    "Missing keys:",
    len(load_result.missing_keys),
)

print(
    "Unexpected keys:",
    len(load_result.unexpected_keys),
)

if len(load_result.missing_keys) > 0:

    print("\nFirst missing keys:")

    for key in load_result.missing_keys[:10]:
        print(" ", key)

if len(load_result.unexpected_keys) > 0:

    print("\nFirst unexpected keys:")

    for key in load_result.unexpected_keys[:10]:
        print(" ", key)


# ============================================================
# PREPARE MODEL
# ============================================================

model = model.to(DEVICE)
model.eval()

print("\nModel is ready.")
print("Running on:", DEVICE)


# ============================================================
# READ ONE 256 x 256 TILE
# ============================================================

print("\n" + "=" * 60)
print("READING SENTINEL-2 TILE")
print("=" * 60)

print("\nTile size:", TILE_SIZE, "x", TILE_SIZE)

# First logical tile of the 2026 scene.
#
# Rasterio expects:
# ((row_start, row_stop), (col_start, col_stop))

window = (
    (0, TILE_SIZE),
    (0, TILE_SIZE),
)


# ============================================================
# READ BAND FUNCTION
# ============================================================

def read_band(path: str) -> np.ndarray:

    print("\nReading:")
    print(path)

    with rasterio.open(path) as src:

        data = src.read(
            1,
            window=window,
        )

    return np.asarray(data)


# ============================================================
# READ RGB BANDS
# ============================================================

red = read_band(B04)

green = read_band(B03)

blue = read_band(B02)


print("\nBand shapes:")

print("Red   :", red.shape)
print("Green :", green.shape)
print("Blue  :", blue.shape)


# ============================================================
# VERIFY TILE SIZE
# ============================================================

if red.shape != (TILE_SIZE, TILE_SIZE):

    print("\nERROR:")
    print("Red tile does not have expected dimensions.")

    sys.exit(1)

if green.shape != (TILE_SIZE, TILE_SIZE):

    print("\nERROR:")
    print("Green tile does not have expected dimensions.")

    sys.exit(1)

if blue.shape != (TILE_SIZE, TILE_SIZE):

    print("\nERROR:")
    print("Blue tile does not have expected dimensions.")

    sys.exit(1)


print("\n256 x 256 tile successfully read.")


# ============================================================
# NORMALIZE SATELLITE BAND
# ============================================================

def normalize_band(
    band: np.ndarray,
) -> np.ndarray:

    # Convert Sentinel-2 values to float32.
    band_float = band.astype(
        np.float32
    )

    # Robust percentile range.
    low = float(
        np.percentile(
            band_float,
            2,
        )
    )

    high = float(
        np.percentile(
            band_float,
            98,
        )
    )

    # Avoid division by zero.
    if high <= low:

        high = low + 1.0

    # Clip extreme values.
    band_float = np.clip(
        band_float,
        low,
        high,
    )

    # Scale to 0-1.
    band_float = (
        band_float - low
    ) / (
        high - low
    )

    # Convert to 8-bit image range.
    band_uint8 = (
        band_float * 255.0
    ).astype(
        np.uint8
    )

    return band_uint8


print("\nNormalizing bands...")

red = normalize_band(red)

green = normalize_band(green)

blue = normalize_band(blue)

print("Normalization complete.")


# ============================================================
# CREATE RGB IMAGE
# ============================================================

print("\n" + "=" * 60)
print("CREATING RGB IMAGE")
print("=" * 60)

rgb = np.stack(
    [
        red,
        green,
        blue,
    ],
    axis=2,
)

image = Image.fromarray(
    rgb,
    mode="RGB",
)

print("\nImage size:", image.size)

print("Image mode:", image.mode)


# ============================================================
# REMOTECLIP PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("REMOTECLIP PREPROCESSING")
print("=" * 60)

image_input = preprocess(
    image
)

image_input = image_input.unsqueeze(
    0
)

print("\nTensor shape:")
print(image_input.shape)

print("\nTensor dtype:")
print(image_input.dtype)


# ============================================================
# GENERATE EMBEDDING
# ============================================================

print("\n" + "=" * 60)
print("GENERATING REMOTECLIP EMBEDDING")
print("=" * 60)

print("\nDevice:", DEVICE)

print(
    "This may take some time because "
    "the current machine is CPU-only."
)

with torch.no_grad():

    image_features = model.encode_image(
        image_input
    )

    # L2 normalize the embedding.
    image_features = (
        image_features
        / image_features.norm(
            dim=-1,
            keepdim=True,
        )
    )


# ============================================================
# EMBEDDING RESULT
# ============================================================

print("\n" + "=" * 60)
print("REMOTECLIP TEST RESULT")
print("=" * 60)

print(
    "\nEmbedding shape:"
)

print(
    image_features.shape
)


# Get embedding dimension.
dimension = int(
    image_features.shape[-1]
)

print(
    "\nEmbedding dimension:"
)

print(
    dimension
)


# Check embedding norm.
embedding_norm = float(
    image_features
    .norm(dim=-1)
    .item()
)

print(
    "\nEmbedding norm:"
)

print(
    embedding_norm
)


# Print first 10 values.
print(
    "\nFirst 10 embedding values:"
)

print(
    image_features[0][:10].tolist()
)


# ============================================================
# DATABASE COMPATIBILITY
# ============================================================

print("\n" + "=" * 60)
print("DATABASE COMPATIBILITY")
print("=" * 60)

DATABASE_DIMENSION = 512

print(
    "\nPostgreSQL column:"
)

print(
    "semantic.tile_embeddings.embedding"
)

print(
    "Database type: vector(512)"
)

print(
    "\nRemoteCLIP dimension:",
    dimension,
)

print(
    "Database dimension:",
    DATABASE_DIMENSION,
)


if dimension == DATABASE_DIMENSION:

    print("\nSUCCESS")

    print(
        "RemoteCLIP ViT-B/32 produces "
        "512-dimensional embeddings."
    )

    print(
        "Compatible with PostgreSQL vector(512)."
    )

else:

    print("\nWARNING")

    print(
        "RemoteCLIP output dimension does "
        "NOT match PostgreSQL vector(512)."
    )

    print(
        "Do NOT insert embeddings into the "
        "database yet."
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)