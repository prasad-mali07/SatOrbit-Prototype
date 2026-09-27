import os
import sys
from typing import Any

import torch
import open_clip


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

DEVICE = "cpu"

TEXT_QUERY = "urban construction"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("SatOrbit - RemoteCLIP Text Embedding Test")
print("=" * 60)

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", DEVICE)
print("Text query:", TEXT_QUERY)


# ============================================================
# CHECK CHECKPOINT
# ============================================================

print("\n" + "=" * 60)
print("CHECKING CHECKPOINT")
print("=" * 60)

print("\nRemoteCLIP checkpoint:")
print(CHECKPOINT)

if not os.path.exists(CHECKPOINT):
    print("\nERROR: RemoteCLIP checkpoint not found.")
    sys.exit(1)

size_mb = os.path.getsize(CHECKPOINT) / (1024 * 1024)

print(f"OK ({size_mb:.2f} MB)")


# ============================================================
# LOAD OPENCLIP MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING REMOTECLIP")
print("=" * 60)

print("\nCreating OpenCLIP ViT-B/32 model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=None,
)

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

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    print("Checkpoint contains 'state_dict'.")
    state_dict = checkpoint["state_dict"]
else:
    print("Checkpoint is a direct state dictionary.")
    state_dict = checkpoint


# ============================================================
# CLEAN STATE DICTIONARY
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


# ============================================================
# LOAD WEIGHTS
# ============================================================

print("\nLoading weights into model...")

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


# ============================================================
# PREPARE MODEL
# ============================================================

model = model.to(DEVICE)
model.eval()

print("\nModel is ready.")
print("Running on:", DEVICE)


# ============================================================
# TOKENIZE TEXT
# ============================================================

print("\n" + "=" * 60)
print("TOKENIZING TEXT")
print("=" * 60)

print("\nQuery:")
print(TEXT_QUERY)

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

text_input = tokenizer(
    [TEXT_QUERY]
)

print("\nToken tensor shape:")
print(text_input.shape)

print("Token dtype:")
print(text_input.dtype)


# ============================================================
# GENERATE TEXT EMBEDDING
# ============================================================

print("\n" + "=" * 60)
print("GENERATING REMOTECLIP TEXT EMBEDDING")
print("=" * 60)

print("\nDevice:", DEVICE)

print(
    "This may take some time because "
    "the current machine is CPU-only."
)

with torch.no_grad():

    text_features = model.encode_text(
        text_input
    )

    # L2 normalize the embedding.
    text_features = (
        text_features
        / text_features.norm(
            dim=-1,
            keepdim=True,
        )
    )


# ============================================================
# EMBEDDING RESULT
# ============================================================

print("\n" + "=" * 60)
print("TEXT EMBEDDING RESULT")
print("=" * 60)

print("\nEmbedding shape:")
print(text_features.shape)


# Get embedding dimension.
dimension = int(
    text_features.shape[-1]
)

print("\nEmbedding dimension:")
print(dimension)


# Check embedding norm.
embedding_norm = float(
    text_features
    .norm(dim=-1)
    .item()
)

print("\nEmbedding norm:")
print(embedding_norm)


# Print first 10 values.
print("\nFirst 10 embedding values:")

print(
    text_features[0][:10].tolist()
)


# ============================================================
# DATABASE COMPATIBILITY
# ============================================================

print("\n" + "=" * 60)
print("DATABASE COMPATIBILITY")
print("=" * 60)

DATABASE_DIMENSION = 512

print("\nPostgreSQL column:")
print("semantic.tile_embeddings.embedding")

print("\nDatabase type:")
print("vector(512)")

print("\nRemoteCLIP text dimension:")
print(dimension)

print("\nDatabase dimension:")
print(DATABASE_DIMENSION)


if dimension == DATABASE_DIMENSION:

    print("\nSUCCESS")

    print(
        "RemoteCLIP text encoder produces "
        "512-dimensional embeddings."
    )

    print(
        "Dimension is compatible with "
        "PostgreSQL vector(512)."
    )

else:

    print("\nWARNING")

    print(
        "RemoteCLIP text output dimension "
        "does NOT match PostgreSQL vector(512)."
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TEXT TEST COMPLETED")
print("=" * 60)