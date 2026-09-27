import os
import sys
from typing import Any

import torch
import open_clip
import rasterio
import numpy as np

from PIL import Image
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "ViT-B-32"

CHECKPOINT = (
    r"C:\database\RemoteCLIP\checkpoints"
    r"\models--chendelong--RemoteCLIP"
    r"\snapshots\bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    r"\RemoteCLIP-ViT-B-32.pt"
)

DATA_ROOT = r"C:\SATORBITDATA\scenes\sentinel-2"

DEVICE = "cpu"

# RemoteCLIP model ID in PostgreSQL
MODEL_ID = 2

# FIRST TEST:
# Process only 10 tiles.
# Later change this to None for all tiles.
MAX_TILES = 10

TILE_SIZE = 256

EMBEDDING_DIMENSION = 512


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL is not set in .env")
    sys.exit(1)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("SatOrbit - RemoteCLIP ViT-B/32 Embedding Generation")
print("=" * 60)

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", DEVICE)


# ============================================================
# CHECK REMOTECLIP CHECKPOINT
# ============================================================

print("\n" + "=" * 60)
print("CHECKING REMOTECLIP CHECKPOINT")
print("=" * 60)

print("\nCheckpoint:")
print(CHECKPOINT)

if not os.path.exists(CHECKPOINT):

    print("ERROR: Checkpoint not found.")
    sys.exit(1)

print("OK")


# ============================================================
# CONNECT TO DATABASE
# ============================================================

print("\n" + "=" * 60)
print("CONNECTING TO POSTGRESQL")
print("=" * 60)

engine = create_engine(DATABASE_URL)

with engine.connect() as connection:

    result = connection.execute(
        text("SELECT current_database();")
    )

    database_name = result.scalar()

print("Database:", database_name)


# ============================================================
# CHECK MODEL
# ============================================================

print("\nChecking database model...")

with engine.connect() as connection:

    result = connection.execute(
        text(
            """
            SELECT
                model_id,
                name,
                embedding_dim
            FROM catalog.ml_models
            WHERE model_id = :model_id
            """
        ),
        {
            "model_id": MODEL_ID
        },
    )

    model_row = result.fetchone()


if model_row is None:

    print(
        f"ERROR: model_id {MODEL_ID} was not found."
    )

    sys.exit(1)


print(
    "Model ID:",
    model_row.model_id
)

print(
    "Model name:",
    model_row.name
)

print(
    "Database embedding dimension:",
    model_row.embedding_dim
)


if model_row.embedding_dim != EMBEDDING_DIMENSION:

    print("\nERROR:")
    print(
        "Database embedding dimension does not "
        "match RemoteCLIP."
    )

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

# Pylance workaround
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


if (
    isinstance(checkpoint, dict)
    and "state_dict" in checkpoint
):

    state_dict = checkpoint["state_dict"]

else:

    state_dict = checkpoint


# Remove "module." prefix if present
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
    len(load_result.missing_keys)
)

print(
    "Unexpected keys:",
    len(load_result.unexpected_keys)
)


if len(load_result.missing_keys) != 0:

    print("\nERROR: Missing model keys.")

    for key in load_result.missing_keys[:10]:

        print(" ", key)

    sys.exit(1)


if len(load_result.unexpected_keys) != 0:

    print("\nERROR: Unexpected model keys.")

    for key in load_result.unexpected_keys[:10]:

        print(" ", key)

    sys.exit(1)


model = model.to(DEVICE)

model.eval()

print("\nModel is ready.")

print("Device:", DEVICE)


# ============================================================
# GET TILES THAT DO NOT HAVE EMBEDDINGS
# ============================================================

print("\n" + "=" * 60)
print("GETTING TILES FROM DATABASE")
print("=" * 60)

sql = """
SELECT
    t.tile_id,
    t.scene_id,
    t.window_x,
    t.window_y,
    t.width,
    t.height,
    EXTRACT(
        YEAR FROM t.acquisition_datetime
    )::integer AS year
FROM catalog.tiles AS t
LEFT JOIN semantic.tile_embeddings AS e
    ON e.tile_id = t.tile_id
    AND e.model_id = :model_id
WHERE e.tile_id IS NULL
ORDER BY
    t.scene_id,
    t.tile_id
"""

if MAX_TILES is not None:

    sql += """
    LIMIT :max_tiles
    """


with engine.connect() as connection:

    parameters = {
        "model_id": MODEL_ID
    }

    if MAX_TILES is not None:

        parameters["max_tiles"] = MAX_TILES

    result = connection.execute(
        text(sql),
        parameters,
    )

    tiles = result.fetchall()


print(
    "\nTiles selected:",
    len(tiles)
)


if len(tiles) == 0:

    print(
        "\nNo tiles need embeddings."
    )

    sys.exit(0)


# ============================================================
# PROCESS TILES
# ============================================================

processed = 0
failed = 0


for tile in tiles:

    tile_id = tile.tile_id

    scene_id = tile.scene_id

    window_x = tile.window_x

    window_y = tile.window_y

    width = tile.width

    height = tile.height

    year = tile.year


    print("\n" + "=" * 60)

    print(
        f"Processing tile "
        f"{processed + failed + 1}/{len(tiles)}"
    )

    print("Tile ID :", tile_id)

    print("Scene ID:", scene_id)

    print("Year    :", year)

    print(
        "Window  :",
        f"x={window_x},",
        f"y={window_y},",
        f"width={width},",
        f"height={height}",
    )


    try:

        # ====================================================
        # BAND FILE PATHS
        # ====================================================

        scene_dir = os.path.join(
            DATA_ROOT,
            str(year),
        )

        B02 = os.path.join(
            scene_dir,
            "B02_10m.jp2",
        )

        B03 = os.path.join(
            scene_dir,
            "B03_10m.jp2",
        )

        B04 = os.path.join(
            scene_dir,
            "B04_10m.jp2",
        )


        # ====================================================
        # CHECK BAND FILES
        # ====================================================

        for path in [B02, B03, B04]:

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"Band file not found: {path}"
                )


        # ====================================================
        # CREATE WINDOW
        # ====================================================

        # Same approach used successfully
        # in 07_test_remoteclip.py.

        window = (
            (
                window_y,
                window_y + height,
            ),
            (
                window_x,
                window_x + width,
            ),
        )


        # ====================================================
        # READ BANDS
        # ====================================================

        with rasterio.open(B04) as src:

            red = src.read(
                1,
                window=window,
            )

        with rasterio.open(B03) as src:

            green = src.read(
                1,
                window=window,
            )

        with rasterio.open(B02) as src:

            blue = src.read(
                1,
                window=window,
            )


        red = np.asarray(red)

        green = np.asarray(green)

        blue = np.asarray(blue)


        print(
            "Band shape:",
            red.shape
        )


        # ====================================================
        # NORMALIZE BAND
        # ====================================================

        def normalize_band(
            band: np.ndarray
        ) -> np.ndarray:

            band_float = band.astype(
                np.float32
            )

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

            if high <= low:

                high = low + 1.0

            band_float = np.clip(
                band_float,
                low,
                high,
            )

            band_float = (
                band_float - low
            ) / (
                high - low
            )

            band_uint8 = (
                band_float * 255.0
            ).astype(
                np.uint8
            )

            return band_uint8


        # ====================================================
        # NORMALIZE RGB
        # ====================================================

        red = normalize_band(red)

        green = normalize_band(green)

        blue = normalize_band(blue)


        # ====================================================
        # PAD EDGE TILES
        # ====================================================

        if (
            red.shape[0] != TILE_SIZE
            or red.shape[1] != TILE_SIZE
        ):

            padded_red = np.zeros(
                (
                    TILE_SIZE,
                    TILE_SIZE,
                ),
                dtype=np.uint8,
            )

            padded_green = np.zeros(
                (
                    TILE_SIZE,
                    TILE_SIZE,
                ),
                dtype=np.uint8,
            )

            padded_blue = np.zeros(
                (
                    TILE_SIZE,
                    TILE_SIZE,
                ),
                dtype=np.uint8,
            )


            padded_red[
                :red.shape[0],
                :red.shape[1]
            ] = red

            padded_green[
                :green.shape[0],
                :green.shape[1]
            ] = green

            padded_blue[
                :blue.shape[0],
                :blue.shape[1]
            ] = blue


            red = padded_red

            green = padded_green

            blue = padded_blue


        # ====================================================
        # CREATE RGB IMAGE
        # ====================================================

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


        # ====================================================
        # REMOTECLIP PREPROCESSING
        # ====================================================

        image_input = preprocess(
            image
        )

        image_input = (
            image_input
            .unsqueeze(0)
        )


        # ====================================================
        # GENERATE EMBEDDING
        # ====================================================

        with torch.no_grad():

            image_features = model.encode_image(
                image_input
            )

            image_features = (
                image_features
                / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )


        # ====================================================
        # CHECK EMBEDDING DIMENSION
        # ====================================================

        dimension = int(
            image_features.shape[-1]
        )

        print(
            "Embedding dimension:",
            dimension
        )


        if dimension != EMBEDDING_DIMENSION:

            raise RuntimeError(
                f"Expected {EMBEDDING_DIMENSION} "
                f"dimensions but received "
                f"{dimension}."
            )


        # ====================================================
        # CONVERT EMBEDDING TO LIST
        # ====================================================

        embedding_values = (
            image_features[
                0
            ]
            .cpu()
            .tolist()
        )


        # Convert to pgvector format:
        # [0.1,0.2,0.3,...]

        embedding_string = (
            "["
            + ",".join(
                str(float(value))
                for value in embedding_values
            )
            + "]"
        )


        # ====================================================
        # INSERT INTO POSTGRESQL
        # ====================================================

        with engine.begin() as connection:

            connection.execute(
                text(
                    """
                    INSERT INTO semantic.tile_embeddings
                    (
                        tile_id,
                        model_id,
                        embedding
                    )
                    VALUES
                    (
                        :tile_id,
                        :model_id,
                        CAST(
                            :embedding AS vector(512)
                        )
                    )
                    ON CONFLICT
                    (
                        tile_id,
                        model_id
                    )
                    DO NOTHING
                    """
                ),
                {
                    "tile_id": tile_id,
                    "model_id": MODEL_ID,
                    "embedding": embedding_string,
                },
            )


        processed += 1

        print(
            f"SUCCESS: Tile {tile_id} "
            f"stored in PostgreSQL."
        )


    except Exception as error:

        failed += 1

        print(
            f"\nERROR processing tile {tile_id}:"
        )

        print(
            str(error)
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDING GENERATION COMPLETED")
print("=" * 60)

print(
    "Successfully processed:",
    processed,
)

print(
    "Failed:",
    failed,
)


# ============================================================
# DATABASE COUNT
# ============================================================

with engine.connect() as connection:

    result = connection.execute(
        text(
            """
            SELECT COUNT(*)
            FROM semantic.tile_embeddings
            WHERE model_id = :model_id
            """
        ),
        {
            "model_id": MODEL_ID
        },
    )

    total = result.scalar()


print(
    "\nTotal RemoteCLIP embeddings in database:",
    total
)

print("=" * 60)