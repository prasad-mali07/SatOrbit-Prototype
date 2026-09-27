from typing import Any
import os
import sys
import time
import warnings

import numpy as np
import torch
import open_clip
import rasterio
from PIL import Image

from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "ViT-B-32"

CHECKPOINT_PATH = (
    r"C:\database\RemoteCLIP\checkpoints"
    r"\models--chendelong--RemoteCLIP"
    r"\snapshots\bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    r"\RemoteCLIP-ViT-B-32.pt"
)

DATA_ROOT = r"C:\SATORBITDATA\scenes\sentinel-2"

DEVICE = "cpu"

MODEL_ID = 2

TILE_SIZE = 256

EMBEDDING_DIMENSION = 512

# Number of tiles fetched from PostgreSQL at one time
DB_BATCH_SIZE = 10

# Print progress after every N successful tiles
PROGRESS_EVERY = 10


# ============================================================
# SUPPRESS NON-FATAL RASTERIO / NUMPY WARNING
# ============================================================

warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_separator() -> None:
    print("=" * 60)


def normalize_band(data: np.ndarray) -> np.ndarray:
    """
    Normalize one Sentinel-2 band using 2nd and 98th percentiles.

    Output:
        Float32 array in range 0.0 - 1.0
    """

    data = np.asarray(
        data,
        dtype=np.float32
    )

    valid = data[np.isfinite(data)]

    if valid.size == 0:
        return np.zeros(
            data.shape,
            dtype=np.float32
        )

    low = float(
        np.percentile(valid, 2)
    )

    high = float(
        np.percentile(valid, 98)
    )

    if high <= low:
        return np.zeros(
            data.shape,
            dtype=np.float32
        )

    data = (
        data - low
    ) / (
        high - low
    )

    data = np.clip(
        data,
        0.0,
        1.0
    )

    return data.astype(
        np.float32
    )


def pad_to_tile_size(
    data: np.ndarray,
    target_size: int = TILE_SIZE
) -> np.ndarray:
    """
    Pad smaller edge tiles to 256 x 256.
    """

    height = int(data.shape[0])
    width = int(data.shape[1])

    if (
        height == target_size
        and width == target_size
    ):
        return data

    padded = np.zeros(
        (
            target_size,
            target_size
        ),
        dtype=data.dtype
    )

    copy_height = min(
        height,
        target_size
    )

    copy_width = min(
        width,
        target_size
    )

    padded[
        :copy_height,
        :copy_width
    ] = data[
        :copy_height,
        :copy_width
    ]

    return padded


def get_band_paths(
    year: int
) -> tuple[str, str, str]:
    """
    Return B04, B03 and B02 paths.
    """

    year_path = os.path.join(
        DATA_ROOT,
        str(year)
    )

    red_path = os.path.join(
        year_path,
        "B04_10m.jp2"
    )

    green_path = os.path.join(
        year_path,
        "B03_10m.jp2"
    )

    blue_path = os.path.join(
        year_path,
        "B02_10m.jp2"
    )

    return (
        red_path,
        green_path,
        blue_path
    )


def read_band(
    path: str,
    window: tuple[
        tuple[int, int],
        tuple[int, int]
    ]
) -> np.ndarray:
    """
    Read one Sentinel-2 band.
    """

    with rasterio.open(path) as src:

        data = src.read(
            1,
            window=window
        )

    return np.asarray(data)


def create_rgb_image(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray
) -> Image.Image:
    """
    Convert Sentinel-2 B04/B03/B02 into RGB PIL image.
    """

    red_normalized = normalize_band(
        red
    )

    green_normalized = normalize_band(
        green
    )

    blue_normalized = normalize_band(
        blue
    )

    red_padded = pad_to_tile_size(
        red_normalized
    )

    green_padded = pad_to_tile_size(
        green_normalized
    )

    blue_padded = pad_to_tile_size(
        blue_normalized
    )

    rgb = np.stack(
        [
            red_padded,
            green_padded,
            blue_padded
        ],
        axis=2
    )

    rgb = (
        rgb * 255.0
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        rgb,
        mode="RGB"
    )


def embedding_to_pgvector(
    embedding: np.ndarray
) -> str:
    """
    Convert NumPy embedding to PostgreSQL pgvector format.
    """

    values = ",".join(
        f"{float(value):.9f}"
        for value in embedding
    )

    return f"[{values}]"


# ============================================================
# START
# ============================================================

print_separator()
print(
    "SatOrbit - RemoteCLIP ViT-B/32"
)
print(
    "FULL EMBEDDING GENERATION"
)
print_separator()

print()

print(
    "PyTorch version:",
    torch.__version__
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# CHECK REMOTECLIP CHECKPOINT
# ============================================================

print_separator()
print(
    "CHECKING REMOTECLIP CHECKPOINT"
)
print_separator()

print()

print(
    "Checkpoint:"
)

print(
    CHECKPOINT_PATH
)

if not os.path.isfile(
    CHECKPOINT_PATH
):

    print()
    print(
        "ERROR: RemoteCLIP checkpoint not found."
    )

    sys.exit(1)

print(
    "OK"
)

print()


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

if DATABASE_URL is None:

    print(
        "ERROR: DATABASE_URL is not set in .env"
    )

    sys.exit(1)


# ============================================================
# CONNECT TO POSTGRESQL
# ============================================================

print_separator()
print(
    "CONNECTING TO POSTGRESQL"
)
print_separator()

print()

engine = create_engine(
    DATABASE_URL
)

with engine.connect() as connection:

    database_name_value = connection.execute(
        text(
            "SELECT current_database()"
        )
    ).scalar()

database_name = str(
    database_name_value
)

print(
    "Database:",
    database_name
)

print()


# ============================================================
# CHECK DATABASE MODEL
# ============================================================

print(
    "Checking database model..."
)

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
        }
    )

    model_row = result.fetchone()


if model_row is None:

    print()
    print(
        f"ERROR: Model ID {MODEL_ID} "
        f"does not exist."
    )

    sys.exit(1)


model_id_value = model_row.model_id
model_name_value = model_row.name
model_dimension_value = model_row.embedding_dim

model_id = int(
    model_id_value
)

model_name = str(
    model_name_value
)

database_embedding_dimension = int(
    model_dimension_value
)

print(
    "Model ID:",
    model_id
)

print(
    "Model name:",
    model_name
)

print(
    "Database embedding dimension:",
    database_embedding_dimension
)

if (
    database_embedding_dimension
    != EMBEDDING_DIMENSION
):

    print()
    print(
        "ERROR: Database embedding dimension "
        "does not match RemoteCLIP."
    )

    sys.exit(1)

print()


# ============================================================
# LOAD REMOTECLIP
# ============================================================

print_separator()
print(
    "LOADING REMOTECLIP"
)
print_separator()

print()

print(
    "Creating OpenCLIP ViT-B/32 model..."
)

model, _, preprocess = (
    open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=None
    )
)

# Help Pylance understand these are callable/model objects
model_any: Any = model
preprocess_fn: Any = preprocess

print(
    "OpenCLIP model created."
)

print()

print(
    "Loading RemoteCLIP checkpoint..."
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

print(
    "Checkpoint file loaded."
)


# ============================================================
# GET STATE DICTIONARY
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    if "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    print(
        "ERROR: Unexpected checkpoint format."
    )

    sys.exit(1)


# ============================================================
# LOAD WEIGHTS
# ============================================================

print(
    "Loading weights into model..."
)

load_result = model_any.load_state_dict(
    state_dict,
    strict=False
)

print()

print(
    "Checkpoint loaded."
)

print(
    "Missing keys:",
    len(load_result.missing_keys)
)

print(
    "Unexpected keys:",
    len(load_result.unexpected_keys)
)

if len(
    load_result.missing_keys
) != 0:

    print()
    print(
        "ERROR: Missing model weights."
    )

    sys.exit(1)

if len(
    load_result.unexpected_keys
) != 0:

    print()
    print(
        "ERROR: Unexpected model weights."
    )

    sys.exit(1)


model_any = model_any.to(
    DEVICE
)

model_any.eval()

print()

print(
    "Model is ready."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# CHECK EMBEDDING STATUS
# ============================================================

print_separator()
print(
    "EMBEDDING STATUS"
)
print_separator()

print()

with engine.connect() as connection:

    total_tiles_result = connection.execute(
        text(
            """
            SELECT COUNT(*)
            FROM catalog.tiles
            """
        )
    ).scalar()

    completed_tiles_result = connection.execute(
        text(
            """
            SELECT COUNT(*)
            FROM semantic.tile_embeddings
            WHERE model_id = :model_id
            """
        ),
        {
            "model_id": MODEL_ID
        }
    ).scalar()


# Explicit int conversion fixes Pylance Optional errors
total_tiles = int(
    total_tiles_result or 0
)

completed_tiles = int(
    completed_tiles_result or 0
)

remaining_tiles = (
    total_tiles
    - completed_tiles
)

print(
    "Total tiles:",
    total_tiles
)

print(
    "Already embedded:",
    completed_tiles
)

print(
    "Remaining:",
    remaining_tiles
)

print()


if remaining_tiles <= 0:

    print(
        "All tiles already have embeddings."
    )

    print()

    print_separator()

    sys.exit(0)


# ============================================================
# START EMBEDDING GENERATION
# ============================================================

print_separator()
print(
    "STARTING EMBEDDING GENERATION"
)
print_separator()

print()

print(
    "Database batch size:",
    DB_BATCH_SIZE
)

print(
    "Embedding dimension:",
    EMBEDDING_DIMENSION
)

print(
    "Tile size:",
    f"{TILE_SIZE} x {TILE_SIZE}"
)

print()


# ============================================================
# COUNTERS
# ============================================================

successful = 0
failed = 0

start_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # GET NEXT BATCH OF UNPROCESSED TILES
    # --------------------------------------------------------

    with engine.connect() as connection:

        result = connection.execute(
            text(
                """
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
                ORDER BY t.tile_id
                LIMIT :batch_size
                """
            ),
            {
                "model_id": MODEL_ID,
                "batch_size": DB_BATCH_SIZE
            }
        )

        tiles = result.fetchall()


    # --------------------------------------------------------
    # NO MORE TILES
    # --------------------------------------------------------

    if not tiles:

        break


    # --------------------------------------------------------
    # PROCESS EACH TILE
    # --------------------------------------------------------

    for tile in tiles:

        # Explicit conversion prevents Pylance Optional errors

        tile_id = int(
            tile.tile_id
        )

        scene_id = tile.scene_id

        window_x = int(
            tile.window_x
        )

        window_y = int(
            tile.window_y
        )

        width = int(
            tile.width
        )

        height = int(
            tile.height
        )

        year = int(
            tile.year
        )


        # Current overall progress
        current_number = (
            completed_tiles
            + successful
            + failed
            + 1
        )


        # ----------------------------------------------------
        # TILE INFORMATION
        # ----------------------------------------------------

        print_separator()

        print(
            f"Processing tile "
            f"{current_number}/{total_tiles}"
        )

        print(
            "Tile ID :",
            tile_id
        )

        print(
            "Scene ID:",
            scene_id
        )

        print(
            "Year    :",
            year
        )

        print(
            "Window  : "
            f"x={window_x}, "
            f"y={window_y}, "
            f"width={width}, "
            f"height={height}"
        )


        try:

            # ------------------------------------------------
            # GET BAND PATHS
            # ------------------------------------------------

            (
                red_path,
                green_path,
                blue_path
            ) = get_band_paths(
                year
            )


            # ------------------------------------------------
            # CHECK BAND FILES
            # ------------------------------------------------

            if not os.path.isfile(
                red_path
            ):

                raise FileNotFoundError(
                    f"B04 file not found: "
                    f"{red_path}"
                )

            if not os.path.isfile(
                green_path
            ):

                raise FileNotFoundError(
                    f"B03 file not found: "
                    f"{green_path}"
                )

            if not os.path.isfile(
                blue_path
            ):

                raise FileNotFoundError(
                    f"B02 file not found: "
                    f"{blue_path}"
                )


            # ------------------------------------------------
            # CREATE RASTERIO WINDOW
            # ------------------------------------------------

            window = (
                (
                    window_y,
                    window_y + height
                ),
                (
                    window_x,
                    window_x + width
                )
            )


            # ------------------------------------------------
            # READ B04
            # ------------------------------------------------

            red = read_band(
                red_path,
                window
            )


            # ------------------------------------------------
            # READ B03
            # ------------------------------------------------

            green = read_band(
                green_path,
                window
            )


            # ------------------------------------------------
            # READ B02
            # ------------------------------------------------

            blue = read_band(
                blue_path,
                window
            )


            # ------------------------------------------------
            # CHECK BAND SHAPES
            # ------------------------------------------------

            if (
                red.shape
                != green.shape
                or red.shape
                != blue.shape
            ):

                raise ValueError(
                    "B02, B03 and B04 "
                    "shapes do not match."
                )


            print(
                "Band shape:",
                red.shape
            )


            # ------------------------------------------------
            # CREATE RGB IMAGE
            # ------------------------------------------------

            image = create_rgb_image(
                red,
                green,
                blue
            )


            # ------------------------------------------------
            # REMOTECLIP PREPROCESSING
            # ------------------------------------------------

            image_input = preprocess_fn(
                image
            )

            image_input = (
                image_input.unsqueeze(0)
            )

            image_input = image_input.to(
                DEVICE
            )


            # ------------------------------------------------
            # GENERATE EMBEDDING
            # ------------------------------------------------

            with torch.no_grad():

                embedding_tensor = (
                    model_any.encode_image(
                        image_input
                    )
                )

                embedding_tensor = (
                    embedding_tensor
                    / embedding_tensor.norm(
                        dim=-1,
                        keepdim=True
                    )
                )


            # ------------------------------------------------
            # CONVERT TO NUMPY
            # ------------------------------------------------

            embedding = (
                embedding_tensor
                .cpu()
                .numpy()[0]
                .astype(
                    np.float32
                )
            )


            # ------------------------------------------------
            # CHECK EMBEDDING DIMENSION
            # ------------------------------------------------

            embedding_dimension = int(
                embedding.shape[0]
            )

            if (
                embedding_dimension
                != EMBEDDING_DIMENSION
            ):

                raise ValueError(
                    f"Expected "
                    f"{EMBEDDING_DIMENSION} "
                    f"dimensions, got "
                    f"{embedding_dimension}"
                )


            # ------------------------------------------------
            # CHECK FOR NaN / INFINITY
            # ------------------------------------------------

            if not np.all(
                np.isfinite(
                    embedding
                )
            ):

                raise ValueError(
                    "Embedding contains "
                    "NaN or infinity."
                )


            # ------------------------------------------------
            # CONVERT TO PGVECTOR STRING
            # ------------------------------------------------

            embedding_string = (
                embedding_to_pgvector(
                    embedding
                )
            )


            # ------------------------------------------------
            # INSERT INTO POSTGRESQL
            # ------------------------------------------------

            with engine.begin() as connection:

                connection.execute(
                    text(
                        """
                        INSERT INTO
                        semantic.tile_embeddings
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
                                :embedding
                                AS vector(512)
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
                        "embedding": embedding_string
                    }
                )


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            successful += 1

            print(
                "Embedding dimension:",
                embedding_dimension
            )

            print(
                "SUCCESS: Tile",
                tile_id,
                "stored in PostgreSQL."
            )


        except Exception as error:

            failed += 1

            print()

            print(
                "ERROR processing tile:",
                tile_id
            )

            print(
                "Error:",
                str(error)
            )

            print(
                "Tile skipped. Continuing..."
            )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            successful > 0
            and successful % PROGRESS_EVERY == 0
        ):

            elapsed = (
                time.time()
                - start_time
            )

            if elapsed > 0:

                rate = (
                    successful
                    / elapsed
                )

            else:

                rate = 0.0


            database_completed = (
                completed_tiles
                + successful
            )

            remaining = (
                total_tiles
                - database_completed
            )

            if rate > 0:

                estimated_seconds = (
                    remaining
                    / rate
                )

            else:

                estimated_seconds = 0.0


            print()

            print(
                "---------------- PROGRESS ----------------"
            )

            print(
                "Completed this run :",
                successful
            )

            print(
                "Failed this run    :",
                failed
            )

            print(
                "Database completed :",
                database_completed
            )

            print(
                "Remaining          :",
                max(
                    remaining,
                    0
                )
            )

            print(
                "Speed              :",
                f"{rate:.4f} tiles/sec"
            )

            print(
                "Estimated remaining:",
                f"{estimated_seconds / 60:.1f} minutes"
            )

            print(
                "-------------------------------------------"
            )


# ============================================================
# FINAL STATUS
# ============================================================

print()

print_separator()

print(
    "EMBEDDING GENERATION COMPLETED"
)

print_separator()

print()

elapsed = (
    time.time()
    - start_time
)

print(
    "Successfully processed this run:",
    successful
)

print(
    "Failed this run:",
    failed
)

print(
    "Time:",
    f"{elapsed / 60:.2f} minutes"
)

print()


# ============================================================
# FINAL DATABASE COUNT
# ============================================================

with engine.connect() as connection:

    final_count_result = connection.execute(
        text(
            """
            SELECT COUNT(*)
            FROM semantic.tile_embeddings
            WHERE model_id = :model_id
            """
        ),
        {
            "model_id": MODEL_ID
        }
    ).scalar()


final_count = int(
    final_count_result or 0
)


print(
    "Total RemoteCLIP embeddings in database:",
    final_count
)

print(
    "Total satellite tiles:",
    total_tiles
)

print()


# ============================================================
# FINAL RESULT
# ============================================================

if final_count >= total_tiles:

    print(
        "SUCCESS: All tiles have embeddings."
    )

else:

    remaining_after_run = (
        total_tiles
        - final_count
    )

    print(
        "Embeddings still remaining:",
        remaining_after_run
    )

print()

print_separator()