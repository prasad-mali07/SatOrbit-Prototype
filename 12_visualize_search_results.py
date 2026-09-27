"""
SatOrbit - Visualize RemoteCLIP Semantic Search Results

Purpose:
    1. Create text embedding using RemoteCLIP.
    2. Search PostgreSQL + pgvector.
    3. Read top matching Sentinel-2 tiles.
    4. Create RGB PNG images.
    5. Save search results to CSV.

Query:
    urban construction
"""

from typing import Any
from pathlib import Path
import csv
import os
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

DEVICE = "cpu"

MODEL_ID = 2

TEXT_QUERY = "urban construction"

TOP_K = 10

DATA_ROOT = Path(r"C:\SATORBITDATA")

SCENES_DIR = (
    DATA_ROOT
    / "scenes"
    / "sentinel-2"
)

RESULTS_DIR = (
    DATA_ROOT
    / "semantic_search_results"
)

# Sentinel-2 RGB bands
RED_BAND = "B04_10m.jp2"
GREEN_BAND = "B03_10m.jp2"
BLUE_BAND = "B02_10m.jp2"

TILE_SIZE = 256


warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_heading(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def percentile_normalize(
    band: np.ndarray,
) -> np.ndarray:
    """
    Convert Sentinel-2 reflectance values to uint8
    using 2nd-98th percentile stretching.
    """

    band_float = np.asarray(
        band,
        dtype=np.float32,
    )

    valid = band_float[
        np.isfinite(band_float)
    ]

    if valid.size == 0:
        return np.zeros(
            band_float.shape,
            dtype=np.uint8,
        )

    low = float(
        np.percentile(valid, 2)
    )

    high = float(
        np.percentile(valid, 98)
    )

    if high <= low:
        return np.zeros(
            band_float.shape,
            dtype=np.uint8,
        )

    normalized = (
        band_float - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return (
        normalized * 255
    ).astype(np.uint8)


def read_band(
    path: Path,
    window_x: int,
    window_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Read one Sentinel-2 band.
    """

    # Tuple form avoids Rasterio/Pylance Window issues.
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

    with rasterio.open(path) as src:

        data = src.read(
            1,
            window=window,
        )

    return np.asarray(data)


def pad_to_256(
    band: np.ndarray,
) -> np.ndarray:
    """
    Pad edge tiles to 256 x 256.
    """

    padded = np.zeros(
        (
            TILE_SIZE,
            TILE_SIZE,
        ),
        dtype=band.dtype,
    )

    height = min(
        band.shape[0],
        TILE_SIZE,
    )

    width = min(
        band.shape[1],
        TILE_SIZE,
    )

    padded[
        :height,
        :width,
    ] = band[
        :height,
        :width,
    ]

    return padded


def create_rgb_image(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
) -> Image.Image:
    """
    Create RGB PIL image from Sentinel-2 bands.
    """

    red_8bit = percentile_normalize(red)

    green_8bit = percentile_normalize(green)

    blue_8bit = percentile_normalize(blue)

    rgb = np.stack(
        [
            red_8bit,
            green_8bit,
            blue_8bit,
        ],
        axis=-1,
    )

    return Image.fromarray(
        rgb,
        mode="RGB",
    )


def get_band_path(
    year: int,
    filename: str,
) -> Path:

    return (
        SCENES_DIR
        / str(year)
        / filename
    )


# ============================================================
# START
# ============================================================

print_heading(
    "SatOrbit - RemoteCLIP Semantic Search Visualization"
)

print()
print("Query:")
print(TEXT_QUERY)

print()
print("PyTorch:")
print(torch.__version__)

print()
print("CUDA available:")
print(torch.cuda.is_available())

print()
print("Device:")
print(DEVICE)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

print_heading(
    "LOADING DATABASE CONFIGURATION"
)

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set in .env"
    )

print("DATABASE_URL found.")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print()
print("Results directory:")
print(RESULTS_DIR)


# ============================================================
# CHECK CHECKPOINT
# ============================================================

print_heading(
    "CHECKING REMOTECLIP CHECKPOINT"
)

checkpoint = Path(
    CHECKPOINT_PATH
)

check_file(
    checkpoint,
    "RemoteCLIP checkpoint",
)

size_mb = (
    checkpoint.stat().st_size
    / (1024 * 1024)
)

print("Checkpoint:")
print(checkpoint)

print(
    f"Size: {size_mb:.2f} MB"
)

print("Checkpoint OK.")


# ============================================================
# DATABASE CONNECTION
# ============================================================

print_heading(
    "CONNECTING TO POSTGRESQL"
)

engine = create_engine(
    DATABASE_URL
)

with engine.connect() as connection:

    connection.execute(
        text("SELECT 1")
    )

print(
    "PostgreSQL connection successful."
)


# ============================================================
# CHECK EMBEDDINGS
# ============================================================

print_heading(
    "CHECKING IMAGE EMBEDDINGS"
)

with engine.connect() as connection:

    embedding_count_result = (
        connection.execute(
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
        ).scalar()
    )

embedding_count = int(
    embedding_count_result or 0
)

print(
    "Image embeddings available:"
)

print(embedding_count)

if embedding_count == 0:

    raise RuntimeError(
        "No embeddings found."
    )


# ============================================================
# LOAD REMOTECLIP
# ============================================================

print_heading(
    "LOADING REMOTECLIP"
)

print(
    "Creating OpenCLIP ViT-B/32 model..."
)

# Explicit Any fixes Pylance Tensor callable errors.
model_any: Any

model_any, _, preprocess_any = (
    open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=None,
    )
)

print(
    "OpenCLIP model created."
)

print()
print(
    "Loading RemoteCLIP checkpoint..."
)

checkpoint_data: Any = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

print(
    "Checkpoint file loaded."
)


# ============================================================
# EXTRACT STATE DICTIONARY
# ============================================================

if isinstance(
    checkpoint_data,
    dict
):

    if "state_dict" in checkpoint_data:

        state_dict = (
            checkpoint_data["state_dict"]
        )

        print(
            "Checkpoint contains state_dict."
        )

    elif "model_state_dict" in checkpoint_data:

        state_dict = (
            checkpoint_data[
                "model_state_dict"
            ]
        )

        print(
            "Checkpoint contains model_state_dict."
        )

    else:

        state_dict = checkpoint_data

        print(
            "Checkpoint is a direct state dictionary."
        )

else:

    raise RuntimeError(
        "Unexpected checkpoint format."
    )


# ============================================================
# CLEAN CHECKPOINT KEYS
# ============================================================

clean_state_dict: dict[str, Any] = {}

for key, value in state_dict.items():

    clean_key = key

    if clean_key.startswith(
        "module."
    ):

        clean_key = (
            clean_key[
                len("module.") :
            ]
        )

    clean_state_dict[
        clean_key
    ] = value


# ============================================================
# LOAD WEIGHTS
# ============================================================

print()
print(
    "Loading weights into model..."
)

missing_keys, unexpected_keys = (
    model_any.load_state_dict(
        clean_state_dict,
        strict=False,
    )
)

print(
    "Checkpoint loaded."
)

print(
    "Missing keys:",
    len(missing_keys),
)

print(
    "Unexpected keys:",
    len(unexpected_keys),
)

if len(missing_keys) > 0:

    print()
    print(
        "Missing key details:"
    )

    print(
        missing_keys
    )


if len(unexpected_keys) > 0:

    print()
    print(
        "Unexpected key details:"
    )

    print(
        unexpected_keys
    )


model_any = model_any.to(
    DEVICE
)

model_any.eval()

print()
print(
    "RemoteCLIP model ready."
)

print(
    "Running on:",
    DEVICE,
)


# ============================================================
# CREATE TEXT EMBEDDING
# ============================================================

print_heading(
    "CREATING TEXT EMBEDDING"
)

print("Query:")
print(TEXT_QUERY)


# Explicit Any also prevents tokenizer
# callable inference problems.
tokenizer_any: Any = (
    open_clip.get_tokenizer(
        MODEL_NAME
    )
)

text_tokens = tokenizer_any(
    [TEXT_QUERY]
)

text_tokens = text_tokens.to(
    DEVICE
)

print()
print("Token shape:")
print(text_tokens.shape)


print()
print(
    "Generating text embedding..."
)

with torch.no_grad():

    text_features = (
        model_any.encode_text(
            text_tokens
        )
    )


# Normalize embedding
text_features = (
    text_features
    / (
        text_features.norm(
            dim=-1,
            keepdim=True,
        )
        + 1e-12
    )
)


print()
print(
    "Text embedding shape:"
)

print(
    text_features.shape
)


embedding_dimension = int(
    text_features.shape[-1]
)

print()
print(
    "Text embedding dimension:"
)

print(
    embedding_dimension
)


print()
print(
    "Text embedding norm:"
)

print(
    float(
        text_features[0].norm().item()
    )
)


if embedding_dimension != 512:

    raise RuntimeError(
        "Expected 512-D embedding, "
        f"got {embedding_dimension}"
    )


# ============================================================
# PREPARE VECTOR
# ============================================================

query_vector = (
    text_features[
        0
    ]
    .detach()
    .cpu()
    .tolist()
)

query_vector_string = (
    "["
    + ",".join(
        str(float(value))
        for value in query_vector
    )
    + "]"
)


print()
print(
    "Text vector prepared for pgvector."
)


# ============================================================
# PGVECTOR SEARCH
# ============================================================

print_heading(
    "RUNNING PGVECTOR SEARCH"
)

print("Top K:")
print(TOP_K)

search_sql = text(
    """
    SELECT
        e.tile_id,
        t.scene_id,

        EXTRACT(
            YEAR
            FROM t.acquisition_datetime
        )::integer AS year,

        t.acquisition_datetime,

        t.tile_row,
        t.tile_col,

        t.window_x,
        t.window_y,

        t.width,
        t.height,

        ST_X(
            ST_Centroid(t.geom)
        ) AS longitude,

        ST_Y(
            ST_Centroid(t.geom)
        ) AS latitude,

        ROUND(
            (
                1 - (
                    e.embedding <=>
                    CAST(
                        :query_vector
                        AS vector
                    )
                )
            )::numeric,
            4
        ) AS cosine_similarity

    FROM semantic.tile_embeddings e

    JOIN catalog.tiles t
        ON t.tile_id = e.tile_id

    WHERE e.model_id = :model_id

    ORDER BY
        e.embedding <=>
        CAST(
            :query_vector
            AS vector
        )

    LIMIT :top_k
    """
)


with engine.connect() as connection:

    result = connection.execute(
        search_sql,
        {
            "query_vector":
                query_vector_string,

            "model_id":
                MODEL_ID,

            "top_k":
                TOP_K,
        },
    )

    results = result.mappings().all()


print()
print(
    "Search completed."
)

print()
print(
    "Number of results:"
)

print(
    len(results)
)


if len(results) == 0:

    raise RuntimeError(
        "No semantic search results."
    )


# ============================================================
# PROCESS RESULTS
# ============================================================

print_heading(
    "CREATING RGB IMAGES"
)

csv_path = (
    RESULTS_DIR
    / "search_results.csv"
)

csv_rows = []

successful = 0
failed = 0


for rank, result in enumerate(
    results,
    start=1,
):

    tile_id = int(
        result["tile_id"]
    )

    year = int(
        result["year"]
    )

    window_x = int(
        result["window_x"]
    )

    window_y = int(
        result["window_y"]
    )

    width = int(
        result["width"]
    )

    height = int(
        result["height"]
    )

    longitude = result[
        "longitude"
    ]

    latitude = result[
        "latitude"
    ]

    similarity = float(
        result[
            "cosine_similarity"
        ]
    )

    print()
    print("-" * 70)

    print(
        f"Rank       : {rank}"
    )

    print(
        f"Tile ID    : {tile_id}"
    )

    print(
        f"Year       : {year}"
    )

    print(
        f"Similarity : {similarity:.4f}"
    )

    print(
        "Window     : "
        f"x={window_x}, "
        f"y={window_y}, "
        f"width={width}, "
        f"height={height}"
    )


    # --------------------------------------------------------
    # BAND PATHS
    # --------------------------------------------------------

    red_path = get_band_path(
        year,
        RED_BAND,
    )

    green_path = get_band_path(
        year,
        GREEN_BAND,
    )

    blue_path = get_band_path(
        year,
        BLUE_BAND,
    )


    try:

        # ----------------------------------------------------
        # CHECK FILES
        # ----------------------------------------------------

        check_file(
            red_path,
            "B04 red band",
        )

        check_file(
            green_path,
            "B03 green band",
        )

        check_file(
            blue_path,
            "B02 blue band",
        )


        # ----------------------------------------------------
        # READ BANDS
        # ----------------------------------------------------

        print(
            "Reading B04..."
        )

        red = read_band(
            red_path,
            window_x,
            window_y,
            width,
            height,
        )


        print(
            "Reading B03..."
        )

        green = read_band(
            green_path,
            window_x,
            window_y,
            width,
            height,
        )


        print(
            "Reading B02..."
        )

        blue = read_band(
            blue_path,
            window_x,
            window_y,
            width,
            height,
        )


        # ----------------------------------------------------
        # PAD EDGE TILES
        # ----------------------------------------------------

        red = pad_to_256(
            red
        )

        green = pad_to_256(
            green
        )

        blue = pad_to_256(
            blue
        )


        # ----------------------------------------------------
        # CREATE RGB
        # ----------------------------------------------------

        print(
            "Creating RGB image..."
        )

        rgb_image = (
            create_rgb_image(
                red,
                green,
                blue,
            )
        )


        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        filename = (
            f"rank_{rank:02d}"
            f"_tile_{tile_id}"
            f"_{year}.png"
        )

        image_path = (
            RESULTS_DIR
            / filename
        )


        rgb_image.save(
            image_path
        )


        print(
            "Saved:"
        )

        print(
            image_path
        )


        successful += 1


        # ----------------------------------------------------
        # CSV ROW
        # ----------------------------------------------------

        csv_rows.append(
            {
                "rank": rank,
                "tile_id": tile_id,
                "year": year,
                "scene_id":
                    str(
                        result[
                            "scene_id"
                        ]
                    ),
                "acquisition_datetime":
                    str(
                        result[
                            "acquisition_datetime"
                        ]
                    ),
                "tile_row":
                    result["tile_row"],
                "tile_col":
                    result["tile_col"],
                "window_x":
                    window_x,
                "window_y":
                    window_y,
                "width":
                    width,
                "height":
                    height,
                "longitude":
                    longitude,
                "latitude":
                    latitude,
                "cosine_similarity":
                    similarity,
                "image_path":
                    str(image_path),
                "status":
                    "success",
            }
        )


    except Exception as error:

        failed += 1

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )


        csv_rows.append(
            {
                "rank": rank,
                "tile_id": tile_id,
                "year": year,
                "scene_id":
                    str(
                        result[
                            "scene_id"
                        ]
                    ),
                "acquisition_datetime":
                    str(
                        result[
                            "acquisition_datetime"
                        ]
                    ),
                "tile_row":
                    result["tile_row"],
                "tile_col":
                    result["tile_col"],
                "window_x":
                    window_x,
                "window_y":
                    window_y,
                "width":
                    width,
                "height":
                    height,
                "longitude":
                    longitude,
                "latitude":
                    latitude,
                "cosine_similarity":
                    similarity,
                "image_path":
                    "",
                "status":
                    f"failed: {error}",
            }
        )


# ============================================================
# SAVE CSV
# ============================================================

print_heading(
    "SAVING SEARCH REPORT"
)

fieldnames = [
    "rank",
    "tile_id",
    "year",
    "scene_id",
    "acquisition_datetime",
    "tile_row",
    "tile_col",
    "window_x",
    "window_y",
    "width",
    "height",
    "longitude",
    "latitude",
    "cosine_similarity",
    "image_path",
    "status",
]


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        csv_rows
    )


print()
print(
    "CSV saved:"
)

print(
    csv_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_heading(
    "VISUALIZATION COMPLETED"
)

print()
print(
    "Query:"
)

print(
    TEXT_QUERY
)

print()
print(
    "Search results:"
)

print(
    len(results)
)

print()
print(
    "Images successfully created:"
)

print(
    successful
)

print()
print(
    "Images failed:"
)

print(
    failed
)

print()
print(
    "Results folder:"
)

print(
    RESULTS_DIR
)

print()
print(
    "Open the folder:"
)

print(
    RESULTS_DIR
)

print()
print(
    "=" * 70
)

print(
    "PROCESS COMPLETED"
)

print(
    "=" * 70
)