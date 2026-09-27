import os
import sys
import json
import time
from typing import Any

import numpy as np
import rasterio

import torch
import torch.nn as nn

import open_clip

from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

BEFORE_YEAR = 2022
AFTER_YEAR = 2025

SEMANTIC_QUERY = "urban construction"

MODEL_ID = 2

PATCH_SIZE = 256

DATA_ROOT = r"C:\SATORBITDATA\scenes\sentinel-2"

MODEL_PATH = r"C:\model_prediction\best_model.pth"

OUTPUT_DIR = r"C:\SATORBITDATA\derived\change_maps"

REMOTECLIP_CHECKPOINT = (
    r"C:\database\RemoteCLIP\checkpoints"
    r"\models--chendelong--RemoteCLIP"
    r"\snapshots\bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    r"\RemoteCLIP-ViT-B-32.pt"
)

REMOTECLIP_MODEL = "ViT-B-32"

MODEL_ID_REMOTECLIP = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def separator() -> None:
    print()
    print("=" * 70)
    print()


def stop(message: str) -> None:
    raise RuntimeError(message)


# ============================================================
# DATABASE CONNECTION
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

if DATABASE_URL is None:
    raise RuntimeError(
        "DATABASE_URL is not set in .env"
    )

engine = create_engine(
    DATABASE_URL
)


# ============================================================
# BAND PATH
# ============================================================

def get_band_path(
    year: int,
    band: str
) -> str:

    path = os.path.join(
        DATA_ROOT,
        str(year),
        f"{band}_10m.jp2"
    )

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Band file not found:\n{path}"
        )

    return path


# ============================================================
# NORMALIZE BAND
# ============================================================

def normalize_band(
    data: np.ndarray
) -> np.ndarray:

    data = data.astype(
        np.float32
    )

    valid = data[
        np.isfinite(data)
        &
        (data > 0)
    ]

    if valid.size == 0:

        return np.zeros(
            data.shape,
            dtype=np.float32
        )

    low = np.percentile(
        valid,
        2
    )

    high = np.percentile(
        valid,
        98
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


# ============================================================
# READ FOUR SENTINEL-2 BANDS
# ============================================================

def read_four_bands(
    year: int,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:

    # Rasterio tuple window.
    # This avoids the Pylance Window(...) error.

    window = (
        (y, y + height),
        (x, x + width)
    )

    bands = []

    for band in [
        "B02",
        "B03",
        "B04",
        "B08"
    ]:

        path = get_band_path(
            year,
            band
        )

        with rasterio.open(
            path
        ) as src:

            data = src.read(
                1,
                window=window
            )

        data = normalize_band(
            data
        )

        bands.append(
            data
        )

    return np.stack(
        bands,
        axis=0
    ).astype(
        np.float32
    )


# ============================================================
# PAD EDGE PATCH
# ============================================================

def pad_patch(
    data: np.ndarray
) -> np.ndarray:

    channels = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]

    if (
        height == PATCH_SIZE
        and
        width == PATCH_SIZE
    ):

        return data

    padded = np.zeros(
        (
            channels,
            PATCH_SIZE,
            PATCH_SIZE
        ),
        dtype=np.float32
    )

    padded[
        :,
        :height,
        :width
    ] = data

    return padded


# ============================================================
# LOAD REMOTECLIP
# ============================================================

def load_remoteclip() -> Any:

    print(
        "Loading RemoteCLIP..."
    )

    if not os.path.isfile(
        REMOTECLIP_CHECKPOINT
    ):

        stop(
            "RemoteCLIP checkpoint not found."
        )

    model: Any
    preprocess: Any

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            REMOTECLIP_MODEL,
            pretrained=None
        )
    )

    checkpoint = torch.load(
        REMOTECLIP_CHECKPOINT,
        map_location="cpu"
    )

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

        stop(
            "Invalid RemoteCLIP checkpoint format."
        )

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    print(
        "RemoteCLIP missing keys:",
        len(result.missing_keys)
    )

    print(
        "RemoteCLIP unexpected keys:",
        len(result.unexpected_keys)
    )

    if result.missing_keys:

        stop(
            "RemoteCLIP checkpoint is incomplete."
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    remoteclip: Any
) -> Any:

    print()
    print(
        "REMOTECLIP SEMANTIC SEARCH"
    )

    print(
        "Query:",
        SEMANTIC_QUERY
    )

    tokenizer = open_clip.get_tokenizer(
        REMOTECLIP_MODEL
    )

    tokens = tokenizer(
        [SEMANTIC_QUERY]
    )

    tokens = tokens.to(
        DEVICE
    )

    with torch.no_grad():

        text_embedding = (
            remoteclip.encode_text(
                tokens
            )
        )

        text_embedding = (
            text_embedding
            /
            text_embedding.norm(
                dim=-1,
                keepdim=True
            )
        )

    embedding = (
        text_embedding
        .cpu()
        .numpy()[0]
    )

    embedding_string = (
        "["
        +
        ",".join(
            str(float(value))
            for value in embedding
        )
        +
        "]"
    )

    query = text(
        """
        SELECT
            t.tile_id,
            t.scene_id,
            t.tile_row,
            t.tile_col,
            t.window_x,
            t.window_y,
            t.width,
            t.height,

            EXTRACT(
                YEAR
                FROM s.acquisition_datetime
            )::integer AS year,

            s.stac_id,

            1 - (
                e.embedding <=>
                CAST(
                    :embedding AS vector(512)
                )
            ) AS similarity

        FROM semantic.tile_embeddings e

        JOIN catalog.tiles t
            ON t.tile_id = e.tile_id

        JOIN catalog.satellite_scenes s
            ON s.scene_id = t.scene_id

        WHERE
            e.model_id = :model_id

        ORDER BY
            e.embedding <=>
            CAST(
                :embedding AS vector(512)
            )

        LIMIT 10
        """
    )

    rows: list[Any] = []

    with engine.connect() as connection:

        result = connection.execute(
            query,
            {
                "embedding": embedding_string,
                "model_id":
                    MODEL_ID_REMOTECLIP
            }
        )

        rows = list(
            result.fetchall()
        )

    if len(rows) == 0:

        stop(
            "No semantic search results found."
        )

    print()

    for index, row in enumerate(
        rows,
        start=1
    ):

        print(
            f"{index}. "
            f"Tile={row.tile_id} "
            f"Year={row.year} "
            f"Similarity="
            f"{float(row.similarity):.4f}"
        )

    selected: Any = rows[0]

    print()
    print(
        "SELECTED RELEVANT TILE"
    )

    print(
        "Tile ID:",
        selected.tile_id
    )

    print(
        "Scene ID:",
        selected.scene_id
    )

    print(
        "Tile row:",
        selected.tile_row
    )

    print(
        "Tile column:",
        selected.tile_col
    )

    print(
        "Window:",
        selected.window_x,
        selected.window_y,
        selected.width,
        selected.height
    )

    print(
        "Scene year:",
        selected.year
    )

    print(
        "Similarity:",
        f"{float(selected.similarity):.4f}"
    )

    return selected


# ============================================================
# GET SCENE USING scene_id
# ============================================================

def get_scene(
    scene_id: Any
) -> Any:

    query = text(
        """
        SELECT
            scene_id,
            stac_id,
            product_id,
            satellite,
            sensor,
            acquisition_datetime,
            cloud_cover,
            crs_epsg,
            resolution_m,
            width_px,
            height_px,
            raster_path,
            band_map

        FROM catalog.satellite_scenes

        WHERE scene_id = :scene_id
        """
    )

    row: Any = None

    with engine.connect() as connection:

        row = connection.execute(
            query,
            {
                "scene_id": scene_id
            }
        ).fetchone()

    if row is None:

        stop(
            f"Scene not found: {scene_id}"
        )

    return row


# ============================================================
# FIND SCENE FOR TARGET YEAR
# ============================================================

def find_scene_for_year(
    selected_scene_id: Any,
    target_year: int
) -> Any:

    query = text(
        """
        SELECT
            scene_id,
            stac_id,
            product_id,
            satellite,
            sensor,
            acquisition_datetime,
            cloud_cover,
            crs_epsg,
            resolution_m,
            width_px,
            height_px,
            raster_path,
            band_map

        FROM catalog.satellite_scenes

        WHERE
            EXTRACT(
                YEAR
                FROM acquisition_datetime
            )::integer = :target_year

            AND crs_epsg = (
                SELECT crs_epsg
                FROM catalog.satellite_scenes
                WHERE scene_id = :selected_scene_id
            )

            AND width_px = (
                SELECT width_px
                FROM catalog.satellite_scenes
                WHERE scene_id = :selected_scene_id
            )

            AND height_px = (
                SELECT height_px
                FROM catalog.satellite_scenes
                WHERE scene_id = :selected_scene_id
            )

        ORDER BY
            cloud_cover ASC

        LIMIT 1
        """
    )

    row: Any = None

    with engine.connect() as connection:

        row = connection.execute(
            query,
            {
                "target_year":
                    target_year,
                "selected_scene_id":
                    selected_scene_id
            }
        ).fetchone()

    if row is None:

        stop(
            f"No scene found for {target_year}."
        )

    return row


# ============================================================
# DOUBLE CONV
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.block(x)


# ============================================================
# SIAMESE U-NET
# ============================================================

class SiameseUNet(nn.Module):

    def __init__(
        self
    ) -> None:

        super().__init__()

        # -----------------------------
        # Encoder
        # -----------------------------

        self.enc1 = DoubleConv(
            4,
            32
        )

        self.pool1 = nn.MaxPool2d(
            2
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(
            2
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(
            2
        )

        self.enc4 = DoubleConv(
            128,
            256
        )

        self.pool4 = nn.MaxPool2d(
            2
        )

        self.bottleneck = DoubleConv(
            256,
            512
        )

        # -----------------------------
        # Decoder
        # -----------------------------

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        self.up0 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec0 = DoubleConv(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        e4 = self.enc4(
            self.pool3(e3)
        )

        b = self.bottleneck(
            self.pool4(e4)
        )

        return (
            e1,
            e2,
            e3,
            e4,
            b
        )

    # --------------------------------------------------------
    # DECODER
    # --------------------------------------------------------

    def decode(
        self,
        features: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
        ]
    ) -> torch.Tensor:

        e1 = features[0]
        e2 = features[1]
        e3 = features[2]
        e4 = features[3]
        b = features[4]

        x = self.up3(
            b
        )

        x = torch.cat(
            [
                x,
                e4
            ],
            dim=1
        )

        x = self.dec3(
            x
        )

        x = self.up2(
            x
        )

        x = torch.cat(
            [
                x,
                e3
            ],
            dim=1
        )

        x = self.dec2(
            x
        )

        x = self.up1(
            x
        )

        x = torch.cat(
            [
                x,
                e2
            ],
            dim=1
        )

        x = self.dec1(
            x
        )

        x = self.up0(
            x
        )

        x = torch.cat(
            [
                x,
                e1
            ],
            dim=1
        )

        x = self.dec0(
            x
        )

        output = self.final(
            x
        )

        return output

    # --------------------------------------------------------
    # SIAMESE FORWARD
    # --------------------------------------------------------

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor
    ) -> torch.Tensor:

        before_features = self.encode(
            before
        )

        after_features = self.encode(
            after
        )

        d1 = torch.abs(
            before_features[0]
            -
            after_features[0]
        )

        d2 = torch.abs(
            before_features[1]
            -
            after_features[1]
        )

        d3 = torch.abs(
            before_features[2]
            -
            after_features[2]
        )

        d4 = torch.abs(
            before_features[3]
            -
            after_features[3]
        )

        db = torch.abs(
            before_features[4]
            -
            after_features[4]
        )

        difference = (
            d1,
            d2,
            d3,
            d4,
            db
        )

        return self.decode(
            difference
        )


# ============================================================
# LOAD CHANGE DETECTION MODEL
# ============================================================

def load_change_model() -> Any:

    print()
    print(
        "LOADING best_model.pth"
    )

    if not os.path.isfile(
        MODEL_PATH
    ):

        stop(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = SiameseUNet()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    print(
        "Missing keys:",
        len(result.missing_keys)
    )

    print(
        "Unexpected keys:",
        len(result.unexpected_keys)
    )

    if result.missing_keys:

        print(
            result.missing_keys
        )

        stop(
            "best_model.pth does not match "
            "the SiameseUNet architecture."
        )

    if result.unexpected_keys:

        print(
            result.unexpected_keys
        )

        stop(
            "best_model.pth contains unexpected keys."
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Change model loaded successfully."
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

def test_model(
    model: Any
) -> None:

    print()
    print(
        "TESTING CHANGE DETECTION MODEL"
    )

    before = torch.zeros(
        (
            1,
            4,
            PATCH_SIZE,
            PATCH_SIZE
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    after = torch.zeros(
        (
            1,
            4,
            PATCH_SIZE,
            PATCH_SIZE
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            before,
            after
        )

    print(
        "Input shape:",
        tuple(before.shape)
    )

    print(
        "Output shape:",
        tuple(output.shape)
    )

    expected_shape = (
        1,
        1,
        PATCH_SIZE,
        PATCH_SIZE
    )

    if tuple(output.shape) != expected_shape:

        stop(
            "Unexpected model output shape: "
            +
            str(tuple(output.shape))
        )

    print(
        "MODEL TEST PASSED"
    )


# ============================================================
# PROCESS COMPLETE SCENE
# ============================================================

def process_scene(
    model: Any,
    before_year: int,
    after_year: int,
    width: int,
    height: int
) -> np.ndarray:

    print()
    print(
        "RUNNING CHANGE DETECTION "
        "ON COMPLETE RELEVANT SCENE"
    )

    print(
        "Before year:",
        before_year
    )

    print(
        "After year:",
        after_year
    )

    print(
        "Scene size:",
        width,
        "x",
        height
    )

    patches_x = (
        width + PATCH_SIZE - 1
    ) // PATCH_SIZE

    patches_y = (
        height + PATCH_SIZE - 1
    ) // PATCH_SIZE

    total_patches = (
        patches_x *
        patches_y
    )

    print(
        "Patches X:",
        patches_x
    )

    print(
        "Patches Y:",
        patches_y
    )

    print(
        "Total patches:",
        total_patches
    )

    probability_map = np.zeros(
        (
            height,
            width
        ),
        dtype=np.float32
    )

    start_time = time.time()

    patch_number = 0

    for y in range(
        0,
        height,
        PATCH_SIZE
    ):

        for x in range(
            0,
            width,
            PATCH_SIZE
        ):

            patch_number += 1

            patch_width = min(
                PATCH_SIZE,
                width - x
            )

            patch_height = min(
                PATCH_SIZE,
                height - y
            )

            print(
                f"Processing patch "
                f"{patch_number}/"
                f"{total_patches}"
            )

            before_data = read_four_bands(
                before_year,
                x,
                y,
                patch_width,
                patch_height
            )

            after_data = read_four_bands(
                after_year,
                x,
                y,
                patch_width,
                patch_height
            )

            before_data = pad_patch(
                before_data
            )

            after_data = pad_patch(
                after_data
            )

            before_tensor = (
                torch.from_numpy(
                    before_data
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            after_tensor = (
                torch.from_numpy(
                    after_data
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():

                logits = model(
                    before_tensor,
                    after_tensor
                )

                probability = torch.sigmoid(
                    logits
                )

            probability = (
                probability[
                    0,
                    0
                ]
                .cpu()
                .numpy()
            )

            probability = probability[
                :patch_height,
                :patch_width
            ]

            probability_map[
                y:y + patch_height,
                x:x + patch_width
            ] = probability

    elapsed = (
        time.time()
        -
        start_time
    )

    print()
    print(
        "COMPLETE SCENE PROCESSING FINISHED"
    )

    print(
        "Time:",
        round(elapsed, 2),
        "seconds"
    )

    return probability_map


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    probability_map: np.ndarray,
    before_year: int,
    after_year: int,
    selected_tile: Any,
    selected_scene: Any,
    before_scene: Any,
    after_scene: Any
) -> None:

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    reference_path = get_band_path(
        before_year,
        "B04"
    )

    with rasterio.open(
        reference_path
    ) as src:

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

    # ========================================================
    # PROBABILITY MAP
    # ========================================================

    probability_path = os.path.join(
        OUTPUT_DIR,
        f"change_probability_"
        f"{before_year}_"
        f"{after_year}.tif"
    )

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate",
        height=probability_map.shape[0],
        width=probability_map.shape[1],
        transform=transform,
        crs=crs
    )

    with rasterio.open(
        probability_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            probability_map,
            1
        )

    # ========================================================
    # BINARY MASK
    # ========================================================

    threshold = 0.5

    binary_map = (
        probability_map >= threshold
    ).astype(
        np.uint8
    )

    binary_path = os.path.join(
        OUTPUT_DIR,
        f"change_mask_"
        f"{before_year}_"
        f"{after_year}.tif"
    )

    binary_profile = profile.copy()

    binary_profile.update(
        dtype="uint8"
    )

    with rasterio.open(
        binary_path,
        "w",
        **binary_profile
    ) as dst:

        dst.write(
            binary_map,
            1
        )

    # ========================================================
    # PNG
    # ========================================================

    from PIL import Image

    png_path = os.path.join(
        OUTPUT_DIR,
        f"change_mask_"
        f"{before_year}_"
        f"{after_year}.png"
    )

    png_image = (
        binary_map * 255
    ).astype(
        np.uint8
    )

    Image.fromarray(
        png_image
    ).save(
        png_path
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    total_pixels = int(
        binary_map.size
    )

    changed_pixels = int(
        binary_map.sum()
    )

    change_percentage = (
        changed_pixels
        /
        total_pixels
        *
        100
    )

    # ========================================================
    # JSON
    # ========================================================

    json_path = os.path.join(
        OUTPUT_DIR,
        f"change_results_"
        f"{before_year}_"
        f"{after_year}.json"
    )

    results = {

        "semantic_query":
            SEMANTIC_QUERY,

        "relevant_tile_id":
            int(selected_tile.tile_id),

        "relevant_tile_scene_id":
            str(selected_tile.scene_id),

        "relevant_tile_row":
            int(selected_tile.tile_row),

        "relevant_tile_col":
            int(selected_tile.tile_col),

        "relevant_tile_window_x":
            int(selected_tile.window_x),

        "relevant_tile_window_y":
            int(selected_tile.window_y),

        "relevant_tile_width":
            int(selected_tile.width),

        "relevant_tile_height":
            int(selected_tile.height),

        "selected_scene_id":
            str(selected_scene.scene_id),

        "selected_scene_stac_id":
            str(selected_scene.stac_id),

        "before_scene_id":
            str(before_scene.scene_id),

        "before_scene_stac_id":
            str(before_scene.stac_id),

        "after_scene_id":
            str(after_scene.scene_id),

        "after_scene_stac_id":
            str(after_scene.stac_id),

        "before_year":
            before_year,

        "after_year":
            after_year,

        "scene_width":
            int(probability_map.shape[1]),

        "scene_height":
            int(probability_map.shape[0]),

        "total_pixels":
            total_pixels,

        "changed_pixels":
            changed_pixels,

        "change_percentage":
            float(change_percentage),

        "threshold":
            threshold,

        "probability_map":
            probability_path,

        "binary_change_map":
            binary_path,

        "png":
            png_path
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print(
        "RESULTS SAVED"
    )

    print()
    print(
        "Probability map:"
    )

    print(
        probability_path
    )

    print()
    print(
        "Binary change map:"
    )

    print(
        binary_path
    )

    print()
    print(
        "PNG:"
    )

    print(
        png_path
    )

    print()
    print(
        "JSON:"
    )

    print(
        json_path
    )

    print()
    print(
        "Changed pixels:",
        changed_pixels
    )

    print(
        "Change percentage:",
        f"{change_percentage:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    separator()

    print(
        "SAT ORBIT"
    )

    print(
        "REMOTECLIP"
    )

    print(
        "      ↓"
    )

    print(
        "RELEVANT TILE"
    )

    print(
        "      ↓"
    )

    print(
        "SCENE ID"
    )

    print(
        "      ↓"
    )

    print(
        "COMPLETE SCENE CHANGE DETECTION"
    )

    print()

    print(
        "Semantic query:",
        SEMANTIC_QUERY
    )

    print(
        "Before year:",
        BEFORE_YEAR
    )

    print(
        "After year:",
        AFTER_YEAR
    )

    print(
        "Device:",
        DEVICE
    )

    # ========================================================
    # STEP 1
    # ========================================================

    separator()

    print(
        "STEP 1: LOAD REMOTECLIP"
    )

    remoteclip = load_remoteclip()

    # ========================================================
    # STEP 2
    # ========================================================

    separator()

    print(
        "STEP 2: SEMANTIC SEARCH"
    )

    selected_tile = semantic_search(
        remoteclip
    )

    # ========================================================
    # STEP 3
    # ========================================================

    separator()

    print(
        "STEP 3: GET COMPLETE SCENE "
        "FROM RELEVANT TILE scene_id"
    )

    selected_scene = get_scene(
        selected_tile.scene_id
    )

    print()
    print(
        "Relevant tile:",
        selected_tile.tile_id
    )

    print(
        "Relevant scene_id:",
        selected_tile.scene_id
    )

    print(
        "Complete scene:",
        selected_scene.stac_id
    )

    print(
        "Scene dimensions:",
        selected_scene.width_px,
        "x",
        selected_scene.height_px
    )

    # ========================================================
    # STEP 4
    # ========================================================

    separator()

    print(
        f"STEP 4: FIND {BEFORE_YEAR} SCENE"
    )

    before_scene = find_scene_for_year(
        selected_scene.scene_id,
        BEFORE_YEAR
    )

    print()
    print(
        "Before scene_id:",
        before_scene.scene_id
    )

    print(
        "Before STAC ID:",
        before_scene.stac_id
    )

    print(
        "Before satellite:",
        before_scene.satellite
    )

    print(
        "Before cloud:",
        before_scene.cloud_cover
    )

    # ========================================================
    # STEP 5
    # ========================================================

    separator()

    print(
        f"STEP 5: FIND {AFTER_YEAR} SCENE"
    )

    after_scene = find_scene_for_year(
        selected_scene.scene_id,
        AFTER_YEAR
    )

    print()
    print(
        "After scene_id:",
        after_scene.scene_id
    )

    print(
        "After STAC ID:",
        after_scene.stac_id
    )

    print(
        "After satellite:",
        after_scene.satellite
    )

    print(
        "After cloud:",
        after_scene.cloud_cover
    )

    # ========================================================
    # STEP 6
    # ========================================================

    separator()

    print(
        "STEP 6: VERIFY SCENES"
    )

    if (
        int(before_scene.width_px)
        !=
        int(after_scene.width_px)
    ):

        stop(
            "Before and after widths do not match."
        )

    if (
        int(before_scene.height_px)
        !=
        int(after_scene.height_px)
    ):

        stop(
            "Before and after heights do not match."
        )

    if (
        int(before_scene.crs_epsg)
        !=
        int(after_scene.crs_epsg)
    ):

        stop(
            "Before and after CRS do not match."
        )

    print(
        "Dimensions match."
    )

    print(
        "CRS:",
        before_scene.crs_epsg
    )

    width = int(
        before_scene.width_px
    )

    height = int(
        before_scene.height_px
    )

    # ========================================================
    # STEP 7
    # ========================================================

    separator()

    print(
        "STEP 7: LOAD CHANGE MODEL"
    )

    model = load_change_model()

    # ========================================================
    # STEP 8
    # ========================================================

    separator()

    print(
        "STEP 8: TEST CHANGE MODEL"
    )

    test_model(
        model
    )

    # ========================================================
    # STEP 9
    # ========================================================

    separator()

    print(
        "STEP 9: PROCESS COMPLETE SCENE"
    )

    probability_map = process_scene(
        model,
        BEFORE_YEAR,
        AFTER_YEAR,
        width,
        height
    )

    # ========================================================
    # STEP 10
    # ========================================================

    separator()

    print(
        "STEP 10: SAVE RESULTS"
    )

    save_results(
        probability_map,
        BEFORE_YEAR,
        AFTER_YEAR,
        selected_tile,
        selected_scene,
        before_scene,
        after_scene
    )

    # ========================================================
    # FINISH
    # ========================================================

    separator()

    print(
        "SAT ORBIT PIPELINE COMPLETED"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()