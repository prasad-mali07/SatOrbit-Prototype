import os
import sys
from typing import Any

import torch
import open_clip
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


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

MODEL_ID = 2

TOP_K = 10

# Change this query whenever you want.
TEXT_QUERY = "urban construction"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SatOrbit - RemoteCLIP Semantic Search")
print("=" * 70)

print("\nQuery:")
print(TEXT_QUERY)

print("\nPyTorch version:")
print(torch.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())

print("\nDevice:")
print(DEVICE)


# ============================================================
# CHECK CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("CHECKING REMOTECLIP CHECKPOINT")
print("=" * 70)

print("\nCheckpoint:")
print(CHECKPOINT)

if not os.path.exists(CHECKPOINT):

    print("\nERROR: RemoteCLIP checkpoint not found.")

    sys.exit(1)

size_mb = os.path.getsize(CHECKPOINT) / (1024 * 1024)

print(f"OK ({size_mb:.2f} MB)")


# ============================================================
# LOAD DATABASE CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATABASE CONFIGURATION")
print("=" * 70)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:

    print("\nERROR: DATABASE_URL is not set in .env")

    sys.exit(1)

print("\nDATABASE_URL found.")


# ============================================================
# CREATE DATABASE ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL
)

print("Database engine created.")


# ============================================================
# TEST DATABASE CONNECTION
# ============================================================

print("\nTesting PostgreSQL connection...")

try:

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT 1")
        )

        result.scalar()

    print("PostgreSQL connection successful.")

except Exception as e:

    print("\nERROR: Could not connect to PostgreSQL.")

    print(e)

    sys.exit(1)


# ============================================================
# CHECK EMBEDDING COUNT
# ============================================================

print("\n" + "=" * 70)
print("CHECKING IMAGE EMBEDDINGS")
print("=" * 70)

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
        }
    )

    embedding_count = int(
        result.scalar() or 0
    )

print("\nImage embeddings available:")
print(embedding_count)

if embedding_count == 0:

    print(
        "\nERROR: No image embeddings found "
        "for model_id = 2."
    )

    sys.exit(1)


# ============================================================
# LOAD REMOTECLIP
# ============================================================

print("\n" + "=" * 70)
print("LOADING REMOTECLIP")
print("=" * 70)

print("\nCreating OpenCLIP ViT-B/32 model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=None,
)

model: Any = model

print("OpenCLIP model created.")


# ============================================================
# LOAD CHECKPOINT
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
    len(load_result.missing_keys)
)

print(
    "Unexpected keys:",
    len(load_result.unexpected_keys)
)

if len(load_result.missing_keys) > 0:

    print("\nWARNING: Missing keys detected.")

    for key in load_result.missing_keys[:10]:

        print(" ", key)

if len(load_result.unexpected_keys) > 0:

    print("\nWARNING: Unexpected keys detected.")

    for key in load_result.unexpected_keys[:10]:

        print(" ", key)


# ============================================================
# PREPARE MODEL
# ============================================================

model = model.to(DEVICE)

model.eval()

print("\nRemoteCLIP model ready.")

print("Running on:", DEVICE)


# ============================================================
# TOKENIZE TEXT QUERY
# ============================================================

print("\n" + "=" * 70)
print("CREATING TEXT EMBEDDING")
print("=" * 70)

tokenizer = open_clip.get_tokenizer(
    MODEL_NAME
)

text_input = tokenizer(
    [TEXT_QUERY]
)

print("\nQuery:")
print(TEXT_QUERY)

print("\nToken shape:")
print(text_input.shape)


# ============================================================
# GENERATE TEXT EMBEDDING
# ============================================================

print("\nGenerating text embedding...")

with torch.no_grad():

    text_features = model.encode_text(
        text_input
    )

    # L2 normalize.
    text_features = (
        text_features
        / text_features.norm(
            dim=-1,
            keepdim=True,
        )
    )


# ============================================================
# VERIFY TEXT EMBEDDING
# ============================================================

dimension = int(
    text_features.shape[-1]
)

embedding_norm = float(
    text_features
    .norm(dim=-1)
    .item()
)

print("\nText embedding shape:")
print(text_features.shape)

print("\nText embedding dimension:")
print(dimension)

print("\nText embedding norm:")
print(embedding_norm)

if dimension != 512:

    print(
        "\nERROR: Text embedding dimension "
        "is not 512."
    )

    sys.exit(1)


# ============================================================
# CONVERT VECTOR TO PGVECTOR FORMAT
# ============================================================

query_vector = text_features[0].cpu().tolist()

query_vector_string = (
    "["
    + ",".join(
        str(float(value))
        for value in query_vector
    )
    + "]"
)

print("\nText vector prepared for pgvector.")


# ============================================================
# SEARCH DATABASE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING PGVECTOR SEARCH")
print("=" * 70)

print("\nTop K:")
print(TOP_K)

print("\nSearching...")


sql = text(
    """
    SELECT
        e.tile_id,
        t.scene_id,
        EXTRACT(
            YEAR FROM t.acquisition_datetime
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
                    e.embedding
                    <=>
                    CAST(
                        :query_vector AS vector
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
        e.embedding
        <=>
        CAST(
            :query_vector AS vector
        )

    LIMIT :top_k
    """
)


with engine.connect() as connection:

    result = connection.execute(
        sql,
        {
            "query_vector": query_vector_string,
            "model_id": MODEL_ID,
            "top_k": TOP_K,
        }
    )

    rows = result.fetchall()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SEMANTIC SEARCH RESULTS")
print("=" * 70)

print("\nQuery:")
print(TEXT_QUERY)

print("\nNumber of results:")
print(len(rows))


if len(rows) == 0:

    print("\nNo matching tiles found.")

else:

    print("\n")

    for rank, row in enumerate(
        rows,
        start=1,
    ):

        print("-" * 70)

        print(f"Rank       : {rank}")

        print(f"Tile ID    : {row.tile_id}")

        print(f"Scene ID   : {row.scene_id}")

        print(f"Year       : {row.year}")

        print(
            f"Acquired   : {row.acquisition_datetime}"
        )

        print(
            f"Tile row   : {row.tile_row}"
        )

        print(
            f"Tile col   : {row.tile_col}"
        )

        print(
            f"Window     : "
            f"x={row.window_x}, "
            f"y={row.window_y}, "
            f"width={row.width}, "
            f"height={row.height}"
        )

        print(
            f"Longitude  : {row.longitude:.6f}"
        )

        print(
            f"Latitude   : {row.latitude:.6f}"
        )

        print(
            f"Similarity : {row.cosine_similarity}"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("SEMANTIC SEARCH COMPLETED")
print("=" * 70)