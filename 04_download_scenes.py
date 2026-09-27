# Step 1: Import required libraries
import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv
from pystac_client import Client


# ============================================================
# Step 2: Load environment variables
# ============================================================

load_dotenv()

ACCESS_KEY = os.getenv("CDSE_S3_ACCESS_KEY")
SECRET_KEY = os.getenv("CDSE_S3_SECRET_KEY")

if not ACCESS_KEY or not SECRET_KEY:
    raise RuntimeError(
        "Copernicus S3 credentials are missing from .env"
    )


# ============================================================
# Step 3: Copernicus STAC
# ============================================================

STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

catalog = Client.open(STAC_URL)

print("Connected to Copernicus STAC")
print()


# ============================================================
# Step 4: Create Copernicus S3 connection
# ============================================================

s3 = boto3.client(
    "s3",
    endpoint_url="https://eodata.dataspace.copernicus.eu",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="default",
)

print("Connected to Copernicus S3")
print()


# ============================================================
# Step 5: Read selected scenes
# ============================================================

with open(
    "selected_scenes.json",
    "r",
    encoding="utf-8"
) as f:

    selected_scenes = json.load(f)


# ============================================================
# Step 6: Local storage directory
# ============================================================

BASE_DIR = Path(
    r"C:\SATORBITDATA\scenes\sentinel-2"
)

BASE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Step 7: Required Sentinel-2 bands
# ============================================================

REQUIRED_BANDS = [
    "B02_10m",
    "B03_10m",
    "B04_10m",
    "B08_10m"
]


# ============================================================
# Step 8: Download function
# ============================================================

def download_s3_asset(s3_url, output_path):

    # Example:
    # s3://eodata/Sentinel-2/....

    if not s3_url.startswith("s3://"):
        raise ValueError(
            f"Unexpected asset URL: {s3_url}"
        )

    path_without_prefix = s3_url[5:]

    bucket, key = path_without_prefix.split(
        "/",
        1
    )

    print(f"Downloading:")
    print(f"  Bucket : {bucket}")
    print(f"  Key    : {key}")
    print(f"  Output : {output_path}")

    s3.download_file(
        bucket,
        key,
        str(output_path)
    )

    print("  Download completed.")
    print()


# ============================================================
# Step 9: Process each selected year
# ============================================================

for year in sorted(
    selected_scenes,
    key=int
):

    scene_id = selected_scenes[year]["scene_id"]

    print()
    print("================================================")
    print(f"YEAR: {year}")
    print("================================================")
    print(f"Scene ID: {scene_id}")
    print()


    # --------------------------------------------------------
    # Search exact scene
    # --------------------------------------------------------

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        ids=[scene_id]
    )

    items = list(search.items())

    if not items:
        print(
            f"Scene not found for {year}: {scene_id}"
        )
        continue

    item = items[0]


    # --------------------------------------------------------
    # Create year directory
    # --------------------------------------------------------

    year_dir = BASE_DIR / year

    year_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Download required bands
    # --------------------------------------------------------

    for band_name in REQUIRED_BANDS:

        if band_name not in item.assets:

            print(
                f"WARNING: {band_name} not found "
                f"for {year}"
            )

            continue


        asset = item.assets[band_name]

        s3_url = asset.href

        output_file = (
            year_dir /
            f"{band_name}.jp2"
        )


        # Skip already downloaded files
        if output_file.exists():

            print(
                f"Already exists: {output_file}"
            )

            continue


        download_s3_asset(
            s3_url,
            output_file
        )


# ============================================================
# Step 10: Finished
# ============================================================

print()
print("================================================")
print("DOWNLOAD PROCESS COMPLETED")
print("================================================")
print()
print(f"Files stored in:")
print(BASE_DIR)