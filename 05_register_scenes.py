import json
import os
import re
from pathlib import Path

import rasterio
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL)


# ============================================================
# 2. Load selected scenes
# ============================================================

with open("selected_scenes.json", "r", encoding="utf-8") as f:
    selected_scenes = json.load(f)


# ============================================================
# 3. Local Sentinel-2 storage
# ============================================================

BASE_DIR = Path(
    r"C:\SATORBITDATA\scenes\sentinel-2"
)


# ============================================================
# 4. Sentinel-2 bands
# ============================================================

BANDS = {
    "B02_10m": "B02",
    "B03_10m": "B03",
    "B04_10m": "B04",
    "B08_10m": "B08",
}


# ============================================================
# 5. Processing baseline
# ============================================================

def get_processing_baseline(scene_id):

    match = re.search(
        r"_N(\d{4})_",
        scene_id
    )

    if match:
        return f"N{match.group(1)}"

    return None


# ============================================================
# 6. Grid signature
# ============================================================

def get_grid_signature(scene_id):

    match = re.search(
        r"_(T\d{2}[A-Z]{3})_",
        scene_id
    )

    if match:
        return f"{match.group(1)}_10m"

    return "sentinel-2_10m"


# ============================================================
# 7. Raster bounds → WKT
# ============================================================

def bounds_to_wkt(src):

    left = src.bounds.left
    bottom = src.bounds.bottom
    right = src.bounds.right
    top = src.bounds.top

    return (
        f"POLYGON(("
        f"{left} {bottom}, "
        f"{right} {bottom}, "
        f"{right} {top}, "
        f"{left} {top}, "
        f"{left} {bottom}"
        f"))"
    )


# ============================================================
# 8. PostgreSQL connection
# ============================================================

with engine.begin() as connection:

    print("Connected to PostgreSQL")
    print()

    # ========================================================
    # 9. Get data source
    # ========================================================

    source_result = connection.execute(
        text("""
            SELECT source_id
            FROM catalog.data_sources
            ORDER BY source_id
            LIMIT 1
        """)
    ).fetchone()

    if not source_result:
        raise RuntimeError(
            "No data source found in catalog.data_sources"
        )

    source_id = source_result[0]

    print(f"Using data source ID: {source_id}")
    print()

    # ========================================================
    # 10. Process scenes
    # ========================================================

    for year in sorted(selected_scenes, key=int):

        scene = selected_scenes[year]

        stac_id = scene["scene_id"]
        acquisition_datetime = scene["datetime"]
        cloud_cover = scene["cloud_cover"]
        platform = scene["platform"]
        collection = scene["collection"]

        year_dir = BASE_DIR / year

        print("==============================================")
        print(f"Registering {year}")
        print("==============================================")
        print(f"STAC ID    : {stac_id}")
        print(f"Date       : {acquisition_datetime}")
        print(f"Platform   : {platform}")
        print(f"Cloud      : {cloud_cover}%")
        print()

        # ====================================================
        # 11. Check existing scene
        # ====================================================

        existing = connection.execute(
            text("""
                SELECT scene_id
                FROM catalog.satellite_scenes
                WHERE stac_id = :stac_id
            """),
            {
                "stac_id": stac_id
            }
        ).fetchone()

        if existing:

            scene_uuid = existing[0]

            print(
                f"Scene already exists: {scene_uuid}"
            )

        else:

            # =================================================
            # 12. Reference raster
            # =================================================

            reference_file = (
                year_dir / "B04_10m.jp2"
            )

            if not reference_file.exists():
                raise FileNotFoundError(
                    f"Missing reference raster: {reference_file}"
                )

            # =================================================
            # 13. Read raster metadata
            # =================================================

            with rasterio.open(reference_file) as src:

                crs_epsg = src.crs.to_epsg()

                if crs_epsg is None:
                    raise RuntimeError(
                        f"Could not determine EPSG for {reference_file}"
                    )

                width_px = src.width
                height_px = src.height

                resolution_x = abs(src.transform.a)
                resolution_y = abs(src.transform.e)

                resolution_m = (
                    resolution_x +
                    resolution_y
                ) / 2

                # IMPORTANT:
                # PostgreSQL constraint expects 6 values,
                # not Rasterio's 9-value Affine representation.
                transform = [
                    src.transform.a,
                    src.transform.b,
                    src.transform.c,
                    src.transform.d,
                    src.transform.e,
                    src.transform.f
                ]

                footprint_wkt = bounds_to_wkt(src)

            # =================================================
            # 14. Calculate total size
            # =================================================

            total_size = 0

            for asset_key in BANDS:

                file_path = (
                    year_dir /
                    f"{asset_key}.jp2"
                )

                if not file_path.exists():
                    raise FileNotFoundError(
                        f"Missing file: {file_path}"
                    )

                total_size += file_path.stat().st_size

            # =================================================
            # 15. Metadata
            # =================================================

            processing_baseline = (
                get_processing_baseline(stac_id)
            )

            grid_signature = (
                get_grid_signature(stac_id)
            )

            band_map = {
                "B02": "B02_10m.jp2",
                "B03": "B03_10m.jp2",
                "B04": "B04_10m.jp2",
                "B08": "B08_10m.jp2"
            }

            stac_properties = {
                "collection": collection,
                "platform": platform,
                "cloud_cover": cloud_cover,
                "tile": "T43QCA"
            }

            # =================================================
            # 16. Insert scene
            # =================================================

            result = connection.execute(
                text("""
                    INSERT INTO catalog.satellite_scenes
                    (
                        source_id,
                        stac_id,
                        product_id,
                        satellite,
                        sensor,
                        acquisition_datetime,
                        cloud_cover,
                        processing_level,
                        processing_baseline,
                        crs_epsg,
                        resolution_m,
                        width_px,
                        height_px,
                        transform,
                        grid_signature,
                        footprint,
                        storage_backend,
                        raster_path,
                        band_map,
                        raster_size_bytes,
                        ingestion_status,
                        stac_properties
                    )
                    VALUES
                    (
                        :source_id,
                        :stac_id,
                        :product_id,
                        :satellite,
                        :sensor,
                        :acquisition_datetime,
                        :cloud_cover,
                        :processing_level,
                        :processing_baseline,
                        :crs_epsg,
                        :resolution_m,
                        :width_px,
                        :height_px,
                        :transform,
                        :grid_signature,

                        ST_Transform(
                            ST_Multi(
                                ST_GeomFromText(
                                    :footprint,
                                    :crs_epsg
                                )
                            ),
                            4326
                        ),

                        :storage_backend,
                        :raster_path,
                        CAST(:band_map AS jsonb),
                        :raster_size_bytes,
                        :ingestion_status,
                        CAST(:stac_properties AS jsonb)
                    )
                    RETURNING scene_id
                """),
                {
                    "source_id": source_id,
                    "stac_id": stac_id,
                    "product_id": stac_id,
                    "satellite": platform,
                    "sensor": "MSI",
                    "acquisition_datetime": acquisition_datetime,
                    "cloud_cover": cloud_cover,
                    "processing_level": "L2A",
                    "processing_baseline": processing_baseline,
                    "crs_epsg": crs_epsg,
                    "resolution_m": resolution_m,
                    "width_px": width_px,
                    "height_px": height_px,

                    # 6-value transform
                    "transform": transform,

                    "grid_signature": grid_signature,
                    "footprint": footprint_wkt,
                    "storage_backend": "local",
                    "raster_path": str(year_dir),
                    "band_map": json.dumps(band_map),
                    "raster_size_bytes": total_size,
                    "ingestion_status": "registered",
                    "stac_properties": json.dumps(
                        stac_properties
                    )
                }
            )

            scene_uuid = result.scalar_one()

            print(
                f"Scene inserted: {scene_uuid}"
            )

        # ====================================================
        # 17. Register band assets
        # ====================================================

        for asset_key, band_name in BANDS.items():

            file_path = (
                year_dir /
                f"{asset_key}.jp2"
            )

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing file: {file_path}"
                )

            # ------------------------------------------------
            # Raster metadata
            # ------------------------------------------------

            with rasterio.open(file_path) as src:

                width_px = src.width
                height_px = src.height

                dtype = src.dtypes[0]

                nodata_value = src.nodata

                resolution_m = (
                    abs(src.res[0]) +
                    abs(src.res[1])
                ) / 2

                block_size = None

                if src.block_shapes:

                    block_height, block_width = (
                        src.block_shapes[0]
                    )

                    if block_height == block_width:
                        block_size = block_width

                compression = None

                if src.profile.get("compress"):
                    compression = str(
                        src.profile["compress"]
                    )

            file_size = file_path.stat().st_size

            # ------------------------------------------------
            # Check existing asset
            # ------------------------------------------------

            asset_exists = connection.execute(
                text("""
                    SELECT asset_id
                    FROM catalog.scene_assets
                    WHERE scene_id = :scene_id
                      AND role = 'band'
                      AND band_name = :band_name
                """),
                {
                    "scene_id": scene_uuid,
                    "band_name": band_name
                }
            ).fetchone()

            if asset_exists:

                print(
                    f"  Already registered: {asset_key}"
                )

                continue

            # ------------------------------------------------
            # Insert asset
            # ------------------------------------------------

            connection.execute(
                text("""
                    INSERT INTO catalog.scene_assets
                    (
                        scene_id,
                        role,
                        band_name,
                        storage_backend,
                        raster_path,
                        media_type,
                        dtype,
                        nodata_value,
                        width_px,
                        height_px,
                        resolution_m,
                        block_size,
                        compression,
                        size_bytes
                    )
                    VALUES
                    (
                        :scene_id,
                        'band',
                        :band_name,
                        'local',
                        :raster_path,
                        'image/jp2',
                        :dtype,
                        :nodata_value,
                        :width_px,
                        :height_px,
                        :resolution_m,
                        :block_size,
                        :compression,
                        :size_bytes
                    )
                """),
                {
                    "scene_id": scene_uuid,
                    "band_name": band_name,
                    "raster_path": str(file_path),
                    "dtype": dtype,
                    "nodata_value": nodata_value,
                    "width_px": width_px,
                    "height_px": height_px,
                    "resolution_m": resolution_m,
                    "block_size": block_size,
                    "compression": compression,
                    "size_bytes": file_size
                }
            )

            print(
                f"  Registered: {asset_key}"
            )

        print()


# ============================================================
# 18. Finished
# ============================================================

print("==============================================")
print("REGISTRATION COMPLETED")
print("==============================================")
print("==============================================")