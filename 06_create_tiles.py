import os
from pathlib import Path

import rasterio
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# 1. Load environment
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL)


# ============================================================
# 2. Configuration
# ============================================================

TILE_SIZE = 256


# ============================================================
# 3. Get registered scenes
# ============================================================

with engine.begin() as connection:

    scenes = connection.execute(
        text("""
            SELECT
                scene_id,
                acquisition_datetime,
                raster_path
            FROM catalog.satellite_scenes
            ORDER BY acquisition_datetime
        """)
    ).fetchall()

    if not scenes:
        raise RuntimeError(
            "No scenes found in catalog.satellite_scenes"
        )

    print(f"Found {len(scenes)} registered scenes")
    print()

    total_tiles = 0


    # ========================================================
    # 4. Process every scene
    # ========================================================

    for scene in scenes:

        scene_id = scene[0]
        acquisition_datetime = scene[1]
        raster_path = Path(str(scene[2]))

        # ----------------------------------------------------
        # Reference raster
        # ----------------------------------------------------

        reference_file = (
            raster_path / "B04_10m.jp2"
        )

        if not reference_file.exists():
            raise FileNotFoundError(
                f"Reference raster not found:\n"
                f"{reference_file}"
            )

        print("==============================================")
        print(
            f"Processing scene: "
            f"{acquisition_datetime}"
        )
        print(
            f"Raster: {reference_file}"
        )
        print("==============================================")


        # ====================================================
        # 5. Open reference raster
        # ====================================================

        with rasterio.open(reference_file) as src:

            raster_width = int(src.width)
            raster_height = int(src.height)

            transform = src.transform
            crs = src.crs

            resolution_x = abs(
                float(transform.a)
            )

            resolution_y = abs(
                float(transform.e)
            )

            resolution_m = (
                resolution_x +
                resolution_y
            ) / 2.0

            print(
                f"Raster size : "
                f"{raster_width} x {raster_height}"
            )

            print(
                f"Resolution  : "
                f"{resolution_m} m"
            )

            print(
                f"CRS         : {crs}"
            )


        # ====================================================
        # 6. Check existing tiles
        # ====================================================

        result = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM catalog.tiles
                WHERE scene_id = :scene_id
            """),
            {
                "scene_id": scene_id
            }
        ).scalar()

        existing_count = int(result or 0)

        if existing_count > 0:

            print(
                f"Tiles already exist: "
                f"{existing_count}"
            )

            total_tiles += existing_count

            print()
            continue


        # ====================================================
        # 7. Create logical 256 x 256 tiles
        # ====================================================

        scene_tile_count = 0

        for window_y in range(
            0,
            raster_height,
            TILE_SIZE
        ):

            for window_x in range(
                0,
                raster_width,
                TILE_SIZE
            ):

                # --------------------------------------------
                # Actual tile dimensions
                # --------------------------------------------

                width = min(
                    TILE_SIZE,
                    raster_width - window_x
                )

                height = min(
                    TILE_SIZE,
                    raster_height - window_y
                )


                # --------------------------------------------
                # Tile row / column
                # --------------------------------------------

                tile_row = (
                    window_y // TILE_SIZE
                )

                tile_col = (
                    window_x // TILE_SIZE
                )


                # --------------------------------------------
                # Calculate native UTM coordinates
                #
                # Sentinel-2 T43QCA:
                # EPSG:32643
                #
                # Raster transform:
                # x = c + column * a
                # y = f + row * e
                # --------------------------------------------

                native_minx = float(
                    transform.c +
                    window_x * transform.a
                )

                native_maxx = float(
                    transform.c +
                    (window_x + width) *
                    transform.a
                )

                native_maxy = float(
                    transform.f +
                    window_y * transform.e
                )

                native_miny = float(
                    transform.f +
                    (window_y + height) *
                    transform.e
                )


                # --------------------------------------------
                # Create polygon in EPSG:32643
                # --------------------------------------------

                native_polygon = (
                    f"POLYGON(("
                    f"{native_minx} {native_miny}, "
                    f"{native_maxx} {native_miny}, "
                    f"{native_maxx} {native_maxy}, "
                    f"{native_minx} {native_maxy}, "
                    f"{native_minx} {native_miny}"
                    f"))"
                )


                # --------------------------------------------
                # Insert logical tile
                # --------------------------------------------

                connection.execute(
                    text("""
                        INSERT INTO catalog.tiles
                        (
                            scene_id,
                            acquisition_datetime,
                            resolution_m,
                            tile_row,
                            tile_col,
                            window_x,
                            window_y,
                            width,
                            height,
                            native_minx,
                            native_miny,
                            native_maxx,
                            native_maxy,
                            geom,
                            valid_pixel_pct,
                            nodata_pct,
                            cloud_pct,
                            is_derived,
                            processing_status,
                            materialized_path
                        )
                        VALUES
                        (
                            :scene_id,
                            :acquisition_datetime,
                            :resolution_m,
                            :tile_row,
                            :tile_col,
                            :window_x,
                            :window_y,
                            :width,
                            :height,
                            :native_minx,
                            :native_miny,
                            :native_maxx,
                            :native_maxy,

                            ST_Transform(
                                ST_GeomFromText(
                                    :native_polygon,
                                    32643
                                ),
                                4326
                            ),

                            NULL,
                            NULL,
                            NULL,
                            FALSE,
                            'registered',
                            NULL
                        )
                    """),
                    {
                        "scene_id": scene_id,
                        "acquisition_datetime":
                            acquisition_datetime,
                        "resolution_m":
                            resolution_m,
                        "tile_row":
                            tile_row,
                        "tile_col":
                            tile_col,
                        "window_x":
                            window_x,
                        "window_y":
                            window_y,
                        "width":
                            width,
                        "height":
                            height,
                        "native_minx":
                            native_minx,
                        "native_miny":
                            native_miny,
                        "native_maxx":
                            native_maxx,
                        "native_maxy":
                            native_maxy,
                        "native_polygon":
                            native_polygon
                    }
                )

                scene_tile_count += 1


        # ====================================================
        # 8. Scene completed
        # ====================================================

        print(
            f"Tiles created: "
            f"{scene_tile_count}"
        )

        total_tiles += scene_tile_count

        print()


# ============================================================
# 9. Finished
# ============================================================

print("==============================================")
print("TILE REGISTRATION COMPLETED")
print("==============================================")
print(
    f"Total logical tiles: {total_tiles}"
)
print("==============================================")