import os
import json
import boto3
import rasterio
import numpy as np
import cv2
import folium
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from rasterio.mask import mask
from rasterio.warp import transform_geom, transform_bounds, transform as warp_transform
from shapely.geometry import Polygon, mapping

load_dotenv()

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
RAW_DIR = OUTPUTS_DIR / "raw"
PROCESSED_DIR = OUTPUTS_DIR / "processed"
CHANGE_DIR = OUTPUTS_DIR / "change_detection"

for d in [RAW_DIR, PROCESSED_DIR, CHANGE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ACCESS_KEY = os.getenv("CDSE_S3_ACCESS_KEY")
SECRET_KEY = os.getenv("CDSE_S3_SECRET_KEY")


def get_s3_client():
    if not ACCESS_KEY or not SECRET_KEY:
        raise ValueError("CDSE S3 credentials not found in environment variables.")
    return boto3.client(
        "s3",
        endpoint_url="https://eodata.dataspace.copernicus.eu",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="default"
    )


def normalize_rgb(red, green, blue):
    """Normalizes raw surface reflectance (DN) to true natural RGB without green bias."""
    rgb = np.dstack((red, green, blue))
    max_val = 2500.0
    rgb_scaled = np.clip(rgb / max_val, 0, 1)
    return (rgb_scaled * 255).astype(np.uint8)


def mask_to_latlon_points(mask_array, transform, src_crs, max_dots=200):
    """Extracts true pixels from a mask array and converts them to (lat, lon) coordinates."""
    rows, cols = np.where(mask_array)
    if len(rows) == 0:
        return []

    # Subsample points evenly to maintain fast map rendering
    if len(rows) > max_dots:
        indices = np.linspace(0, len(rows) - 1, max_dots, dtype=int)
        rows, cols = rows[indices], cols[indices]

    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lons, lats = warp_transform(src_crs, "EPSG:4326", xs, ys)
    return list(zip(lats, lons))


def run_change_detection_pipeline(aoi_id: str, coords: list, date1: str, date2: str):
    print(f"\n[Pipeline] Starting for AOI: {aoi_id} | Dates: {date1} vs {date2}")
    
    aoi_polygon = Polygon(coords)
    geometry = [mapping(aoi_polygon)]
    dates = [date1, date2]

    scene_info = {
        date1: {
            "scene": "Sentinel-2/MSI/L2A/2024/02/28/S2B_MSIL2A_20240228T052749_N0510_R105_T43QCA_20240228T080327.SAFE/",
            "granule": "GRANULE/L2A_T43QCA_A036453_20240228T053352/IMG_DATA/R10m/",
            "prefix": "T43QCA_20240228T052749"
        },
        date2: {
            "scene": "Sentinel-2/MSI/L2A/2026/04/03/S2C_MSIL2A_20260403T052651_N0512_R105_T43QCA_20260403T104411.SAFE/",
            "granule": "GRANULE/L2A_T43QCA_A008226_20260403T053828/IMG_DATA/R10m/",
            "prefix": "T43QCA_20260403T052651"
        }
    }

    s3 = get_s3_client()
    band_names = ["B02_10m", "B03_10m", "B04_10m", "B08_10m"]

    aoi_proc_dir = PROCESSED_DIR / aoi_id
    aoi_proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download & Clip Bands
    for date, info in scene_info.items():
        scene = info["scene"]
        granule = info["granule"]
        prefix = info["prefix"]

        raw_folder = RAW_DIR / date
        raw_folder.mkdir(parents=True, exist_ok=True)
        
        date_proc_folder = aoi_proc_dir / date
        date_proc_folder.mkdir(parents=True, exist_ok=True)

        for band in band_names:
            s3_path = scene + granule + f"{prefix}_{band}.jp2"
            local_jp2 = raw_folder / f"{band}.jp2"
            local_tif = date_proc_folder / f"{band}.tif"

            if not local_jp2.exists():
                try:
                    print(f"Downloading S3 asset: {band} for {date}...")
                    s3.download_file("eodata", s3_path, str(local_jp2))
                except Exception as e:
                    print(f"[Warning] S3 download error for {band}: {e}")

            if local_jp2.exists() and not local_tif.exists():
                with rasterio.open(local_jp2) as src:
                    aoi_transformed = transform_geom("EPSG:4326", src.crs, geometry[0])
                    clipped_image, clipped_transform = mask(src, [aoi_transformed], crop=True)
                    profile = src.profile.copy()
                    profile.update({
                        "driver": "GTiff",
                        "height": clipped_image.shape[1],
                        "width": clipped_image.shape[2],
                        "transform": clipped_transform
                    })
                    with rasterio.open(local_tif, "w", **profile) as dst:
                        dst.write(clipped_image)

    # 2. Create Natural RGB & Spectral Indices
    rgb_images = {}
    raster_crs = None
    raster_bounds = None
    raster_transform = None

    for date in dates:
        date_proc_folder = aoi_proc_dir / date
        red_file = date_proc_folder / "B04_10m.tif"
        green_file = date_proc_folder / "B03_10m.tif"
        blue_file = date_proc_folder / "B02_10m.tif"
        nir_file = date_proc_folder / "B08_10m.tif"

        if red_file.exists() and green_file.exists() and blue_file.exists():
            with rasterio.open(red_file) as src:
                red = src.read(1).astype(np.float32)
                profile = src.profile.copy()
                raster_crs = src.crs
                raster_bounds = src.bounds
                raster_transform = src.transform
            with rasterio.open(green_file) as src:
                green = src.read(1).astype(np.float32)
            with rasterio.open(blue_file) as src:
                blue = src.read(1).astype(np.float32)
            with rasterio.open(nir_file) as src:
                nir = src.read(1).astype(np.float32)

            rgb_arr = normalize_rgb(red, green, blue)
            rgb_images[date] = rgb_arr

            rgb_profile = profile.copy()
            rgb_profile.update(dtype=rasterio.uint8, count=3, compress="lzw")
            with rasterio.open(date_proc_folder / "RGB.tif", "w", **rgb_profile) as dst:
                dst.write(rgb_arr[:, :, 0], 1)
                dst.write(rgb_arr[:, :, 1], 2)
                dst.write(rgb_arr[:, :, 2], 3)

            # NDVI & NDWI
            ndvi_denom = nir + red
            ndvi = np.divide(nir - red, ndvi_denom, out=np.zeros_like(nir), where=ndvi_denom != 0)
            ndwi_denom = green + nir
            ndwi = np.divide(green - nir, ndwi_denom, out=np.zeros_like(green), where=ndwi_denom != 0)

            idx_profile = profile.copy()
            idx_profile.update(dtype=rasterio.float32, count=1, compress="lzw")
            with rasterio.open(date_proc_folder / "NDVI.tif", "w", **idx_profile) as dst:
                dst.write(ndvi, 1)
            with rasterio.open(date_proc_folder / "NDWI.tif", "w", **idx_profile) as dst:
                dst.write(ndwi, 1)

    # 3. Change Detection & Morphology
    img_2024 = rgb_images[dates[0]].astype(np.float32)
    img_2026 = rgb_images[dates[1]].astype(np.float32)

    difference = np.mean(np.abs(img_2026 - img_2024), axis=2)
    change_map = np.where(difference > 25, 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    clean_map = cv2.morphologyEx(change_map, cv2.MORPH_OPEN, kernel)
    clean_map = cv2.morphologyEx(clean_map, cv2.MORPH_CLOSE, kernel)

    # 4. Category Classification
    def load_band_direct(p):
        with rasterio.open(p) as s:
            return s.read(1).astype(np.float32)

    ndvi_2024 = load_band_direct(aoi_proc_dir / dates[0] / "NDVI.tif")
    ndvi_2026 = load_band_direct(aoi_proc_dir / dates[1] / "NDVI.tif")
    ndwi_2024 = load_band_direct(aoi_proc_dir / dates[0] / "NDWI.tif")
    ndwi_2026 = load_band_direct(aoi_proc_dir / dates[1] / "NDWI.tif")

    changed = clean_map == 255
    ndvi_drop = ndvi_2024 - ndvi_2026
    ndwi_rise = ndwi_2026 - ndwi_2024

    vegetation_loss = changed & (ndvi_drop > 0.15)
    surface_water = changed & (np.abs(ndwi_rise) > 0.15)
    new_construction = changed & (~vegetation_loss) & (~surface_water)

    total_pixels = clean_map.size
    changed_pixels = int(np.sum(changed))
    pixel_area_km2 = (10 * 10) / 1_000_000

    # 5. Output Standard PNG Images
    plt.imsave(aoi_proc_dir / f"rgb_date1_{aoi_id}_{dates[0]}.png", rgb_images[dates[0]])
    plt.imsave(aoi_proc_dir / f"rgb_date2_{aoi_id}_{dates[1]}.png", rgb_images[dates[1]])

    overlay_img = rgb_images[dates[1]].copy()
    construction_mask = cv2.dilate(new_construction.astype(np.uint8), kernel, iterations=1)
    water_mask = cv2.dilate(surface_water.astype(np.uint8), kernel, iterations=1)
    overlay_img[construction_mask == 1] = [255, 0, 0]
    overlay_img[water_mask == 1] = [0, 100, 255]
    plt.imsave(aoi_proc_dir / f"change_overlay_{aoi_id}_{dates[1]}.png", overlay_img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb_images[dates[0]])
    axes[0].set_title(f"Date 1 (Before): {dates[0]}", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    axes[1].imshow(rgb_images[dates[1]])
    axes[1].set_title(f"Date 2 (After): {dates[1]}", fontsize=12, fontweight='bold')
    axes[1].axis("off")
    fig.suptitle(f"SatOrbit Change Analysis: {aoi_id.capitalize()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(aoi_proc_dir / f"comparison_{aoi_id}_{dates[1]}.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 6. Build Interactive Folium Map with Layer Checkboxes
    print("[Pipeline] Generating interactive Folium satellite map...")

    # Calculate center coordinate from raster bounds
    lon_min, lat_min, lon_max, lat_max = transform_bounds(
        raster_crs, "EPSG:4326", *raster_bounds
    )
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="Esri.WorldImagery",
    )

    # Create Feature Groups for individual layer checkboxes
    fg_aoi = folium.FeatureGroup(name="Yellow AOI Border", show=True)
    fg_red = folium.FeatureGroup(name="Red Dots (New Construction)", show=True)
    fg_blue = folium.FeatureGroup(name="Blue Dots (Surface Water)", show=True)

    # 6a. Format and add Yellow AOI Border
    raw_coords = coords[0] if (isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], list) and isinstance(coords[0][0], list)) else coords
    aoi_latlon = [[pt[1], pt[0]] for pt in raw_coords]

    folium.Polygon(
        locations=aoi_latlon,
        color="yellow",
        weight=3,
        fill=True,
        fill_color="yellow",
        fill_opacity=0.15,
        tooltip="AOI Boundary"
    ).add_to(fg_aoi)

    # 6b. Add Red Dots (New Construction)
    red_points = mask_to_latlon_points(new_construction, raster_transform, raster_crs, max_dots=200)
    for lat, lon in red_points:
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.85,
            popup="New Construction"
        ).add_to(fg_red)

    # 6c. Add Blue Dots (Surface Water)
    blue_points = mask_to_latlon_points(surface_water, raster_transform, raster_crs, max_dots=200)
    for lat, lon in blue_points:
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.85,
            popup="Surface Water Change"
        ).add_to(fg_blue)

    # Add feature groups to map
    fg_aoi.add_to(m)
    fg_red.add_to(m)
    fg_blue.add_to(m)

    # Pin Layer Control box to the TOP-LEFT corner (fully expanded)
    folium.LayerControl(position="topleft", collapsed=False).add_to(m)

    interactive_map_path = aoi_proc_dir / f"interactive_map_{aoi_id}_{dates[1]}.html"
    m.save(str(interactive_map_path))

    metrics = {
        "aoi_id": aoi_id,
        "date1": dates[0],
        "date2": dates[1],
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "change_percentage": round(float((changed_pixels / total_pixels) * 100), 2),
        "changed_area_km2": round(float(changed_pixels * pixel_area_km2), 2),
        "classification": {
            "new_construction_km2": round(float(np.sum(new_construction) * pixel_area_km2), 2),
            "vegetation_loss_km2": round(float(np.sum(vegetation_loss) * pixel_area_km2), 2),
            "surface_water_km2": round(float(np.sum(surface_water) * pixel_area_km2), 2)
        },
        "raw_image_url": f"http://localhost:8000/outputs/processed/{aoi_id}/rgb_date1_{aoi_id}_{dates[0]}.png",
        "processed_image_url": f"http://localhost:8000/outputs/processed/{aoi_id}/rgb_date2_{aoi_id}_{dates[1]}.png",
        "clean_map_url": f"http://localhost:8000/outputs/processed/{aoi_id}/change_overlay_{aoi_id}_{dates[1]}.png",
        "comparison_url": f"http://localhost:8000/outputs/processed/{aoi_id}/comparison_{aoi_id}_{dates[1]}.png",
        "map_url": f"http://localhost:8000/outputs/processed/{aoi_id}/interactive_map_{aoi_id}_{dates[1]}.html"
    }

    print(f"[Pipeline] Completed successfully for {aoi_id}!")
    return metrics