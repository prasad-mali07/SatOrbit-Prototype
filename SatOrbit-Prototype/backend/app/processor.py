import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "outputs" / "processed"


def process_change_detection(
    raw_path_t1: Path, raw_path_t2: Path, aoi_id: str, date2: str
) -> dict:
    """
    Reads raw satellite images from Date 1 and Date 2, calculates absolute pixel delta,
    applies thresholding (thresh=30) and morphological filtering, and saves output artifacts.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    place_name = aoi_id.lower().replace(" ", "_")
    year2 = date2.split("-")[0]

    processed_filename = f"processed_{place_name}_{date2}_{year2}.png"
    clean_map_filename = f"clean_change_map_{place_name}_{date2}_{year2}.png"
    comparison_filename = f"comparison_{place_name}_{date2}_{year2}.png"

    # 1. Load raw images from disk
    img1 = cv2.imread(str(raw_path_t1))
    img2 = cv2.imread(str(raw_path_t2))

    if img1 is None or img2 is None:
        raise ValueError("Failed to load raw image files for processing.")

    # Resize img2 to match img1 dimensions if resolutions differ
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 2. Convert to Grayscale & Calculate Pixel Delta
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)

    # 3. Apply Thresholding (thresh=30) & Morphological Noise Cleaning (5x5 kernel)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4. Generate Side-by-Side Comparison Plot (Img1 | Img2 | Mask)
    mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    comparison_plot = np.hstack((img1, img2, mask_bgr))

    # 5. Save Output Images to data/outputs/processed/
    cv2.imwrite(str(PROCESSED_DIR / processed_filename), img2)
    cv2.imwrite(str(PROCESSED_DIR / clean_map_filename), clean_mask)
    cv2.imwrite(str(PROCESSED_DIR / comparison_filename), comparison_plot)

    # 6. Calculate Area & Pixel Metrics
    total_pixels = int(img1.shape[0] * img1.shape[1])
    changed_pixels = int(np.count_nonzero(clean_mask))
    change_percentage = round((changed_pixels / total_pixels) * 100, 2)
    changed_area_km2 = round(changed_pixels * 0.0001, 4)

    return {
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "change_percentage": change_percentage,
        "changed_area_km2": changed_area_km2,
        "filenames": {
            "processed": processed_filename,
            "clean_map": clean_map_filename,
            "comparison": comparison_filename,
        },
    }