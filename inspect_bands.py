import os
import rasterio
import numpy as np

BASE = r"C:\deep_learning_model\data\OSCD"

MUMBAI = os.path.join(
    BASE,
    "Onera Satellite Change Detection dataset - Images",
    "Onera Satellite Change Detection dataset - Images",
    "mumbai"
)

bands = ["B02.tif", "B03.tif", "B04.tif", "B08.tif"]

for date_folder in ["imgs_1_rect", "imgs_2_rect"]:

    print("\n==============================")
    print(date_folder)
    print("==============================")

    folder = os.path.join(MUMBAI, date_folder)

    for band in bands:

        path = os.path.join(folder, band)

        with rasterio.open(path) as src:
            image = src.read(1)

            print(
                f"{band}: "
                f"shape={image.shape}, "
                f"dtype={image.dtype}, "
                f"min={image.min()}, "
                f"max={image.max()}"
            )

print("\nBand inspection complete!")