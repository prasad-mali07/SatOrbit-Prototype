import json
from pystac_client import Client


# Copernicus STAC
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

# Connect
catalog = Client.open(STAC_URL)

print("Connected to Copernicus STAC")
print()


# Read selected scenes
with open("selected_scenes.json", "r", encoding="utf-8") as f:
    selected_scenes = json.load(f)


# Select 2026 scene for inspection
scene_id = selected_scenes["2026"]["scene_id"]

print("Checking scene:")
print(scene_id)
print()


# Search for the exact scene
search = catalog.search(
    collections=["sentinel-2-l2a"],
    ids=[scene_id]
)

items = list(search.items())

if not items:
    raise RuntimeError("Selected scene was not found in Copernicus STAC")


item = items[0]


# Display assets
print("==============================================")
print("AVAILABLE ASSETS")
print("==============================================")

for asset_name, asset in item.assets.items():

    print()
    print("Asset name :", asset_name)
    print("Title      :", asset.title)
    print("Media type :", asset.media_type)
    print("URL        :", asset.href)

print()
print("==============================================")
print(f"Total assets: {len(item.assets)}")
print("==============================================")