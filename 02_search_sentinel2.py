# Step 1: Import required libraries
from pystac_client import Client


# Step 2: Copernicus STAC endpoint
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"


# Step 3: Hinjewadi Phase 3 AOI
AOI = {
    "type": "Polygon",
    "coordinates": [[
        [73.65466509172526, 18.594065836721327],
        [73.65706835102091, 18.56933350549375],
        [73.70384607659673, 18.594675963992422],
        [73.70753679622932, 18.572059119244887],
        [73.65466509172526, 18.594065836721327]
    ]]
}


# Step 4: Connect to Copernicus STAC
catalog = Client.open(STAC_URL)

print("Connected to Copernicus STAC")
print()


# Step 5: Search with a limited number of results
search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=AOI,
    query={
        "eo:cloud_cover": {
            "lte": 20
        }
    },
    max_items=100
)


# Step 6: Get results
items = list(search.items())


# Step 7: Display number of scenes
print(f"Scenes found: {len(items)}")
print()


# Step 8: Display scene information
for item in items:

    properties = item.properties

    print("----------------------------------------")
    print("Scene ID :", item.id)
    print("Date     :", properties.get("datetime"))
    print("Cloud %  :", properties.get("eo:cloud_cover"))
    print("Platform :", properties.get("platform"))