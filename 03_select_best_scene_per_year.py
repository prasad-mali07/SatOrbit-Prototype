# Step 1: Import required libraries
from pystac_client import Client
import json


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


# Step 5: Dictionary to store selected scenes
best_scenes = {}


# Step 6: Search year by year
for year in range(2015, 2027):

    print(f"Searching {year}...")

    start_date = f"{year}-01-01T00:00:00Z"
    end_date = f"{year}-12-31T23:59:59Z"

    try:

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=AOI,
            datetime=f"{start_date}/{end_date}",
            query={
                "eo:cloud_cover": {
                    "lte": 20
                }
            },
            max_items=100
        )

        items = list(search.items())

        if not items:
            print(f"  No scenes found for {year}")
            continue


        # Step 7: Select scene with lowest cloud cover
        best_scene = min(
            items,
            key=lambda item: item.properties.get(
                "eo:cloud_cover", 100
            )
        )


        # Step 8: Store useful information
        best_scenes[str(year)] = {
            "scene_id": best_scene.id,
            "datetime": best_scene.properties.get("datetime"),
            "cloud_cover": best_scene.properties.get("eo:cloud_cover"),
            "platform": best_scene.properties.get("platform"),
            "collection": "sentinel-2-l2a"
        }


        print(f"  Scenes found : {len(items)}")
        print(f"  Best scene   : {best_scene.id}")
        print(f"  Date         : {best_scene.properties.get('datetime')}")
        print(f"  Cloud cover  : {best_scene.properties.get('eo:cloud_cover')}%")


    except Exception as e:

        print(f"  Error searching {year}: {e}")

    print()


# Step 9: Save selected scenes to JSON
output_file = "selected_scenes.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(best_scenes, f, indent=4)


# Step 10: Display final results
print("==============================================")
print("BEST SENTINEL-2 SCENE FOR EACH YEAR")
print("==============================================")


for year in sorted(best_scenes):

    scene = best_scenes[year]

    print()
    print("----------------------------------------------")
    print("Year       :", year)
    print("Scene ID   :", scene["scene_id"])
    print("Date       :", scene["datetime"])
    print("Cloud %    :", scene["cloud_cover"])
    print("Platform   :", scene["platform"])


print()
print("==============================================")
print("Total years with scenes:", len(best_scenes))
print("==============================================")

print()
print(f"Selected scenes saved to: {output_file}")