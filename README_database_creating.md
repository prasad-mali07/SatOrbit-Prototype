# SatOrbit --- Database Setup & Project Guide

This README explains how a new SatOrbit team member can reproduce the
database environment used in the `database_creating` branch on Windows.

## 1. What this branch contains

The branch contains the database and data-processing setup for SatOrbit:

``` text
Copernicus STAC
    -> Sentinel-2 scene search
    -> best scene selection per year
    -> B02/B03/B04/B08 download
    -> PostgreSQL scene registration
    -> 256x256 tile creation
    -> RemoteCLIP embeddings
    -> pgvector semantic search
    -> semantic result -> change detection
```

The repository contains code and the database schema. Large satellite
rasters, model checkpoints, and credentials stay on each developer's
machine.

## 2. Files in the pushed folder

``` text
.gitignore
01_test_copernicus.py
02_search_sentinel2.py
03_select_best_scene_per_year.py
04_check_scene_assets.py
04_download_scenes.py
05_register_scenes.py
06_create_tiles.py
07_test_remoteclip.py
08_generate_embeddings.py
09_generate_all_embeddings.py
10_test_remoteclip_text.py
11_semantic_search.py
12_visualize_search_results.py
13_semantic_to_change_detection.py
main.py
selected_scenes.json
satorbit_01_schema.sql
```

### Script-by-script explanation

**`01_test_copernicus.py`**\
Tests access to the Copernicus Data Space Ecosystem STAC service.

**`02_search_sentinel2.py`**\
Searches Sentinel-2 L2A scenes using the SatOrbit AOI, date, collection
and cloud-cover constraints.

**`03_select_best_scene_per_year.py`**\
Chooses one suitable scene for each requested year and writes the
selection to `selected_scenes.json`.

**`04_check_scene_assets.py`**\
Inspects the assets available for a selected scene. The main SatOrbit
workflow uses the 10 m bands B02, B03, B04 and B08.

**`04_download_scenes.py`**\
Downloads the selected 10 m bands to local storage.

**`05_register_scenes.py`**\
Registers scenes and their band assets in PostgreSQL/PostGIS.

**`06_create_tiles.py`**\
Creates logical 256x256 pixel tile windows and stores their metadata and
geometry in `catalog.tiles`.

**`07_test_remoteclip.py`**\
Checks RemoteCLIP image loading, preprocessing and 512-dimensional
embedding generation.

**`08_generate_embeddings.py`**\
A smaller embedding-generation/test workflow.

**`09_generate_all_embeddings.py`**\
Generates and stores RemoteCLIP embeddings for many tiles. It is
resume-safe and skips embeddings that already exist.

**`10_test_remoteclip_text.py`**\
Tests converting a text query such as `urban construction` into a
512-dimensional RemoteCLIP text embedding.

**`11_semantic_search.py`**\
Uses pgvector cosine similarity to find satellite tiles whose image
embeddings are close to the text embedding.

**`12_visualize_search_results.py`**\
Creates RGB previews of semantic-search results for visual inspection.

**`13_semantic_to_change_detection.py`**\
Connects semantic search to temporal change detection: relevant tile -\>
scene_id -\> complete scene pair -\> B02/B03/B04/B08 -\> Siamese U-Net
-\> change map.

**`main.py`**\
Runs the basic FastAPI application and database connectivity test.

**`selected_scenes.json`**\
Stores the selected Sentinel-2 scenes so later steps can reuse the same
scene choices.

**`satorbit_01_schema.sql`**\
Creates the SatOrbit database schemas, tables, indexes and related
database objects.

## 3. What must NOT be pushed to GitHub

Do not commit:

``` text
.env
*.pth
*.pt
*.ckpt
*.safetensors
*.jp2
*.tif
*.tiff
SATORBITDATA/
RemoteCLIP/checkpoints/
```

The `.env` file contains credentials. Satellite imagery and ML
checkpoints are large local files.

If a credential is accidentally exposed, rotate/revoke it immediately.

## 4. Required software

Recommended versions matching the current development environment:

  Software        Version
  --------------- -------------------------------------------
  Windows         10/11 64-bit
  PostgreSQL      16.x
  PostGIS         3.6.x
  pgvector        0.8.6
  Python          3.12.x
  Git             2.x
  Visual Studio   Community with C++ tools
  FastAPI         0.115.0
  OpenCLIP        3.3.0
  PyTorch         2.14.0+cpu in the current CPU environment

## 5. Install Git

Download Git for Windows:

https://git-scm.com/download/win

Verify:

``` cmd
git --version
```

## 6. Install PostgreSQL 16

Official Windows download:

https://www.postgresql.org/download/windows/

Install PostgreSQL 16 through the Windows installer.

Use the normal installation path:

``` text
C:\Program Files\PostgreSQL\16
```

Keep port:

``` text
5432
```

Remember the password created for the `postgres` user.

Verify:

``` cmd
"C:\Program Files\PostgreSQL\16\bin\psql.exe" --version
```

## 7. Install PostGIS

PostGIS provides spatial functionality used by SatOrbit for AOIs, scene
footprints, tile polygons and spatial queries.

Official Windows guide:

https://postgis.net/documentation/getting_started/install_windows/

The easiest Windows method is StackBuilder:

1.  Open StackBuilder.
2.  Select the PostgreSQL 16 installation.
3.  Select **Spatial Extensions**.
4.  Install the compatible PostGIS bundle.
5.  Finish installation.

Create the database:

``` cmd
"C:\Program Files\PostgreSQL\16\bin\psql.exe" -U postgres
```

Then:

``` sql
CREATE DATABASE satorbit;
\c satorbit
CREATE EXTENSION IF NOT EXISTS postgis;
SELECT PostGIS_Version();
```

## 8. Install Visual Studio C++ tools

pgvector is compiled on Windows.

Download:

https://visualstudio.microsoft.com/downloads/

Install Visual Studio Community and include C++ development tools, MSVC,
Windows SDK and the x64 build tools.

Open:

``` text
x64 Native Tools Command Prompt for VS
```

Run it as Administrator.

Verify:

``` cmd
nmake /?
```

## 9. Install pgvector 0.8.6

Official project:

https://github.com/pgvector/pgvector

SatOrbit's current setup uses pgvector 0.8.6.

In the Administrator x64 Native Tools Command Prompt:

``` cmd
set "PGROOT=C:\Program Files\PostgreSQL\16"
cd %TEMP%
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

If the `pgvector` folder already exists, do not clone it again; enter
the existing folder and run the build/install commands.

Then enable the extension:

``` cmd
"C:\Program Files\PostgreSQL\16\bin\psql.exe" -U postgres -d satorbit
```

``` sql
CREATE EXTENSION IF NOT EXISTS vector;

SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('postgis', 'vector');
```

Expected result is a PostGIS 3.x version and vector 0.8.6.

## 10. Clone the SatOrbit repository

``` cmd
cd C:\
git clone https://github.com/prasad-mali07/SatOrbit-Prototype.git
cd SatOrbit-Prototype
git fetch origin
git switch database_creating
```

If the branch is not available locally:

``` cmd
git fetch origin
git switch -c database_creating --track origin/database_creating
```

Verify:

``` cmd
git branch
```

You should see:

``` text
* database_creating
```

## 11. Create the local data folders

The satellite data is intentionally outside Git.

Run:

``` cmd
mkdir C:\SATORBITDATA
mkdir C:\SATORBITDATA\scenes
mkdir C:\SATORBITDATA\scenes\sentinel-2
mkdir C:\SATORBITDATA\tiles
mkdir C:\SATORBITDATA\tiles\sentinel-2
mkdir C:\SATORBITDATA\derived
mkdir C:\SATORBITDATA\derived\change_maps
mkdir C:\SATORBITDATA\derived\cloud_masks
mkdir C:\SATORBITDATA\derived\indices
mkdir C:\SATORBITDATA\temp
```

Structure:

``` text
C:\SATORBITDATA
├── derived
│   ├── change_maps
│   ├── cloud_masks
│   └── indices
├── scenes
│   └── sentinel-2
├── temp
└── tiles
    └── sentinel-2
```

## 12. Create the Python environment

From the project folder:

``` cmd
cd C:\SatOrbit-Prototype
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install the API/database packages:

``` cmd
pip install fastapi==0.115.0 uvicorn sqlalchemy psycopg[binary] python-dotenv
```

Install the raster/ML packages required by the scripts:

``` cmd
pip install numpy pillow rasterio torch torchvision open_clip_torch
```

If the team later standardizes a `requirements.txt`, use that file
instead so everyone gets identical package versions.

## 13. Create `.env`

Create:

``` text
C:\SatOrbit-Prototype\.env
```

Example:

``` env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/satorbit
CDSE_S3_ACCESS_KEY=YOUR_ACCESS_KEY
CDSE_S3_SECRET_KEY=YOUR_SECRET_KEY
```

Replace the placeholders with the developer's own values.

Never commit `.env`.

Check:

``` cmd
git status
```

`.env` should not be listed as a file to commit.

## 14. Load the SatOrbit schema

Connect first:

``` cmd
"C:\Program Files\PostgreSQL\16\bin\psql.exe" -U postgres -d satorbit
```

Enable extensions:

``` sql
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

Load the schema:

``` cmd
"C:\Program Files\PostgreSQL\16\bin\psql.exe" -U postgres -d satorbit -f satorbit_01_schema.sql
```

## 15. Verify the database

Connect:

``` cmd
"C:\Program Files\PostgreSQL\16\bin\psql.exe" -U postgres -d satorbit
```

Check extensions:

``` sql
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('postgis', 'vector');
```

Check schemas:

``` sql
\dn
```

Expected project schemas:

``` text
catalog
semantic
analysis
ops
```

Check tables:

``` sql
\dt catalog.*
\dt semantic.*
\dt analysis.*
\dt ops.*
```

A fresh installation will have zero scenes/tiles until the processing
scripts are run.

## 16. SatOrbit database design

### `catalog`

Contains source, model, AOI, scene, asset and tile metadata.

Main tables:

``` text
catalog.data_sources
catalog.ml_models
catalog.satellite_scenes
catalog.scene_assets
catalog.aoi
catalog.tiles
```

### `semantic`

Contains semantic information and vector embeddings:

``` text
semantic.tile_semantics
semantic.tile_embeddings
```

The current embedding column is:

``` text
vector(512)
```

The current RemoteCLIP model is represented by `model_id = 2`.

The embedding table uses a cosine-distance HNSW index for the embedding
model.

### `analysis`

Contains temporal change-analysis structures:

``` text
analysis.tile_pairs
analysis.change_classes
analysis.change_results
analysis.change_features
```

### `ops`

Contains operational/application tracking:

``` text
ops.app_users
ops.ingestion_jobs
ops.ingestion_items
ops.processing_jobs
ops.search_log
```

## 17. Scene and tile workflow

The current selected scene set covers:

``` text
2016
2017
2018
2019
2020
2021
2022
2023
2024
2025
2026
```

The current development selection contains 11 scenes.

Each selected Sentinel-2 scene uses these 10 m bands:

``` text
B02_10m.jp2
B03_10m.jp2
B04_10m.jp2
B08_10m.jp2
```

A scene is approximately:

``` text
10980 x 10980 pixels
```

The database tile size is:

``` text
256 x 256 pixels
```

For a full 10980 x 10980 scene, the current tile grid is:

``` text
43 x 43 = 1849 tiles
```

For 11 scenes:

``` text
11 x 1849 = 20339 tiles
```

## 18. Run the scripts in order

Recommended order:

``` text
01_test_copernicus.py
        ↓
02_search_sentinel2.py
        ↓
03_select_best_scene_per_year.py
        ↓
04_check_scene_assets.py
        ↓
04_download_scenes.py
        ↓
05_register_scenes.py
        ↓
06_create_tiles.py
        ↓
07_test_remoteclip.py
        ↓
09_generate_all_embeddings.py
        ↓
10_test_remoteclip_text.py
        ↓
11_semantic_search.py
        ↓
12_visualize_search_results.py
        ↓
13_semantic_to_change_detection.py
```

`08_generate_embeddings.py` is useful as a smaller embedding test;
`09_generate_all_embeddings.py` is the bulk/resume-safe workflow.

## 19. What each major stage does

### Stage A --- Satellite discovery

``` text
Copernicus STAC
    -> search
    -> select scenes
    -> selected_scenes.json
```

### Stage B --- Local satellite archive

``` text
CDSE
    -> B02/B03/B04/B08
    -> C:\SATORBITDATA\scenes\sentinel-2\YEAR
```

### Stage C --- Database registration

``` text
local JP2 files
    -> catalog.satellite_scenes
    -> catalog.scene_assets
```

### Stage D --- Tile creation

``` text
full scene
    -> 256x256 windows
    -> catalog.tiles
```

### Stage E --- Semantic embeddings

RemoteCLIP uses an RGB representation:

``` text
B04 = Red
B03 = Green
B02 = Blue
```

The output embedding has:

``` text
512 dimensions
```

It is stored in:

``` text
semantic.tile_embeddings
```

### Stage F --- Semantic search

``` text
"urban construction"
        ↓
RemoteCLIP text encoder
        ↓
512-D vector
        ↓
pgvector cosine similarity
        ↓
top matching tiles
```

### Stage G --- Change detection

The semantic result provides a relevant tile and its `scene_id`.

The intended workflow then uses that scene ID to identify the complete
satellite scene and compare two dates:

``` text
Relevant tile
    ↓
scene_id
    ↓
Before scene + After scene
    ↓
B02 B03 B04 B08 for both dates
    ↓
Siamese U-Net
    ↓
change probability/map
```

RemoteCLIP is therefore used to locate relevant areas, while the
change-detection model analyzes temporal change.

## 20. FastAPI test

Run:

``` cmd
uvicorn main:app --reload
```

Open:

``` text
http://127.0.0.1:8000/
```

Expected:

``` json
{"message":"SatOrbit API is running"}
```

Then open:

``` text
http://127.0.0.1:8000/database-test
```

Expected on a fresh schema:

``` json
{"database":"connected","tile_count":0}
```

The tile count becomes non-zero after scene/tile registration.

## 21. Useful database verification queries

Scenes:

``` sql
SELECT COUNT(*) FROM catalog.satellite_scenes;
```

Scene assets:

``` sql
SELECT COUNT(*) FROM catalog.scene_assets;
```

Tiles:

``` sql
SELECT COUNT(*) FROM catalog.tiles;
```

Embeddings:

``` sql
SELECT COUNT(*)
FROM semantic.tile_embeddings
WHERE model_id = 2;
```

Embedding dimensions:

``` sql
SELECT vector_dims(embedding) AS dimensions, COUNT(*)
FROM semantic.tile_embeddings
WHERE model_id = 2
GROUP BY vector_dims(embedding);
```

Expected embedding dimension:

``` text
512
```

## 22. Current development numbers

The current development environment has:

``` text
Selected scenes:       11
Band assets:           44
Tiles:                 20,339
Embedding dimension:   512
Development embeddings:15,068
```

The 15,068 embedding count is a partial development run, not a
requirement that a new member must reproduce exactly before testing the
system.

## 23. RemoteCLIP

The project currently uses:

``` text
RemoteCLIP ViT-B/32
```

The checkpoint is large and is intentionally excluded from Git.

Each developer must download/configure the checkpoint locally and update
the local path expected by the scripts.

Do not commit the checkpoint.

## 24. Change-detection checkpoint

The current development environment uses:

``` text
C:\model_prediction\best_model.pth
```

This checkpoint is also intentionally excluded from Git.

Each developer who needs to run change detection must obtain the
approved project checkpoint separately.

## 25. Troubleshooting

### `psql is not recognized`

Use:

``` cmd
"C:\Program Files\PostgreSQL\16\bin\psql.exe" --version
```

### PostGIS extension is unavailable

Install the PostGIS bundle compatible with PostgreSQL 16 using
StackBuilder, then restart/reopen the PostgreSQL connection.

### pgvector extension is unavailable

Check:

``` cmd
set "PGROOT=C:\Program Files\PostgreSQL\16"
```

Then, from the Administrator x64 Native Tools Command Prompt:

``` cmd
cd %TEMP%\pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

Then:

``` sql
CREATE EXTENSION vector;
```

### pgvector build cannot find `postgres.h`

Make sure `PGROOT` points to the actual PostgreSQL 16 installation:

``` cmd
set "PGROOT=C:\Program Files\PostgreSQL\16"
```

### pgvector installation says Access Denied

Use the x64 Native Tools Command Prompt as Administrator and rerun:

``` cmd
nmake /F Makefile.win install
```

### Database connection refused

Check that PostgreSQL is running and test:

``` cmd
"C:\Program Files\PostgreSQL\16\bin\pg_isready.exe"
```

### `DATABASE_URL` is missing

Check `.env`:

``` env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/satorbit
```

### RemoteCLIP checkpoint not found

The checkpoint is not stored in Git. Download it separately and
configure the local path.

## 26. Team Git workflow

Before starting:

``` cmd
git switch database_creating
git pull
```

Check:

``` cmd
git status
```

After making a code/schema/documentation change:

``` cmd
git add .
git commit -m "Describe the change"
git push
```

For a separate feature:

``` cmd
git switch -c your_feature_name
git push -u origin your_feature_name
```

Do not use `git push --force` on the shared branch unless the team
explicitly agrees.

## 27. Security checklist

Before every commit:

``` cmd
git status
```

Make sure these are not staged:

``` text
.env
CDSE credentials
*.pth
*.pt
*.ckpt
*.jp2
*.tif
SATORBITDATA/
RemoteCLIP/checkpoints/
```

Never paste CDSE access keys, secret keys or database passwords into
GitHub issues, README files or source code.

## 28. Quick new-member checklist

``` text
[ ] Install Git
[ ] Install PostgreSQL 16
[ ] Install PostGIS
[ ] Install Visual Studio C++ tools
[ ] Build/install pgvector 0.8.6
[ ] Create satorbit database
[ ] Enable postgis
[ ] Enable vector
[ ] Clone SatOrbit
[ ] Switch to database_creating
[ ] Create Python virtual environment
[ ] Install Python dependencies
[ ] Create local .env
[ ] Create C:\SATORBITDATA
[ ] Load satorbit_01_schema.sql
[ ] Verify PostgreSQL extensions
[ ] Verify SatOrbit tables
[ ] Test FastAPI
[ ] Configure CDSE credentials
[ ] Download satellite data
[ ] Register scenes
[ ] Create tiles
[ ] Configure RemoteCLIP
[ ] Generate embeddings
[ ] Test semantic search
[ ] Configure change-detection checkpoint if needed
```

## 29. Official references

PostgreSQL Windows: https://www.postgresql.org/download/windows/

PostGIS Windows:
https://postgis.net/documentation/getting_started/install_windows/

pgvector: https://github.com/pgvector/pgvector

Git: https://git-scm.com/download/win

SatOrbit repository: https://github.com/prasad-mali07/SatOrbit-Prototype

## 30. Final architecture

``` text
                         SAT ORBIT
                             |
             +---------------+---------------+
             |                               |
             v                               v
       Satellite Data                    User Query
             |                               |
             v                               v
     Sentinel-2 / STAC                  RemoteCLIP
             |                               |
             v                               v
       Local JP2 Files                  512-D Vector
             |                               |
             +---------------+---------------+
                             |
                             v
                  PostgreSQL + PostGIS
                             |
                             v
                           Tiles
                             |
                             v
                      pgvector search
                             |
                             v
                      Relevant Tile
                             |
                             v
                          scene_id
                             |
                             v
                   Complete Scene Pair
                             |
                             v
                     B02/B03/B04/B08
                             |
                             v
                       Siamese U-Net
                             |
                             v
                    Change Map / Results
```

This README is the shared setup document for the SatOrbit
`database_creating` branch.
