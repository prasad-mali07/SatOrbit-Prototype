-- =====================================================================
-- SatOrbit database schema  v1.0
-- Target : PostgreSQL 16+ (17 / 18 recommended), PostGIS 3.4+, pgvector 0.8+
-- Run as : the database owner (role satorbit_owner) inside database "satorbit"
-- Notes  : * Values marked  -- [DEPENDS ON EMBEDDING MODEL]  must be confirmed
--            after the embedding-model experiment (see document section 15).
--          * Everything here was executed against PostgreSQL 16 + PostGIS 3.4
--            + pgvector 0.8.0 before being included in the documentation.
-- =====================================================================

\set ON_ERROR_STOP on

-- ---------------------------------------------------------------------
-- 0. Database creation (run once, as a superuser, from the "postgres" DB)
-- ---------------------------------------------------------------------
-- CREATE ROLE satorbit_owner    LOGIN PASSWORD 'change-me';
-- CREATE ROLE satorbit_app      LOGIN PASSWORD 'change-me';   -- FastAPI + workers
-- CREATE ROLE satorbit_readonly LOGIN PASSWORD 'change-me';   -- BI / QGIS / analysts
-- CREATE DATABASE satorbit OWNER satorbit_owner
--        ENCODING 'UTF8' TEMPLATE template0;
-- \connect satorbit

-- ---------------------------------------------------------------------
-- 1. Extensions  (superuser, or a role allowed to create trusted extensions)
-- ---------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS postgis;   -- geometry types, GiST spatial ops
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector: vector, halfvec, HNSW/IVFFlat
-- Recommended for monitoring (needs shared_preload_libraries = 'pg_stat_statements'):
-- CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
-- NOT installed on purpose: postgis_raster (pixels stay out of the DB), postgis_topology,
-- pgcrypto (gen_random_uuid() is built in since PostgreSQL 13).

-- ---------------------------------------------------------------------
-- 2. Schemas
-- ---------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS catalog;   -- registries + scenes + tiles + AOI
CREATE SCHEMA IF NOT EXISTS semantic;  -- tile semantics + embeddings
CREATE SCHEMA IF NOT EXISTS analysis;  -- temporal pairs + change detection outputs
CREATE SCHEMA IF NOT EXISTS ops;       -- jobs, users, logs

SET search_path = catalog, semantic, analysis, ops, public;

-- ---------------------------------------------------------------------
-- 3. Helper: updated_at trigger function
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ops.set_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END $$;

-- =====================================================================
-- 4. REFERENCE / REGISTRY TABLES
-- =====================================================================

-- 4.1 data_sources : one row per imagery family (lookup, grows without migrations)
CREATE TABLE catalog.data_sources (
    source_id            smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code                 text        NOT NULL UNIQUE
                         CHECK (code ~ '^[A-Z0-9_]+$'),
    display_name         text        NOT NULL,
    provider             text,
    modality             text        NOT NULL CHECK (modality IN ('optical','sar')),
    stac_collection      text,
    default_band_config  jsonb       NOT NULL DEFAULT '{}'::jsonb,
    notes                text,
    is_active            boolean     NOT NULL DEFAULT true,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now()
);

-- 4.2 ml_models : registry of every model / rule-set / dataset-derivation that writes results
CREATE TABLE catalog.ml_models (
    model_id       integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name           text        NOT NULL,
    version        text        NOT NULL,
    task           text        NOT NULL CHECK (task IN
                   ('change_detection','change_classification','embedding',
                    'zero_shot_labeling','captioning','land_cover','spectral_rules')),
    architecture   text,                              -- e.g. 'siamese_unet', 'clip_vit_b32'
    framework      text        NOT NULL DEFAULT 'pytorch',
    embedding_dim  integer     CHECK (embedding_dim > 0),   -- only for task='embedding'
    weights_path   text,                              -- relative to MODEL_ROOT, never absolute
    weights_sha256 text        CHECK (weights_sha256 ~ '^[0-9a-f]{64}$'),
    input_spec     jsonb       NOT NULL DEFAULT '{}'::jsonb, -- bands, patch size, normalisation
    training_data  text,                              -- e.g. 'OSCD'
    metrics        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    license        text,
    is_active      boolean     NOT NULL DEFAULT true,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (name, version),
    CHECK (task <> 'embedding' OR embedding_dim IS NOT NULL)
);

-- 4.3 app_users (optional; keep if the UI has login / review workflow)
CREATE TABLE ops.app_users (
    user_id       uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    username      text        NOT NULL UNIQUE,
    display_name  text,
    role          text        NOT NULL DEFAULT 'analyst'
                  CHECK (role IN ('admin','analyst','viewer')),
    password_hash text,                                -- NULL when auth is external
    is_active     boolean     NOT NULL DEFAULT true,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

-- 4.4 ingestion_jobs (created before scenes because scenes reference it)
CREATE TABLE ops.ingestion_jobs (
    job_id         uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id      smallint    REFERENCES catalog.data_sources(source_id),
    mode           text        NOT NULL
                   CHECK (mode IN ('online_download','local_import','incremental_sync')),
    status         text        NOT NULL DEFAULT 'queued'
                   CHECK (status IN ('queued','running','paused_offline','completed',
                                     'completed_with_errors','failed','cancelled')),
    params         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    items_total    integer     NOT NULL DEFAULT 0 CHECK (items_total  >= 0),
    items_done     integer     NOT NULL DEFAULT 0 CHECK (items_done   >= 0),
    items_failed   integer     NOT NULL DEFAULT 0 CHECK (items_failed >= 0),
    requested_by   uuid        REFERENCES ops.app_users(user_id) ON DELETE SET NULL,
    error          text,
    started_at     timestamptz,
    finished_at    timestamptz,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

-- =====================================================================
-- 5. CATALOG: scenes, assets, AOI, tiles
-- =====================================================================

-- 5.1 satellite_scenes : the original acquisition (one row per product)
CREATE TABLE catalog.satellite_scenes (
    scene_id             uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id            smallint    NOT NULL REFERENCES catalog.data_sources(source_id),
    stac_id              text,                         -- STAC Item id (if the source is STAC)
    product_id           text        NOT NULL,         -- provider product / granule id
    satellite            text        NOT NULL,         -- 'Sentinel-2B', 'Landsat 9', ...
    sensor               text        NOT NULL,         -- 'MSI', 'OLI', 'C-SAR', ...
    acquisition_datetime timestamptz NOT NULL,
    cloud_cover          real        CHECK (cloud_cover BETWEEN 0 AND 100),  -- NULL for SAR
    processing_level     text        NOT NULL,         -- 'L2A', 'L1C', 'GRD', 'C2L2', ...
    processing_baseline  text,                         -- e.g. Sentinel-2 '05.11'
    crs_epsg             integer     NOT NULL CHECK (crs_epsg > 0),
    resolution_m         numeric(8,3) NOT NULL CHECK (resolution_m > 0),
    width_px             integer     NOT NULL CHECK (width_px  > 0),
    height_px            integer     NOT NULL CHECK (height_px > 0),
    transform            double precision[] NOT NULL
                         CHECK (array_length(transform, 1) = 6),   -- affine a,b,c,d,e,f
    grid_signature       text        NOT NULL,         -- hash(crs, transform, width, height)
    footprint            geometry(MultiPolygon, 4326) NOT NULL
                         CHECK (ST_IsValid(footprint)),
    storage_backend      text        NOT NULL DEFAULT 'local'
                         CHECK (storage_backend IN ('local','s3')),
    raster_path          text        NOT NULL,         -- RELATIVE path or object key (ML-ready COG)
    band_map             jsonb       NOT NULL,         -- {"B02":1,"B03":2,"B04":3,"B08":4}
    raster_size_bytes    bigint      CHECK (raster_size_bytes >= 0),
    raster_sha256        text        CHECK (raster_sha256 ~ '^[0-9a-f]{64}$'),
    ingestion_status     text        NOT NULL DEFAULT 'registered'
                         CHECK (ingestion_status IN ('registered','tiling','ready','failed','archived')),
    stac_properties      jsonb       NOT NULL DEFAULT '{}'::jsonb,
    ingestion_job_id     uuid        REFERENCES ops.ingestion_jobs(job_id) ON DELETE SET NULL,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_scene_product UNIQUE (source_id, product_id),
    -- superkey used by tiles' composite FK (keeps denormalised tile columns honest)
    CONSTRAINT uq_scene_time_res UNIQUE (scene_id, acquisition_datetime, resolution_m)
);
COMMENT ON COLUMN catalog.satellite_scenes.raster_path IS
  'Path relative to the storage root (backend=local) or object key (backend=s3). Never store absolute paths.';

-- 5.2 scene_assets : auxiliary rasters/files that belong to a scene
CREATE TABLE catalog.scene_assets (
    asset_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scene_id         uuid        NOT NULL REFERENCES catalog.satellite_scenes(scene_id) ON DELETE CASCADE,
    role             text        NOT NULL CHECK (role IN
                     ('bands_stack','band','scl','cloud_mask','preview','thumbnail','metadata')),
    band_name        text,                            -- 'B02', 'SCL', 'VV', ...
    storage_backend  text        NOT NULL DEFAULT 'local' CHECK (storage_backend IN ('local','s3')),
    raster_path      text        NOT NULL,
    media_type       text        NOT NULL DEFAULT 'image/tiff; application=geotiff; profile=cloud-optimized',
    dtype            text,                            -- 'uint16', 'uint8', 'float32'
    nodata_value     double precision,
    width_px         integer     CHECK (width_px  > 0),
    height_px        integer     CHECK (height_px > 0),
    resolution_m     numeric(8,3) CHECK (resolution_m > 0),
    block_size       integer     CHECK (block_size > 0),   -- internal COG tile size
    overview_levels  integer[],
    compression      text,
    size_bytes       bigint      CHECK (size_bytes >= 0),
    sha256           text        CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    created_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_asset_role UNIQUE NULLS NOT DISTINCT (scene_id, role, band_name)
);

-- 5.3 aoi : areas of interest (user-drawn, administrative, imported)
CREATE TABLE catalog.aoi (
    aoi_id       uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    name         text        NOT NULL,
    description  text,
    kind         text        NOT NULL DEFAULT 'user_drawn'
                 CHECK (kind IN ('user_drawn','admin_boundary','imported','system')),
    geom         geometry(MultiPolygon, 4326) NOT NULL CHECK (ST_IsValid(geom)),
    area_km2     double precision GENERATED ALWAYS AS (ST_Area(geom::geography) / 1000000.0) STORED,
    created_by   uuid        REFERENCES ops.app_users(user_id) ON DELETE SET NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- 5.4 tiles : logical processing windows inside a scene (NO pixels, NO per-tile file by default)
CREATE TABLE catalog.tiles (
    tile_id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scene_id             uuid         NOT NULL,
    acquisition_datetime timestamptz  NOT NULL,        -- copied from scene (kept honest by FK)
    resolution_m         numeric(8,3) NOT NULL,        -- copied from scene (kept honest by FK)
    tile_row             integer      CHECK (tile_row >= 0),
    tile_col             integer      CHECK (tile_col >= 0),
    window_x             integer      NOT NULL CHECK (window_x >= 0),   -- pixel column offset
    window_y             integer      NOT NULL CHECK (window_y >= 0),   -- pixel row offset
    width                integer      NOT NULL CHECK (width  BETWEEN 1 AND 4096),
    height               integer      NOT NULL CHECK (height BETWEEN 1 AND 4096),
    native_minx          double precision NOT NULL,    -- bounds in the scene CRS (metres for UTM)
    native_miny          double precision NOT NULL,
    native_maxx          double precision NOT NULL,
    native_maxy          double precision NOT NULL,
    geom                 geometry(Polygon, 4326) NOT NULL CHECK (ST_IsValid(geom)),
    valid_pixel_pct      real         CHECK (valid_pixel_pct BETWEEN 0 AND 100),  -- usable pixels (not NoData, not cloud/shadow)
    nodata_pct           real         CHECK (nodata_pct      BETWEEN 0 AND 100),  -- pixels equal to NoData (outside swath)
    cloud_pct            real         CHECK (cloud_pct BETWEEN 0 AND 100),
    is_derived           boolean      NOT NULL DEFAULT false,   -- true = window derived from another scene's tile (temporal matching)
    processing_status    text         NOT NULL DEFAULT 'registered'
                         CHECK (processing_status IN
                         ('registered','stats_done','semantic_done','embedded','ready','excluded','failed')),
    materialized_path    text,                          -- NULL unless a physical patch file was exported
    created_at           timestamptz  NOT NULL DEFAULT now(),
    updated_at           timestamptz  NOT NULL DEFAULT now(),
    CONSTRAINT fk_tiles_scene FOREIGN KEY (scene_id, acquisition_datetime, resolution_m)
        REFERENCES catalog.satellite_scenes (scene_id, acquisition_datetime, resolution_m)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT uq_tile_window UNIQUE (scene_id, window_x, window_y, width, height),
    CONSTRAINT ck_tile_pct    CHECK (valid_pixel_pct + nodata_pct <= 100.01),
    CONSTRAINT ck_tile_bounds CHECK (native_maxx > native_minx AND native_maxy > native_miny),
    CONSTRAINT ck_tile_rowcol CHECK ((tile_row IS NULL) = (tile_col IS NULL)),
    -- stats may be NULL only while the tile is still 'registered' (statistics job not run yet)
    CONSTRAINT ck_tile_stats  CHECK (processing_status IN ('registered','excluded','failed')
                                     OR (valid_pixel_pct IS NOT NULL AND nodata_pct IS NOT NULL))
);

-- =====================================================================
-- 6. OPS: processing jobs (needs scenes, aoi, models)
-- =====================================================================
CREATE TABLE ops.processing_jobs (
    job_id        uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type      text        NOT NULL CHECK (job_type IN
                  ('tile_registration','tile_stats','semantic_labeling','embedding_generation',
                   'temporal_matching','change_detection','reindex','export')),
    status        text        NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
    priority      smallint    NOT NULL DEFAULT 5 CHECK (priority BETWEEN 1 AND 9),
    scene_id      uuid        REFERENCES catalog.satellite_scenes(scene_id) ON DELETE CASCADE,
    aoi_id        uuid        REFERENCES catalog.aoi(aoi_id) ON DELETE SET NULL,
    model_id      integer     REFERENCES catalog.ml_models(model_id),
    params        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    progress      real        NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 1),
    attempts      integer     NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    max_attempts  integer     NOT NULL DEFAULT 3 CHECK (max_attempts >= 1),
    locked_by     text,
    locked_at     timestamptz,
    heartbeat_at  timestamptz,
    error         text,
    requested_by  uuid        REFERENCES ops.app_users(user_id) ON DELETE SET NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    started_at    timestamptz,
    finished_at   timestamptz,
    updated_at    timestamptz NOT NULL DEFAULT now()
);

-- =====================================================================
-- 7. SEMANTIC: labels and embeddings
-- =====================================================================

-- 7.1 tile_semantics : one row per (tile, semantic model/version)
CREATE TABLE semantic.tile_semantics (
    semantic_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tile_id          bigint   NOT NULL REFERENCES catalog.tiles(tile_id) ON DELETE CASCADE,
    model_id         integer  NOT NULL REFERENCES catalog.ml_models(model_id),
    primary_label    text,
    labels           jsonb    NOT NULL DEFAULT '{}'::jsonb,   -- {"urban":0.71,"vegetation":0.20}
    land_cover       jsonb    NOT NULL DEFAULT '{}'::jsonb,   -- full class fractions (0..100)
    built_up_pct     real     CHECK (built_up_pct   BETWEEN 0 AND 100),
    vegetation_pct   real     CHECK (vegetation_pct BETWEEN 0 AND 100),
    water_pct        real     CHECK (water_pct      BETWEEN 0 AND 100),
    ndvi_mean        real     CHECK (ndvi_mean BETWEEN -1 AND 1),
    ndwi_mean        real     CHECK (ndwi_mean BETWEEN -1 AND 1),
    detected_objects jsonb    NOT NULL DEFAULT '[]'::jsonb,
    caption          text,
    search_vector    tsvector GENERATED ALWAYS AS
                     (to_tsvector('english', coalesce(primary_label,'') || ' ' || coalesce(caption,''))) STORED,
    created_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_semantics UNIQUE (tile_id, model_id)
);

-- 7.2 tile_embeddings : image embeddings (one row per tile per embedding model)
CREATE TABLE semantic.tile_embeddings (
    tile_id     bigint   NOT NULL REFERENCES catalog.tiles(tile_id) ON DELETE CASCADE,
    model_id    integer  NOT NULL REFERENCES catalog.ml_models(model_id),
    embedding   vector(512) NOT NULL,        -- [DEPENDS ON EMBEDDING MODEL] 512 = CLIP ViT-B family
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tile_id, model_id)
);
COMMENT ON COLUMN semantic.tile_embeddings.embedding IS
  'L2-normalised image embedding. Dimension is fixed by the model in ml_models.embedding_dim.';

-- =====================================================================
-- 8. ANALYSIS: temporal pairs and change detection
-- =====================================================================

-- 8.1 tile_pairs : before/after relation between two tiles (self many-to-many on tiles)
CREATE TABLE analysis.tile_pairs (
    pair_id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    before_tile_id   bigint   NOT NULL REFERENCES catalog.tiles(tile_id) ON DELETE CASCADE,
    after_tile_id    bigint   NOT NULL REFERENCES catalog.tiles(tile_id) ON DELETE CASCADE,
    before_datetime  timestamptz NOT NULL,     -- filled by trigger from tiles
    after_datetime   timestamptz NOT NULL,     -- filled by trigger from tiles
    matching_method  text     NOT NULL CHECK (matching_method IN
                     ('grid_aligned','derived_window','warped_overlap','manual')),
    resampling_required boolean NOT NULL DEFAULT false,   -- true => reader must warp before-pixels onto the after grid
    overlap_ratio    real     NOT NULL CHECK (overlap_ratio BETWEEN 0 AND 1),  -- |A∩B| / |A|
    iou              real     CHECK (iou BETWEEN 0 AND 1),                     -- |A∩B| / |A∪B|
    overlap_geom     geometry(MultiPolygon, 4326),
    match_score      real,
    status           text     NOT NULL DEFAULT 'candidate'
                     CHECK (status IN ('candidate','validated','rejected')),
    aoi_id           uuid     REFERENCES catalog.aoi(aoi_id) ON DELETE SET NULL,
    created_by_job   uuid     REFERENCES ops.processing_jobs(job_id) ON DELETE SET NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_pair       UNIQUE (before_tile_id, after_tile_id),
    CONSTRAINT ck_pair_diff  CHECK (before_tile_id <> after_tile_id),
    CONSTRAINT ck_pair_order CHECK (before_datetime < after_datetime)
);

CREATE OR REPLACE FUNCTION analysis.fill_pair_datetimes() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    SELECT acquisition_datetime INTO STRICT NEW.before_datetime
      FROM catalog.tiles WHERE tile_id = NEW.before_tile_id;
    SELECT acquisition_datetime INTO STRICT NEW.after_datetime
      FROM catalog.tiles WHERE tile_id = NEW.after_tile_id;
    RETURN NEW;
END $$;

-- BEFORE INSERT only: datetimes are immutable facts of the tiles, so no drift is possible.
CREATE TRIGGER trg_pair_datetimes BEFORE INSERT ON analysis.tile_pairs
    FOR EACH ROW EXECUTE FUNCTION analysis.fill_pair_datetimes();

-- Temporal-matching helper: return (creating if needed) the window in scene p_scene_id that has EXACTLY
-- the same footprint as tile p_ref_tile_id. Possible without resampling only when both scenes share
-- CRS + pixel size + pixel lattice (typical for Sentinel-2 scenes of one UTM zone). Otherwise NULL,
-- and the caller falls back to matching_method = 'warped_overlap'.
CREATE OR REPLACE FUNCTION analysis.get_or_create_derived_tile(p_scene_id uuid, p_ref_tile_id bigint)
RETURNS bigint LANGUAGE plpgsql AS $fn$
DECLARE
    s   catalog.satellite_scenes%ROWTYPE;   -- target scene (where the window is wanted)
    rs  catalog.satellite_scenes%ROWTYPE;   -- reference tile's scene
    r   catalog.tiles%ROWTYPE;              -- reference tile
    col_off double precision;
    row_off double precision;
    c integer; rw integer; tid bigint;
BEGIN
    SELECT * INTO s  FROM catalog.satellite_scenes WHERE scene_id = p_scene_id;
    SELECT * INTO r  FROM catalog.tiles            WHERE tile_id  = p_ref_tile_id;
    SELECT * INTO rs FROM catalog.satellite_scenes WHERE scene_id = r.scene_id;
    IF s.crs_epsg <> rs.crs_epsg OR s.resolution_m <> rs.resolution_m THEN RETURN NULL; END IF;

    col_off := (r.native_minx - s.transform[3]) / s.resolution_m::float8;
    row_off := (s.transform[6] - r.native_maxy) / s.resolution_m::float8;
    IF abs(col_off - round(col_off)) > 0.001 OR abs(row_off - round(row_off)) > 0.001 THEN
        RETURN NULL;                         -- sub-pixel phase difference: needs resampling
    END IF;
    c := round(col_off); rw := round(row_off);
    IF c < 0 OR rw < 0 OR c + r.width > s.width_px OR rw + r.height > s.height_px THEN
        RETURN NULL;                         -- window falls outside the scene
    END IF;

    INSERT INTO catalog.tiles (scene_id, acquisition_datetime, resolution_m, window_x, window_y, width, height,
                               native_minx, native_miny, native_maxx, native_maxy, geom, is_derived)
    VALUES (p_scene_id, s.acquisition_datetime, s.resolution_m, c, rw, r.width, r.height,
            r.native_minx, r.native_miny, r.native_maxx, r.native_maxy, r.geom, true)
    ON CONFLICT ON CONSTRAINT uq_tile_window DO NOTHING
    RETURNING tile_id INTO tid;

    IF tid IS NULL THEN                      -- already existed (or created by a concurrent worker)
        SELECT tile_id INTO tid FROM catalog.tiles
         WHERE scene_id = p_scene_id AND window_x = c AND window_y = rw
           AND width = r.width AND height = r.height;
    END IF;
    RETURN tid;
END $fn$;

-- ---------------------------------------------------------------------
-- Temporal matching (footprint-driven, NOT row/col based)
--   analysis.match_candidates()  step 1-2: choose the best BEFORE scene for every AFTER tile
--   analysis.match_tiles()       step 3-4: materialise the before window, verify, store the pair
-- ---------------------------------------------------------------------
-- Candidate selection: BEFORE scenes whose footprint COVERS the after tile (same source, level,
-- resolution), ranked by scene cloud, seasonal similarity and a same-grid bonus.
-- The ranking weights are STARTING VALUES - tune them on real data.
CREATE OR REPLACE FUNCTION analysis.match_candidates(
    p_aoi_id          uuid,
    p_before_start    timestamptz,
    p_before_end      timestamptz,
    p_after_start     timestamptz,
    p_after_end       timestamptz,
    p_max_tile_cloud  real DEFAULT 30,
    p_min_valid_pct   real DEFAULT 70,
    p_max_scene_cloud real DEFAULT 40)
RETURNS TABLE (after_tile_id bigint, before_scene_id uuid, same_grid boolean,
               window_x integer, window_y integer, width integer, height integer)
LANGUAGE sql STABLE AS $fn$
    SELECT a.tile_id, m.before_scene_id, m.same_grid, a.window_x, a.window_y, a.width, a.height
    FROM   catalog.aoi ar
    JOIN   catalog.tiles a             ON ST_Intersects(a.geom, ar.geom)
    JOIN   catalog.satellite_scenes sa ON sa.scene_id = a.scene_id
    CROSS  JOIN LATERAL (
            SELECT sb.scene_id AS before_scene_id,
                   (sb.grid_signature = sa.grid_signature) AS same_grid
            FROM   catalog.satellite_scenes sb
            WHERE  ST_Covers(sb.footprint, a.geom)                     -- scene covers the whole tile (GiST)
              AND  sb.source_id = sa.source_id                         -- never mix modalities / families
              AND  sb.resolution_m = sa.resolution_m
              AND  sb.processing_level = sa.processing_level
              AND  sb.ingestion_status = 'ready'
              AND  sb.acquisition_datetime >= p_before_start AND sb.acquisition_datetime < p_before_end
              AND  COALESCE(sb.cloud_cover, 0) <= p_max_scene_cloud
            ORDER  BY 0.5 * (1 - COALESCE(sb.cloud_cover, 0) / 100.0)
                    + 0.3 * (1 - LEAST(LEAST(abs(EXTRACT(doy FROM sb.acquisition_datetime) - EXTRACT(doy FROM a.acquisition_datetime)),
                                             365 - abs(EXTRACT(doy FROM sb.acquisition_datetime) - EXTRACT(doy FROM a.acquisition_datetime))) / 90.0, 1))
                    + 0.2 * (sb.grid_signature = sa.grid_signature)::int DESC
            LIMIT  1
    ) m
    WHERE  ar.aoi_id = p_aoi_id
      AND  a.acquisition_datetime >= p_after_start AND a.acquisition_datetime < p_after_end
      AND  a.processing_status = 'ready'
      AND  a.cloud_pct <= p_max_tile_cloud AND a.valid_pixel_pct >= p_min_valid_pct
$fn$;

-- Pair creation. Two statements on purpose: rows created by get_or_create_derived_tile() are only
-- visible to LATER statements (one statement = one snapshot).
CREATE OR REPLACE FUNCTION analysis.match_tiles(
    p_aoi_id          uuid,
    p_before_start    timestamptz,
    p_before_end      timestamptz,
    p_after_start     timestamptz,
    p_after_end       timestamptz,
    p_min_overlap     real DEFAULT 0.95,
    p_max_tile_cloud  real DEFAULT 30,
    p_min_valid_pct   real DEFAULT 70,
    p_max_scene_cloud real DEFAULT 40)
RETURNS TABLE (matching_method text, pairs_created integer)
LANGUAGE plpgsql AS $fn$
BEGIN
    -- Statement 1: make sure a window exists in the chosen before scene (derived windows are created here)
    PERFORM analysis.get_or_create_derived_tile(c.before_scene_id, c.after_tile_id)
    FROM   analysis.match_candidates(p_aoi_id, p_before_start, p_before_end, p_after_start, p_after_end,
                                     p_max_tile_cloud, p_min_valid_pct, p_max_scene_cloud) c
    WHERE  NOT c.same_grid;

    -- Statement 2: resolve, VERIFY geometrically, and store
    RETURN QUERY
    WITH resolved AS (
        SELECT c.after_tile_id, c.same_grid,
               COALESCE(
                 (SELECT t.tile_id FROM catalog.tiles t              -- grid_aligned: same window, same grid
                   WHERE c.same_grid AND t.scene_id = c.before_scene_id
                     AND t.window_x = c.window_x AND t.window_y = c.window_y
                     AND t.width = c.width AND t.height = c.height),
                 analysis.get_or_create_derived_tile(c.before_scene_id, c.after_tile_id)  -- derived_window (exists now)
               ) AS before_tile_id
        FROM   analysis.match_candidates(p_aoi_id, p_before_start, p_before_end, p_after_start, p_after_end,
                                         p_max_tile_cloud, p_min_valid_pct, p_max_scene_cloud) c
    ), verified AS (
        SELECT r.before_tile_id, r.after_tile_id, r.same_grid,
               ST_Area(ST_Intersection(a.geom, b.geom)) / ST_Area(a.geom)                 AS overlap_ratio,
               ST_Area(ST_Intersection(a.geom, b.geom)) / ST_Area(ST_Union(a.geom, b.geom)) AS iou,
               ST_Multi(ST_CollectionExtract(ST_Intersection(a.geom, b.geom), 3))         AS overlap_geom
        FROM   resolved r
        JOIN   catalog.tiles a ON a.tile_id = r.after_tile_id
        JOIN   catalog.tiles b ON b.tile_id = r.before_tile_id
        WHERE  r.before_tile_id IS NOT NULL
    ), ins AS (
        INSERT INTO analysis.tile_pairs
               (before_tile_id, after_tile_id, before_datetime, after_datetime, matching_method,
                resampling_required, overlap_ratio, iou, overlap_geom, aoi_id)
        SELECT v.before_tile_id, v.after_tile_id, now(), now(),      -- datetimes overwritten by trigger
               CASE WHEN v.same_grid THEN 'grid_aligned' ELSE 'derived_window' END,
               false, v.overlap_ratio::real, v.iou::real, v.overlap_geom, p_aoi_id
        FROM   verified v
        WHERE  v.overlap_ratio >= p_min_overlap
        ON CONFLICT (before_tile_id, after_tile_id) DO NOTHING
        RETURNING analysis.tile_pairs.matching_method
    )
    SELECT ins.matching_method, count(*)::integer FROM ins GROUP BY ins.matching_method;
END $fn$;

-- 8.2 change_classes : lookup (add classes with INSERT, no migration)
CREATE TABLE analysis.change_classes (
    class_id      smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code          text     NOT NULL UNIQUE CHECK (code ~ '^[a-z_]+$'),
    display_name  text     NOT NULL,
    description   text,
    color_hex     text     CHECK (color_hex ~ '^#[0-9A-Fa-f]{6}$'),
    sort_order    smallint NOT NULL DEFAULT 100
);

-- 8.3 change_results : ONE ROW PER (pair, model, parameters) RUN - also when nothing changed
CREATE TABLE analysis.change_results (
    change_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pair_id              bigint   NOT NULL REFERENCES analysis.tile_pairs(pair_id) ON DELETE CASCADE,
    model_id             integer  NOT NULL REFERENCES catalog.ml_models(model_id),
    classifier_model_id  integer  REFERENCES catalog.ml_models(model_id),
    job_id               uuid     REFERENCES ops.processing_jobs(job_id) ON DELETE SET NULL,
    params               jsonb    NOT NULL DEFAULT '{}'::jsonb,   -- threshold, min_area_m2, ...
    params_hash          text     NOT NULL,
    status               text     NOT NULL DEFAULT 'running'
                         CHECK (status IN ('running','completed','failed')),
    changed_pixel_count  integer  CHECK (changed_pixel_count >= 0),
    changed_area_m2      double precision NOT NULL DEFAULT 0 CHECK (changed_area_m2 >= 0),
    changed_pct          real     CHECK (changed_pct BETWEEN 0 AND 100),
    mean_confidence      real     CHECK (mean_confidence BETWEEN 0 AND 1),
    feature_count        integer  NOT NULL DEFAULT 0 CHECK (feature_count >= 0),
    storage_backend      text     NOT NULL DEFAULT 'local' CHECK (storage_backend IN ('local','s3')),
    mask_path            text,            -- binary change mask (COG GeoTIFF, uint8)
    probability_path     text,            -- optional float probability raster
    error                text,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_change_run UNIQUE (pair_id, model_id, params_hash),
    CONSTRAINT ck_change_mask CHECK (status <> 'completed'
                              OR mask_path IS NOT NULL OR coalesce(changed_pixel_count, 0) = 0)
);

-- 8.4 change_features : one row per changed polygon, with its class
CREATE TABLE analysis.change_features (
    feature_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    change_id             bigint   NOT NULL REFERENCES analysis.change_results(change_id) ON DELETE CASCADE,
    class_id              smallint NOT NULL REFERENCES analysis.change_classes(class_id),
    classification_method text     NOT NULL DEFAULT 'none'
                          CHECK (classification_method IN ('none','rule_based','model','manual')),
    confidence            real     NOT NULL CHECK (confidence BETWEEN 0 AND 1),   -- mean change probability
    class_confidence      real     CHECK (class_confidence BETWEEN 0 AND 1),
    area_m2               double precision NOT NULL CHECK (area_m2 > 0),
    pixel_count           integer  CHECK (pixel_count > 0),
    geom                  geometry(MultiPolygon, 4326) NOT NULL CHECK (ST_IsValid(geom)),
    attributes            jsonb    NOT NULL DEFAULT '{}'::jsonb,  -- ndvi_before/after, brightness_delta ...
    review_status         text     NOT NULL DEFAULT 'unreviewed'
                          CHECK (review_status IN ('unreviewed','confirmed','rejected')),
    reviewed_by           uuid     REFERENCES ops.app_users(user_id) ON DELETE SET NULL,
    reviewed_at           timestamptz,
    created_at            timestamptz NOT NULL DEFAULT now()
);

-- =====================================================================
-- 9. OPS (continued): per-product ingestion state and search log
-- =====================================================================
CREATE TABLE ops.ingestion_items (
    item_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id             uuid     NOT NULL REFERENCES ops.ingestion_jobs(job_id) ON DELETE CASCADE,
    source_id          smallint NOT NULL REFERENCES catalog.data_sources(source_id),
    product_id         text     NOT NULL,
    remote_assets      jsonb    NOT NULL DEFAULT '{}'::jsonb,   -- URLs / hrefs discovered via STAC
    status             text     NOT NULL DEFAULT 'discovered'
                       CHECK (status IN ('discovered','queued','downloading','downloaded','validated',
                                         'converted','registered','failed','skipped')),
    attempts           integer  NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_retry_at      timestamptz,
    bytes_expected     bigint   CHECK (bytes_expected >= 0),
    bytes_done         bigint   NOT NULL DEFAULT 0 CHECK (bytes_done >= 0),
    checksum_expected  text,
    staging_path       text,
    last_error         text,
    scene_id           uuid     REFERENCES catalog.satellite_scenes(scene_id) ON DELETE SET NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_ingest_item UNIQUE (job_id, product_id)
);

CREATE TABLE ops.search_log (
    search_id      uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    requested_by   uuid        REFERENCES ops.app_users(user_id) ON DELETE SET NULL,
    query_text     text,
    parsed_intent  jsonb       NOT NULL DEFAULT '{}'::jsonb,
    filters        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    embedding_model_id integer REFERENCES catalog.ml_models(model_id),
    result_count   integer,
    duration_ms    integer,
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- =====================================================================
-- 10. INDEXES  (every index below has a stated purpose - see document section 14)
-- =====================================================================

-- scenes: small table (10^3 - 10^5 rows). Only time + spatial + lookups.
CREATE UNIQUE INDEX ux_scenes_stac         ON catalog.satellite_scenes (source_id, stac_id) WHERE stac_id IS NOT NULL;
CREATE INDEX        ix_scenes_acq_time     ON catalog.satellite_scenes (acquisition_datetime);
CREATE INDEX        gix_scenes_footprint   ON catalog.satellite_scenes USING gist (footprint);
CREATE INDEX        ix_scenes_grid         ON catalog.satellite_scenes (grid_signature, acquisition_datetime);
CREATE INDEX        ix_scenes_job          ON catalog.satellite_scenes (ingestion_job_id) WHERE ingestion_job_id IS NOT NULL;

-- assets: uq_asset_role already covers scene_id lookups.

-- aoi
CREATE INDEX gix_aoi_geom ON catalog.aoi USING gist (geom);

-- tiles: the big table. scene_id is covered by the LEADING column of uq_tile_window.
CREATE INDEX gix_tiles_geom        ON catalog.tiles USING gist (geom);
CREATE INDEX ix_tiles_acq_time     ON catalog.tiles (acquisition_datetime);
CREATE INDEX ix_tiles_work_queue   ON catalog.tiles (processing_status)
       WHERE processing_status NOT IN ('ready','excluded');

-- semantics: uq_semantics covers tile_id. Full-text on captions/labels.
CREATE INDEX gix_semantics_fts ON semantic.tile_semantics USING gin (search_vector);

-- embeddings: PK (tile_id, model_id) covers the tile FK.
-- HNSW is created PER EMBEDDING MODEL as a PARTIAL index (vectors of different models must never
-- share one graph). Call the helper AFTER the model is registered - ideally after bulk-loading
-- the embeddings, which makes the build much faster.            [DEPENDS ON EMBEDDING MODEL]
CREATE OR REPLACE FUNCTION ops.create_embedding_index(
        p_model_id integer, p_m integer DEFAULT 16, p_ef_construction integer DEFAULT 64)
RETURNS void LANGUAGE plpgsql AS $fn$
BEGIN
    EXECUTE format(
      'CREATE INDEX IF NOT EXISTS %I ON semantic.tile_embeddings
         USING hnsw (embedding vector_cosine_ops) WITH (m = %s, ef_construction = %s)
         WHERE model_id = %s',
      'hnsw_emb_model' || p_model_id, p_m, p_ef_construction, p_model_id);
END $fn$;

-- pairs: uq_pair covers before_tile_id; add after_tile_id + AOI.
CREATE INDEX ix_pairs_after   ON analysis.tile_pairs (after_tile_id);
CREATE INDEX ix_pairs_aoi     ON analysis.tile_pairs (aoi_id) WHERE aoi_id IS NOT NULL;
CREATE INDEX ix_pairs_times   ON analysis.tile_pairs (before_datetime, after_datetime);

-- change results / features: uq_change_run covers pair_id.
CREATE INDEX ix_features_change ON analysis.change_features (change_id);
CREATE INDEX gix_features_geom  ON analysis.change_features USING gist (geom);
CREATE INDEX ix_features_class  ON analysis.change_features (class_id, confidence DESC);

-- jobs
CREATE INDEX ix_proc_queue  ON ops.processing_jobs (priority DESC, created_at)  WHERE status = 'queued';
CREATE INDEX ix_proc_scene  ON ops.processing_jobs (scene_id) WHERE scene_id IS NOT NULL;
CREATE INDEX ix_ing_retry   ON ops.ingestion_items (next_retry_at) WHERE status IN ('queued','failed');

-- =====================================================================
-- 11. updated_at triggers (attached to every table that has the column)
-- =====================================================================
DO $$
DECLARE r record;
BEGIN
  FOR r IN
    SELECT c.table_schema, c.table_name
      FROM information_schema.columns c
      JOIN information_schema.tables  t
        ON t.table_schema = c.table_schema AND t.table_name = c.table_name AND t.table_type = 'BASE TABLE'
     WHERE c.column_name = 'updated_at'
       AND c.table_schema IN ('catalog','semantic','analysis','ops')
  LOOP
    EXECUTE format('CREATE TRIGGER trg_%I_updated BEFORE UPDATE ON %I.%I
                    FOR EACH ROW EXECUTE FUNCTION ops.set_updated_at()',
                   r.table_name, r.table_schema, r.table_name);
  END LOOP;
END $$;

-- =====================================================================
-- 12. Views used by the API / Rasterio service
-- =====================================================================
-- Everything Rasterio needs to read one window - the ONLY place raster paths are resolved.
CREATE VIEW catalog.v_tile_raster_ref AS
SELECT t.tile_id,
       s.scene_id,
       s.storage_backend,
       s.raster_path,
       s.band_map,
       s.crs_epsg,
       s.transform,
       t.window_x, t.window_y, t.width, t.height,
       t.native_minx, t.native_miny, t.native_maxx, t.native_maxy,
       t.materialized_path
FROM catalog.tiles t
JOIN catalog.satellite_scenes s USING (scene_id);

-- One row per changed polygon with everything the map needs.
CREATE VIEW analysis.v_change_map AS
SELECT f.feature_id, f.change_id, r.pair_id,
       c.code AS change_type, c.display_name,
       f.confidence, f.class_confidence, f.classification_method,
       f.area_m2, f.review_status, f.geom,
       p.before_datetime, p.after_datetime,
       m.name AS model_name, m.version AS model_version,
       r.created_at
FROM analysis.change_features f
JOIN analysis.change_results  r ON r.change_id = f.change_id
JOIN analysis.tile_pairs      p ON p.pair_id   = r.pair_id
JOIN analysis.change_classes  c ON c.class_id  = f.class_id
JOIN catalog.ml_models        m ON m.model_id  = r.model_id;

-- =====================================================================
-- 13. Reference data (seed)
-- =====================================================================
INSERT INTO catalog.data_sources (code, display_name, provider, modality, stac_collection, default_band_config, notes) VALUES
 ('SENTINEL2', 'Sentinel-2 MSI',            'ESA / Copernicus', 'optical', 'sentinel-2-l2a',
   '{"ml_bands":["B02","B03","B04","B08"],"native_resolution_m":10}', 'Current ML pipeline source'),
 ('SENTINEL1', 'Sentinel-1 C-SAR',          'ESA / Copernicus', 'sar',     'sentinel-1-grd',
   '{"bands":["VV","VH"]}', 'Stored and searchable; needs a SAR-specific change model'),
 ('LANDSAT',   'Landsat Collection 2',      'USGS / NASA',      'optical', 'landsat-c2-l2',
   '{"native_resolution_m":30}', 'Optical, 30 m'),
 ('BHUVAN',    'Bhuvan / ISRO products',    'NRSC / ISRO',      'optical', NULL,
   '{}', 'ASSUMPTION: product list, format and licence must be confirmed');

INSERT INTO analysis.change_classes (code, display_name, description, color_hex, sort_order) VALUES
 ('construction',    'Possible construction', 'New built-up / bare-to-built change candidate', '#E4572E', 10),
 ('vegetation_loss', 'Vegetation loss',       'Vegetation to non-vegetated',                   '#F2A541', 20),
 ('water_change',    'Water change',          'Water extent gain or loss',                     '#2E86DE', 30),
 ('other_change',    'Other change',          'Change that matches no specific rule',          '#9B59B6', 40),
 ('unclassified',    'Unclassified change',   'Binary change mask only, no class assigned',    '#7F8C8D', 50);

-- Placeholder registry rows - REPLACE weights/versions after the model experiments.
INSERT INTO catalog.ml_models (name, version, task, architecture, embedding_dim, input_spec, training_data) VALUES
 ('siamese-unet-s2-4band', '0.1.0', 'change_detection', 'siamese_unet', NULL,
   '{"bands":["B02","B03","B04","B08"],"patch_size":256,"scale":"reflectance_0_1"}', 'OSCD'),
 ('embedding-model-tbd',   '0.0.0', 'embedding',        'vit',          512,
   '{"bands":["B04","B03","B02"],"resize":224,"stretch":"p2_p98"}', 'TBD - see experiment E1'),
 ('spectral-rules',        '1.0.0', 'spectral_rules',   'ndvi_ndwi_thresholds', NULL,
   '{"bands":["B02","B03","B04","B08"]}', NULL),
 ('change-rules-v1',       '1.0.0', 'change_classification', 'rule_based', NULL,
   '{"uses":["ndvi_delta","ndwi_delta","brightness_delta"]}', NULL);

-- Build the HNSW index for the (placeholder) embedding model registered above.
SELECT ops.create_embedding_index(model_id) FROM catalog.ml_models WHERE task = 'embedding';
