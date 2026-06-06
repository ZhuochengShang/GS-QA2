# Data Ingestion

This folder documents the data-loading steps used before running the GS-QA2
baselines. The benchmark expects two data sources:

- OSM-derived vector layers loaded into PostGIS tables.
- DEM GeoTIFF tiles loaded into a PostGIS raster table.

The raw OSM and DEM files are not committed to this repository.

## OSM Vector Tables

Expected raw directory layout:

```text
$DATA_ROOT/osm_extract/
  lakes/
  parks/
  pois/
  postal_codes/
  roads/
```

Load the tables with:

```bash
cd GS-QA/ingestion
./ingest_osm_postgis.sh
```

Required environment variables:

```bash
export DATA_ROOT="/path/to/data"
export PGHOST="localhost"
export PGPORT="5432"
export PGDATABASE="gsqa"
export PGUSER="postgres"
export PGPASSWORD=""
```

The loader creates these benchmark tables:

- `pois`
- `roads`
- `parks`
- `lakes`
- `regions`

Each table stores a `GEOGRAPHY(..., 4326)` geometry column and the attributes
defined in the corresponding schema file under `GS-QA/generator/`.

## DEM Raster Table

Load DEM GeoTIFF tiles with:

```bash
cd GS-QA/ingestion
./ingest_dem_postgis.sh
```

Required environment variables:

```bash
export DEM_ROOT="/path/to/dem_tiles"
export DEM_TABLE="public.dem_us"
export POSTGIS_DSN="postgresql://postgres@localhost:5432/gsqa"
```

The script uses `raster2pgsql` with 256 by 256 tiling and creates raster
indexes. The table name should match the table name supplied to raster
Text2SQL/CHESS through `--dem-table` or schema prompts.

## Ground Truth

The benchmark JSON/JSONL files contain SQL programs used to compute ground-truth
answers. After the OSM and DEM tables are loaded, the evaluation scripts execute
these SQL programs against PostGIS and compare baseline outputs to the reference
answers.
