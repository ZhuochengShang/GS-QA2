import os


_SQL_DIALECT_ALIASES = {
    "postgresql": "postgres",
}


def get_sql_dialect() -> str:
    dialect = os.getenv("CHESS_SQL_DIALECT", "sqlite").strip().lower()
    dialect = _SQL_DIALECT_ALIASES.get(dialect, dialect)
    if dialect not in {"sqlite", "postgres", "postgis"}:
        return "sqlite"
    return dialect


def uses_postgres() -> bool:
    return get_sql_dialect() in {"postgres", "postgis"}


def get_sqlglot_dialect() -> str:
    return "postgres" if uses_postgres() else "sqlite"


def get_prompt_sql_dialect_label() -> str:
    dialect = get_sql_dialect()
    if dialect == "postgis":
        return "PostgreSQL with PostGIS"
    if dialect == "postgres":
        return "PostgreSQL"
    return "SQLite"


def get_spatial_extension_label() -> str:
    return "PostGIS" if get_sql_dialect() == "postgis" else "SpatiaLite"


def get_date_function_guidance() -> str:
    if uses_postgres():
        return "Use PostgreSQL date functions such as EXTRACT(...) or DATE_TRUNC(...) when date manipulation is needed."
    return "Use STRFTIME() for date manipulation when needed."


def get_gsqa_table_descriptions() -> str:
    if get_sql_dialect() != "postgis":
        return ""
    if os.getenv("CHESS_DOMAIN", "").strip().lower() == "mapqa":
        return """MapQA PostGIS table descriptions:
Table 1: planet_osm_point
contains points of interest and point map features.
Important columns: osm_id, name, amenity, shop, tourism, leisure, highway, railway, historic, sport, operator, way.
The way column is PostGIS geometry in EPSG:3857. For meter distance/radius, use ST_Transform(way,4326)::geography.

Table 2: planet_osm_line
contains roads, streets, paths, and linear map features.
Important columns: osm_id, name, highway, railway, tracktype, way.
Use this table for street intersection questions.

Table 3: planet_osm_polygon
contains polygon/area features. Negative OSM ids in MapQA often refer to this table.
Important columns: osm_id, name, amenity, shop, tourism, leisure, historic, sport, way.
Use ST_Centroid(way) when a polygon is used as a reference point.

MapQA rules:
- Use only planet_osm_point, planet_osm_line, and planet_osm_polygon.
- Do not use GS-QA tables such as pois, parks, roads, lakes, regions, or dem_us.
- For type/amenity lookup use COALESCE(amenity, shop, tourism, leisure, highway, railway, historic, sport).
- For nearest neighbor, prefer ORDER BY candidate.way <-> ref.way LIMIT 1 in native EPSG:3857.
- For distances and within-radius predicates, use ST_Distance/ST_DWithin on ST_Transform(...,4326)::geography.
- For named entity lookup, prefer the supplied osm_id in the hint. If point lookup returns no row and the osm_id is negative, use planet_osm_polygon.
- For street intersections, find street names in planet_osm_line using ILIKE, compute an intersection point if possible, and use ST_ClosestPoint/ST_DWithin fallback when lines do not exactly intersect.
- Return only the answer column requested by the question: name, amenity/type, or distance in meters."""
    return """Table 1: pois
contains the points of interest
Schema:
   column_name    |     description
------------------+-------------------
 id               | unique identifier
 geometry         | geography type represents the shape and position on earth
 poi_name         | name of the poi
 wikidata         | unique identifier reference to wikidata
 wikipedia        | unique name reference to wikipedia page
 addr_state       | the state where the poi is located
 addr_city        | the city where the poi is located
 cuisine          | type of cuisine the associated with restaurant pois
 leisure          | type of leisure
 tourism          | type of tourism
 takeaway         | indicates if a resturant offers takeaway
 drive_through    | indicates if a resturant offers drive through
 museum           | type of museum
 healthcare       | type of healthcare service
 outdoor_seating  | indicates if an amenity has outdoor seating
 emergency        | indicates if a healthcare service provides emergency service
 restaurant       | attribute related to restaurants
 amenity          | type of amenity provided

Table 2: lakes
contains lakes, rivers, waterbodies, etc.
Schema:
   column_name    |     description
------------------+-------------------
 id               | unique identifier
 geometry         | geography type represents the shape and position on earth
 lake_name        | name of the lake
 wikidata         | unique identifier reference to wikidata
 wikipedia        | unique name reference to wikipedia page
 addr_country     | the country where the lake is located
 addr_state       | the state where the lake is located
 addr_county      | the county where the lake is located
 addr_city        | the city where the lake is located
 addr_postcode    | postcode where the lake is located
 addr_street      | street where the lake is located
 addr_housenumber | house number specific to this lake
 waterway         | type of waterway
 water            | type of waterbody

Table 3: parks
contains parks, gardens, etc.
Schema:
   column_name    |     description
------------------+-------------------
 id               | unique identifier
 geometry         | geography type represents the shape and position on earth
 park_name        | name of the park
 wikidata         | unique identifier reference to wikidata
 wikipedia        | unique name reference to wikipedia page
 leisure          | type of leisure
 park             | type of park
 tourism          | type of tourism

Table 4: roads
contains roads, walkways, etc.
Schema:
   column_name    |     description
------------------+-------------------
 id               | unique identifier
 geometry         | geography type represents the shape and position on earth
 road_name        | name of the road
 wikidata         | unique identifier reference to wikidata
 wikipedia        | unique name reference to wikipedia page
 highway          | attribute associated with roads of type highway
 sidewalk         | attribute associated with roads of type sidewalk
 foot             | attribute associated with roads of type foot
 bicycle          | attribute associated with roads of type bycycle
 cycleway         | attribute associated with roads of type cycleway

Table 5: regions
contains adminstrative region boundaries, like cities and states, etc.
Schema:
  column_name  |     data_type
---------------+-------------------
 id            | integer
 geometry      | geography type represents the shape and position on earth
 region_name   | name of the region
  border_type   | the type of border
  wikidata      | unique identifier reference to wikidata
  wikipedia     | unique name reference to wikipedia page

Table 6: dem_us
contains ASTER GDEM Version 3 elevation raster tiles for the contiguous U.S. study area
Schema:
   column_name    |     description
------------------+-------------------
 rid              | raster tile identifier
 rast             | PostGIS raster tile; band 1 stores elevation values in meters
 filename         | source ASTER GDEM tile filename, for example ASTGTMV003_N24W075_dem.tif

DEM metadata:
- public.dem_us has 265950 PostGIS tile rows.
- The raster extent is approximately lon -126.000138888889 to -65.99986111111112 and lat 23.99986111111112 to 50.0001388888889.
- The raster uses EPSG:4326, one band, 256 x 256 pixel tiles, 0.000277777777777778 degree pixel scale, and 16BSI pixel type.

Raster / PostGIS rules:
- Use public.dem_us for raster and terrain computations.
- Use pois, lakes, parks, roads, or regions only to obtain geometry referenced in the question.
- Do not use vector tables for the final numeric terrain answer except as geometry sources or spatial filters.
- Use filename only for provenance/debugging; do not use filename to answer terrain questions unless the question explicitly asks about source files.
- Use only columns explicitly listed for each table.
- In pois, lakes, parks, roads, and regions, the geometry column is of type geography.
- For raster operations such as ST_Intersects(rast, ...), ST_Clip, and raster/geometry operations, cast vector geography to geometry with geometry::geometry when needed.
- Do not use addr_state or addr_city on regions.
- To disambiguate places in regions, use region_name, border_type, wikidata, wikipedia, or spatial intersection with another regions row.
- Treat dem_us band 1 as elevation.
- Do not assume dem_us already stores slope, aspect, or other derived terrain products.
- For ST_Slope and ST_Aspect output pixel type, use only '32BF' unless there is a clear reason to use '64BF'. Never invent pixel type strings such as '3DDDA'.
- If slope, aspect, or steepness is requested, derive it from the DEM before summarizing."""


def get_postgis_example_block() -> str:
    if get_sql_dialect() != "postgis":
        return ""
    if os.getenv("CHESS_DOMAIN", "").strip().lower() == "mapqa":
        return """MapQA PostGIS examples:
Example 1: nearest restaurant to a reference POI by osm_id
```sql
WITH ref AS (
  SELECT way FROM planet_osm_point WHERE osm_id = 9742148508
)
SELECT p.name
FROM planet_osm_point AS p, ref
WHERE p.amenity = 'restaurant'
  AND p.osm_id <> 9742148508
  AND p.name IS NOT NULL
ORDER BY p.way <-> ref.way
LIMIT 1;
```

Example 2: named POIs within 50 meters
```sql
WITH ref AS (
  SELECT way FROM planet_osm_point WHERE osm_id = 9742148508
)
SELECT string_agg(p.name, '; ' ORDER BY p.name) AS names
FROM planet_osm_point AS p, ref
WHERE ST_DWithin(
    ST_Transform(p.way, 4326)::geography,
    ST_Transform(ref.way, 4326)::geography,
    50
  )
  AND p.osm_id <> 9742148508
  AND p.name IS NOT NULL;
```

Example 3: amenity/type lookup
```sql
SELECT COALESCE(amenity, shop, tourism, leisure, highway, railway, historic, sport) AS amenity_type
FROM planet_osm_point
WHERE osm_id = 54383999;
```

Example 4: distance in meters
```sql
SELECT ST_Distance(
  ST_Transform(a.way, 4326)::geography,
  ST_Transform(b.way, 4326)::geography
) AS distance_m
FROM planet_osm_point AS a, planet_osm_point AS b
WHERE a.osm_id = 8711071009
  AND b.osm_id = 504754363;
```

Example 5: polygon fallback for a negative osm_id reference
```sql
WITH ref AS (
  SELECT ST_Centroid(way) AS way
  FROM planet_osm_polygon
  WHERE osm_id = -17059953
)
SELECT p.name
FROM planet_osm_point AS p, ref
WHERE p.amenity = 'social_centre'
  AND p.name IS NOT NULL
ORDER BY p.way <-> ref.way
LIMIT 1;
```"""
    return """PostGIS examples:
Example 1: What is the largest park in Tuscon, Arizona?
```sql
SELECT *, ST_Area(parks.geometry::geography) AS computed_area
FROM parks
WHERE leisure = 'park'
  AND ST_Intersects(
    parks.geometry::geography,
    (SELECT geometry FROM regions WHERE wikipedia = 'en:Tucson, Arizona' LIMIT 1)::geography
  )
ORDER BY computed_area DESC
LIMIT 1;
```

Example 2: What is the total area of all gardens in Riverside, California?
```sql
SELECT SUM(ST_Area(parks.geometry::geography)) AS area
FROM parks
WHERE leisure = 'garden'
  AND ST_Intersects(
    parks.geometry::geography,
    (SELECT geometry FROM regions WHERE wikipedia = 'en:Riverside, California' LIMIT 1)::geography
  );

Example 3: What is the elevation at a point?
```sql
SELECT ST_Value(d.rast, 1, ST_GeomFromText('POINT (-121.655715 36.720859)', 4326)) AS elevation_meters
FROM public.dem_us AS d
WHERE ST_Intersects(d.rast, ST_GeomFromText('POINT (-121.655715 36.720859)', 4326));
```

Example 4: What is the maximum elevation inside a named region?
```sql
WITH region_geom AS (
  SELECT geometry::geometry AS geom
  FROM regions
  WHERE region_name ILIKE '%Roseland%'
    AND border_type ILIKE '%city%'
  LIMIT 1
)
SELECT MAX((ST_SummaryStats(ST_Clip(d.rast, 1, r.geom, TRUE), 1, TRUE)).max) AS max_elevation
FROM public.dem_us AS d
CROSS JOIN region_geom AS r
WHERE ST_Intersects(d.rast, r.geom);
```

Example 5: What percentage of a region has elevation above a threshold?
```sql
WITH region_geom AS (
  SELECT geometry::geometry AS geom
  FROM regions
  WHERE region_name ILIKE '%Cumming%'
    AND border_type ILIKE '%city%'
  LIMIT 1
),
clipped AS (
  SELECT ST_Clip(d.rast, 1, r.geom, TRUE) AS rast
  FROM public.dem_us AS d
  CROSS JOIN region_geom AS r
  WHERE ST_Intersects(d.rast, r.geom)
)
SELECT SUM(CASE WHEN (p).val >= 380 THEN 1 ELSE 0 END)::float / COUNT((p).val) AS share
FROM clipped
CROSS JOIN LATERAL ST_PixelAsPoints(rast, 1) AS p;
```

Example 6: Raster type rule
```sql
-- For raster functions, cast vector geography to geometry.
SELECT ST_Intersects(d.rast, r.geometry::geometry)
FROM public.dem_us AS d
JOIN regions AS r ON TRUE
LIMIT 1;
```"""


def get_postgres_connection_kwargs(db_name: str | None = None) -> dict:
    dbname = db_name or os.getenv("CHESS_PG_DATABASE") or os.getenv("PGDATABASE") or "gsqa"
    kwargs = {
        "dbname": dbname,
        "host": os.getenv("CHESS_PG_HOST") or os.getenv("PGHOST") or "localhost",
        "user": os.getenv("CHESS_PG_USER") or os.getenv("PGUSER") or "clockorangezoe",
        "password": os.getenv("CHESS_PG_PASSWORD") or os.getenv("PGPASSWORD") or "",
        "port": int(os.getenv("CHESS_PG_PORT") or os.getenv("PGPORT") or "5432"),
    }
    return kwargs
