import csv
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO

import language_tool_python
import psycopg2
import psycopg2.extensions
import psycopg2.pool
import shapely
from textblob.blob import Word
from tqdm import tqdm


# DB connection pool + helpers
_db_pool = None


def _init_pool():
    global _db_pool
    if _db_pool is None:
        _db_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host=os.getenv("PGHOST", "localhost"),
            dbname=os.getenv("PGDATABASE", "gsqa"),
            user=os.getenv("PGUSER", os.getenv("USER", "postgres")),
            password=os.getenv("PGPASSWORD", ""),
            port=int(os.getenv("PGPORT", "5432")),
            options="-c statement_timeout=120000 -c client_min_messages=warning",
        )
    return _db_pool


class QueryTimeoutError(Exception):
    """Raised when a SQL statement exceeds statement_timeout."""


def run_sql_select(sql, return_dict=False, timeout_ms=None):
    pool = _init_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            if timeout_ms is not None:
                cur.execute("SET LOCAL statement_timeout = %s", (int(timeout_ms),))
            cur.execute(sql)
            colnames = [desc[0] for desc in cur.description]
            records = cur.fetchall()
        conn.commit()
        if return_dict:
            return [
                {colnames[j]: records[i][j] for j in range(len(colnames)) if records[i][j] is not None}
                for i in range(len(records))
            ]
        return records
    except psycopg2.extensions.QueryCanceledError:
        try:
            conn.rollback()
        except Exception:
            pass
        raise QueryTimeoutError("statement_timeout exceeded")
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        pool.putconn(conn)


BASE_DIR = os.path.dirname(__file__)
output_folder = os.path.join(BASE_DIR, "..", "questions_raster")
output_folder = os.path.abspath(output_folder)
output_folder_extended = os.path.join(output_folder, "extended")
output_folder_vector_only = os.path.join(output_folder, "raster_only")
output_folder_raster_vector = os.path.join(output_folder, "raster_vector")
os.makedirs(output_folder_extended, exist_ok=True)
os.makedirs(output_folder_vector_only, exist_ok=True)
os.makedirs(output_folder_raster_vector, exist_ok=True)
N = 20
ELEVATION_MIN_M = 20
ELEVATION_MAX_M = 500
ELEVATION_STEP_M = 20
SLOPE_MIN_DEG = 2
SLOPE_MAX_DEG = 30

# Timing
TIMING_DIR = os.path.join(BASE_DIR, "timing")
QUESTION_TIMES_COLS = [
    "template_name", "question_index", "sql_execution_seconds",
    "entity_sampling_seconds", "total_seconds", "timestamp", "success",
]
TEMPLATE_SUMMARY_COLS = [
    "template_name", "category", "num_generated", "num_failed", "num_timeout",
    "avg_sql_seconds", "max_sql_seconds", "min_sql_seconds", "total_seconds",
]

try:
    language_tool = language_tool_python.LanguageTool("en-US")
    is_bad_rule = lambda rule: "spelling" in rule.message.lower() and len(rule.replacements) and rule.replacements[0][0].isupper()
except Exception as e:
    language_tool = None
    is_bad_rule = None
    print(f"language_tool init skipped ({e})")

_language_tool_failed = False


pois_selector = {
    "tourism": [
        "aquarium", "attraction", "viewpoint", "art gallery", "theme park", "museum",
        "bed and breakfast", "gallery", "zoo", "hotel"
    ],
    "amenity": ["restaurant", "hospital", "university", "food", "fast food restaurant", "coffee shop", "cafe"],
    "leisure": ["park", "beach resort", "golf course", "nature reserve", "garden", "stadium"],
}

named_pois_selector = {
    "tourism": ["museum", "attraction", "hotel", "gallery"],
    "amenity": ["restaurant", "university", "hospital"],
    "leisure": ["park"],
}

poi_filters = json.loads(open(os.path.join(BASE_DIR, "filter_labels.json"), "r").read())


park_lake_selector = {
    "parks": ["recreation ground", "common", "nature reserve", "park", "garden", "sports centre", "golf course", "marina"],
    "water": ["bay", "harbour", "lake", "reservoir"],
}

road_waterway_selector = {
    "roads": ["road", "footway", "cycleway", "secondary", "pedestrian", "primary", "residential", "track", "motorway"],
    "waterway": ["canal", "river", "stream"],
}

road_only_selector = {
    "roads": ["road", "secondary", "pedestrian", "primary", "track", "motorway"],
}

road_slope_selector = {
    "roads": ["residential", "footway", "cycleway", "pedestrian", "service", "living_street"],
}

regions_selector = ["city", "town", "village", "island", "municipality", "county", "neighbourhood", "suburb", "state"]
regions_selector_small = ["town", "village", "neighbourhood", "suburb"]

sql_no_ref = "SELECT * FROM TABLE2 TABLESAMPLE SYSTEM(5) WHERE PREDICATE MUST_HAVE IS NOT NULL LIMIT 1"
sql_with_ref = "SELECT * FROM TABLE2 TABLESAMPLE SYSTEM(5) WHERE PREDICATE MUST_HAVE IS NOT NULL AND ST_DWithin(TABLE.geometry, ST_GeomFromText('WKT',4326)::geography, 1E5) LIMIT 1"


def poi_category_to_sql_name(value):
    if " " in value:
        if value == "coffee shop":
            value = "coffee"
        elif value == "fast food restaurant":
            value = "fast_food"
        else:
            value = value.replace(" ", "_")
    return value


def safe_pluralize(value):
    """
    Pluralize POI labels more safely than appending 's'.
    Handles multi-word phrases by pluralizing the head noun (last token),
    and supports a few explicit domain overrides.
    """
    if not isinstance(value, str):
        return str(value)

    raw = value.strip().replace("_", " ")
    if not raw:
        return raw

    lower = raw.lower()
    overrides = {
        "fast food restaurant": "fast food restaurants",
        "bed and breakfast": "bed and breakfasts",
        "art gallery": "art galleries",
        "coffee shop": "coffee shops",
        "cafe": "cafes",
        "café": "cafes",
    }
    if lower in overrides:
        return overrides[lower]

    parts = raw.split()
    if len(parts) == 1:
        plural = Word(raw).pluralize()
        if plural:
            return plural
        if raw.endswith("s"):
            return raw
        return raw + "s"

    head = " ".join(parts[:-1])
    tail = parts[-1]
    # Avoid forcing pluralization for known uncountables.
    if tail.lower() in {"equipment", "information", "traffic", "rice", "water"}:
        return raw

    plural_tail = Word(tail).pluralize()
    if not plural_tail:
        if tail.endswith("s"):
            plural_tail = tail
        elif tail.endswith("y") and len(tail) > 1 and tail[-2].lower() not in "aeiou":
            plural_tail = tail[:-1] + "ies"
        else:
            plural_tail = tail + "s"

    return f"{head} {plural_tail}"


def clean_question_text(text):
    """
    Normalize artifacts introduced by grammar correction.
    LanguageTool can rewrite "<=" as a left-arrow glyph in short phrases.
    """
    return (
        text
        .replace("←", "<=")
        .replace("\\u2190", "<=")
        .replace("cafés", "cafes")
        .replace("café", "cafe")
        .replace("Cafés", "Cafes")
        .replace("Café", "Cafe")
    )


def pick_distance_m():
    distance = random.randint(5, 50) * 100
    return distance


def format_distance_text(distance_m):
    if distance_m % 1000 == 0:
        km = distance_m // 1000
        unit = "kilometer" if km == 1 else "kilometers"
        return f"{km} {unit}"
    unit = "meter" if distance_m == 1 else "meters"
    return f"{distance_m} {unit}"


def pick_elevation_threshold():
    return random.choice(list(range(ELEVATION_MIN_M, ELEVATION_MAX_M + 1, ELEVATION_STEP_M)))


def pick_percentile_2_10():
    return random.randint(2, 10)


def pick_high_elevation_threshold():
    return random.choice(list(range(1000, 2001, 100)))


def pick_elevation_threshold_high_realistic():
    return random.choice(list(range(200, 501, 50)))


def pick_proximity_distance_m():
    return random.randint(5, 50) * 100


def pick_slope_threshold_deg():
    return random.randint(SLOPE_MIN_DEG, SLOPE_MAX_DEG)


def pick_slope_flatness_threshold_deg():
    return random.randint(1, 5)


def pick_slope_high_threshold_deg():
    return random.randint(30, 45)


def pick_top_n():
    return random.choice([3, 5, 10])


def pick_elevation_band():
    vals = list(range(1000, 2001, 100))
    low_idx = random.randint(0, len(vals) - 2)
    high_idx = random.randint(low_idx + 1, len(vals) - 1)
    return vals[low_idx], vals[high_idx]


def normalize_sql(sql):
    return re.sub(r"\s+", " ", sql.strip()).lower()


def question_generator(
    text_templates, variable_types, template_tokens, sql_template, answer_type,
    n=100, writer=None, seen_sql_seed=None,
    timing_writer=None, timing_file=None, template_name=None, category=None, timing_stats=None,
    sql_timeout_ms=None,
):
    global language_tool, _language_tool_failed
    generated_questions = []
    generated_count = 0
    progress = tqdm(total=n)
    seen_sql = set(seen_sql_seed or [])
    max_attempts = max(2000, n * 500)
    attempts = 0

    while (len(generated_questions) if writer is None else generated_count) < n:
        attempts += 1
        if attempts > max_attempts:
            print(f"stopping early: reached max attempts ({max_attempts}) with dedup enabled")
            break
        selected_entities = {}
        ref_geo_wkt = None
        geo_wkt = None
        poi_mbr = [1e100, 1e100, -1e100, -1e100]
        skip_iteration = False

        t_entity_start = time.perf_counter()

        for k, v in variable_types:
            if "poi_filter" in v:
                sub_category = random.choice(list(poi_filters.keys()))
                main_category = None
                for kk in pois_selector:
                    if sub_category in pois_selector[kk]:
                        main_category = kk
                        break
                selected_filter = random.choice(poi_filters[sub_category])
                selected_entities[k] = {
                    "main_category": main_category,
                    "sub_category": sub_category,
                    "poi_filter_desc": selected_filter[0],
                    "poi_filter_sql": selected_filter[1],
                }
            elif "poi_named" in v:
                main_category = random.choice(list(named_pois_selector.keys()))
                sub_category = random.choice(named_pois_selector[main_category])
                output = run_sql_select(
                    sql_no_ref.replace("TABLE2", "pois")
                    .replace("MUST_HAVE", "wikidata")
                    .replace(
                        "PREDICATE",
                        "addr_state IS NOT NULL AND %s ILIKE '%s' AND "
                        % (main_category, poi_category_to_sql_name(sub_category)),
                    ),
                    return_dict=True,
                )
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_poi = output[0]
                ref_geo_wkt = shapely.to_wkt(shapely.from_wkb(random_poi["geometry"]))
                display_name = random_poi["poi_name"]
                if "addr_city" in random_poi:
                    display_name += ", " + random_poi["addr_city"]
                if "addr_state" in random_poi:
                    display_name += ", " + random_poi["addr_state"]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_poi["geometry"]))
                p = shapely.from_wkb(random_poi["geometry"])
                poi_mbr = [min(poi_mbr[0], p.x), min(poi_mbr[1], p.y), max(poi_mbr[2], p.x), max(poi_mbr[3], p.y)]
                random_poi["geometry"] = geo_wkt
                selected_entities[k] = {
                    "main_category": main_category,
                    "sub_category": sub_category,
                    "display_name": display_name,
                    "geo_wkt": geo_wkt,
                    "poi": random_poi,
                }
            elif "poi" in v:
                main_category = random.choice(list(pois_selector.keys()))
                sub_category = random.choice(pois_selector[main_category])
                if "distance_limited" in v and ref_geo_wkt is not None:
                    output = run_sql_select(
                        sql_with_ref.replace("TABLE2", "pois")
                        .replace("MUST_HAVE", "addr_state")
                        .replace("WKT", ref_geo_wkt)
                        .replace("PREDICATE", "%s ILIKE '%s' AND " % (main_category, poi_category_to_sql_name(sub_category))),
                        return_dict=True,
                    )
                    if len(output) == 0:
                        skip_iteration = True
                        break
                    random_poi = output[0]
                elif "mbr_limited" in v and poi_mbr is not None:
                    mbr_wkt = shapely.box(*poi_mbr).wkt
                    output = run_sql_select(
                        sql_no_ref.replace("TABLE2", "pois")
                        .replace("MUST_HAVE", "addr_state")
                        .replace("PREDICATE", "ST_Intersects(geometry, ST_GeomFromText('%s',4326)::geography) AND " % mbr_wkt),
                        return_dict=True,
                    )
                    if len(output) == 0:
                        skip_iteration = True
                        break
                    random_poi = output[0]
                    main_category = None
                    sub_category = None
                    for kk in pois_selector:
                        if kk in random_poi:
                            main_category = kk
                            sub_category = random_poi[kk]
                            break
                else:
                    output = run_sql_select(
                        sql_no_ref.replace("TABLE2", "pois")
                        .replace("MUST_HAVE", "addr_state")
                        .replace("PREDICATE", "%s ILIKE '%s' AND " % (main_category, poi_category_to_sql_name(sub_category))),
                        return_dict=True,
                    )
                    if len(output) == 0:
                        skip_iteration = True
                        break
                    random_poi = output[0]
                    ref_geo_wkt = shapely.to_wkt(shapely.from_wkb(random_poi["geometry"]))
                display_name = random_poi["poi_name"]
                if "addr_city" in random_poi:
                    display_name += ", " + random_poi["addr_city"]
                if "addr_state" in random_poi:
                    display_name += ", " + random_poi["addr_state"]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_poi["geometry"]))
                p = shapely.from_wkb(random_poi["geometry"])
                poi_mbr = [min(poi_mbr[0], p.x), min(poi_mbr[1], p.y), max(poi_mbr[2], p.x), max(poi_mbr[3], p.y)]
                random_poi["geometry"] = geo_wkt
                selected_entities[k] = {
                    "main_category": main_category,
                    "sub_category": sub_category,
                    "display_name": display_name,
                    "geo_wkt": geo_wkt,
                    "poi": random_poi,
                }
            elif "region" in v:
                if geo_wkt is None:
                    ss = "SELECT * FROM regions WHERE wikipedia IS NOT NULL AND ST_NPoints(geometry::geometry) < 5000 AND geometry::geometry && ST_MakeEnvelope(-126, 24, -66, 50, 4326) ORDER BY random() LIMIT 1"
                    output = run_sql_select(ss, return_dict=True)
                else:
                    predicate = " ST_Intersects(geometry, ST_GeomFromText('%s',4326)::geography) " % geo_wkt
                    ss = ("SELECT * FROM regions WHERE PREDICATE AND wikipedia IS NOT NULL AND ST_NPoints(geometry::geometry) < 5000 AND geometry::geometry && ST_MakeEnvelope(-126, 24, -66, 50, 4326) ORDER BY random() LIMIT 1").replace("PREDICATE", predicate)
                    output = run_sql_select(ss, return_dict=True)
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_region = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_region["geometry"]))
                random_region["geometry"] = geo_wkt
                selected_entities[k] = {"region_name": random_region["wikipedia"][3:], "geo_wkt": geo_wkt, "region": random_region}
            elif "region_small" in v:
                border_type = random.choice(regions_selector_small)
                if geo_wkt is None:
                    ss = (
                        "SELECT * FROM regions "
                        "WHERE wikipedia IS NOT NULL "
                        f"AND border_type = '{border_type}' "
                        "AND ST_NPoints(geometry::geometry) < 5000 "
                        "AND geometry::geometry && ST_MakeEnvelope(-126, 24, -66, 50, 4326) "
                        "ORDER BY random() LIMIT 1"
                    )
                    output = run_sql_select(ss, return_dict=True)
                else:
                    predicate = " ST_Intersects(geometry, ST_GeomFromText('%s',4326)::geography) " % geo_wkt
                    ss = (
                        "SELECT * FROM regions "
                        "WHERE PREDICATE AND wikipedia IS NOT NULL "
                        f"AND border_type = '{border_type}' "
                        "AND ST_NPoints(geometry::geometry) < 5000 "
                        "AND geometry::geometry && ST_MakeEnvelope(-126, 24, -66, 50, 4326) "
                        "ORDER BY random() LIMIT 1"
                    ).replace("PREDICATE", predicate)
                    output = run_sql_select(ss, return_dict=True)
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_region = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_region["geometry"]))
                random_region["geometry"] = geo_wkt
                selected_entities[k] = {"region_name": random_region["wikipedia"][3:], "geo_wkt": geo_wkt, "region": random_region}
            elif "park_lake" in v:
                __tmp = random.choice(list(park_lake_selector.keys()))
                table = "parks" if __tmp == "parks" else "lakes"
                main_category = "leisure" if table == "parks" else __tmp
                sub_category = random.choice(list(park_lake_selector[__tmp]))
                predicate = "leisure IS NOT NULL AND " if main_category == "leisure" else " (waterway ILIKE '%s' OR water ILIKE '%s') AND " % (sub_category, sub_category)
                predicate = predicate + " ST_Area(geometry) > 0 AND "
                output = run_sql_select(
                    sql_no_ref.replace("TABLE2", table).replace("MUST_HAVE", "wikipedia").replace("PREDICATE", predicate),
                    return_dict=True,
                )
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_obj = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_obj["geometry"]))
                random_obj["geometry"] = geo_wkt
                sub_category_label = random_obj["leisure"] if table == "parks" else random_obj["water"]
                selected_entities[k] = {
                    "main_category": main_category,
                    "sub_category": sub_category,
                    "sub_category_label": sub_category_label,
                    "table": table,
                    Word(table).singularize(): random_obj,
                }
            elif "road_waterway" in v:
                __tmp = random.choice(list(road_waterway_selector.keys()))
                table = "roads" if __tmp == "roads" else "lakes"
                main_category = "highway" if table == "roads" else __tmp
                sub_category = random.choice(list(road_waterway_selector[__tmp]))
                if main_category == "highway" and not (sub_category.endswith("way") or sub_category == "road"):
                    sub_category_label = sub_category + " road"
                else:
                    sub_category_label = sub_category
                predicate = (
                    "(road_name IS NOT NULL OR wikipedia IS NOT NULL) AND highway = '%s' "
                    % sub_category
                    if main_category == "highway"
                    else "(lake_name IS NOT NULL OR wikipedia IS NOT NULL) AND (waterway = '%s' OR water = '%s') "
                    % (sub_category, sub_category)
                )
                ss = """
                SELECT * FROM TABLE2
                TABLESAMPLE SYSTEM(5)
                WHERE PREDICATE
                LIMIT 1;
                """.replace("TABLE2", table).replace("PREDICATE", predicate)
                output = run_sql_select(ss, return_dict=True)
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_obj = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_obj["geometry"]))
                random_obj["geometry"] = geo_wkt
                selected_entities[k] = {
                    "main_category": main_category,
                    "sub_category": sub_category,
                    "sub_category_label": sub_category_label,
                    "table": table,
                    Word(table).singularize(): random_obj,
                }
            elif "road_only" in v:
                table = "roads"
                main_category = "highway"
                sub_category = random.choice(list(road_only_selector["roads"]))
                if not (sub_category.endswith("way") or sub_category == "road"):
                    sub_category_label = sub_category + " road"
                else:
                    sub_category_label = sub_category
                predicate = "(road_name IS NOT NULL OR wikipedia IS NOT NULL) AND highway = '%s' " % sub_category
                ss = """
                SELECT * FROM TABLE2
                TABLESAMPLE SYSTEM(5)
                WHERE PREDICATE
                LIMIT 1;
                """.replace("TABLE2", table).replace("PREDICATE", predicate)
                output = run_sql_select(ss, return_dict=True)
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_obj = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_obj["geometry"]))
                random_obj["geometry"] = geo_wkt
                selected_entities[k] = {
                    "main_category": main_category,
                    "sub_category": sub_category,
                    "sub_category_label": sub_category_label,
                    "table": table,
                    Word(table).singularize(): random_obj,
                }
            elif "distance_proximity_m" in v:
                selected_entities[k] = {"distance": pick_proximity_distance_m()}
            elif "distance_m" in v:
                d = pick_distance_m()
                selected_entities[k] = {"distance": d, "text": format_distance_text(d)}
            elif "elevation_threshold_high_realistic" in v:
                selected_entities[k] = {"elevation": pick_elevation_threshold_high_realistic()}
            elif "elevation_threshold_high" in v:
                selected_entities[k] = {"elevation": pick_high_elevation_threshold()}
            elif "elevation_threshold" in v:
                selected_entities[k] = {"elevation": pick_elevation_threshold()}
            elif "percentile_2_10" in v:
                selected_entities[k] = {"value": pick_percentile_2_10()}
            elif "slope_threshold_flat_deg" in v:
                selected_entities[k] = {"value": pick_slope_flatness_threshold_deg()}
            elif "slope_threshold_high_deg" in v:
                selected_entities[k] = {"value": pick_slope_high_threshold_deg()}
            elif "slope_threshold_deg" in v:
                selected_entities[k] = {"value": pick_slope_threshold_deg()}
            elif "elevation_band_min" in v:
                low, high = pick_elevation_band()
                selected_entities[k] = {"value": low, "paired_high": high}
            elif "elevation_band_max" in v:
                low = selected_entities.get("[2]", {}).get("value")
                high = selected_entities.get("[2]", {}).get("paired_high")
                if low is None or high is None:
                    low, high = pick_elevation_band()
                selected_entities[k] = {"value": high, "paired_low": low}
            elif "road_slope_type" in v:
                slope_road_type = random.choice(road_slope_selector["roads"])
                if not (slope_road_type.endswith("way") or slope_road_type == "road"):
                    sub_category_label = slope_road_type + " road"
                else:
                    sub_category_label = slope_road_type
                selected_entities[k] = {
                    "main_category": "highway",
                    "sub_category": slope_road_type,
                    "sub_category_label": sub_category_label,
                }
            elif "road_slope" in v:
                slope_road_type = random.choice(road_slope_selector["roads"])
                ss = (
                    "SELECT * FROM roads TABLESAMPLE SYSTEM(5) "
                    "WHERE road_name IS NOT NULL "
                    "AND highway = '%s' "
                    "AND ST_Dimension(geometry::geometry) = 1 "
                    "AND ST_NPoints(geometry::geometry) BETWEEN 2 AND 50 "
                    "AND geometry::geometry && ST_MakeEnvelope(-126, 24, -66, 50, 4326) "
                    "LIMIT 1" % slope_road_type
                )
                output = run_sql_select(ss, return_dict=True)
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_road = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_road["geometry"]))
                random_road["geometry"] = geo_wkt
                selected_entities[k] = {"display_name": random_road["road_name"], "geo_wkt": geo_wkt, "road": random_road}
            elif "road" in v:
                ss = (
                    "SELECT * FROM roads TABLESAMPLE SYSTEM(1) "
                    "WHERE road_name IS NOT NULL "
                    "AND ST_NPoints(geometry::geometry) BETWEEN 2 AND 200 "
                    "AND geometry::geometry && ST_MakeEnvelope(-126, 24, -66, 50, 4326) "
                    "LIMIT 1"
                )
                output = run_sql_select(ss, return_dict=True)
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_road = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_road["geometry"]))
                random_road["geometry"] = geo_wkt
                selected_entities[k] = {"display_name": random_road["road_name"], "geo_wkt": geo_wkt, "road": random_road}
            elif "top_n" in v:
                selected_entities[k] = {"value": pick_top_n()}
            else:
                skip_iteration = True
                break

        t_entity_end = time.perf_counter()
        entity_sampling_seconds = t_entity_end - t_entity_start

        # Duplication checks
        # For compare-route templates, ensure two different routes are sampled
        if "[3]" in selected_entities and "[4]" in selected_entities:
            r3 = selected_entities["[3]"].get("road", {})
            r4 = selected_entities["[4]"].get("road", {})
            same_id = r3.get("id") is not None and r3.get("id") == r4.get("id")
            same_geom = selected_entities["[3]"].get("geo_wkt") == selected_entities["[4]"].get("geo_wkt")
            if same_id or same_geom:
                skip_iteration = True

        # For two-POI comparison templates, ensure two different POIs are sampled
        if "[3]" in selected_entities and "[4]" in selected_entities:
            p3 = selected_entities["[3]"].get("poi", {})
            p4 = selected_entities["[4]"].get("poi", {})
            same_id = p3.get("id") is not None and p3.get("id") == p4.get("id")
            same_geom = selected_entities["[3]"].get("geo_wkt") == selected_entities["[4]"].get("geo_wkt")
            if same_id or same_geom:
                skip_iteration = True

        # For compare-categories templates, ensure two different POI types are sampled
        if "[1]" in selected_entities and "[2]" in selected_entities:
            sc1 = selected_entities["[1]"].get("sub_category")
            sc2 = selected_entities["[2]"].get("sub_category")
            if sc1 is not None and sc2 is not None and sc1 == sc2:
                skip_iteration = True

        if skip_iteration:
            continue

        t = random.choice(text_templates)
        sql = sql_template

        for tt in template_tokens:
            if len(tt) == 3:
                k, v, opt = tt
            else:
                k, v = tt
                opt = None
            for _k in selected_entities:
                if _k in v:
                    _value = selected_entities[_k][v[v.find(" ") + 1:]]
                    if k in sql:
                        if "sub_category" in v:
                            value = poi_category_to_sql_name(str(_value))
                        else:
                            value = str(_value)
                        if "name" in k:
                            value = value.replace("'", "''")
                        sql = sql.replace(k, value)
                    if k in t:
                        value = _value.replace("_", " ") if isinstance(_value, str) else str(_value)
                        if opt == "append_an":
                            an = "an " if value and value[0] in "aeoiu" else "a "
                            if "any [1]" in t:
                                an = ""
                            value = an + value
                        elif opt == "pluralize":
                            value = Word(value).pluralize()
                        elif opt == "safe_pluralize":
                            value = safe_pluralize(value)
                        t = t.replace(k, value)

        if language_tool is not None:
            try:
                matches = language_tool.check(t)
                matches = [rule for rule in matches if not is_bad_rule(rule)]
                t = language_tool_python.utils.correct(t, matches)
            except Exception as e:
                # Keep generation running if LT server dies/misbehaves.
                if not _language_tool_failed:
                    print(f"language_tool check failed, disabling grammar correction ({e})")
                _language_tool_failed = True
                language_tool = None
        t = clean_question_text(t)

        t_sql_start = time.perf_counter()
        sql_succeeded = False
        try:
            answers = run_sql_select(sql, return_dict=True, timeout_ms=sql_timeout_ms)
            sql_succeeded = True
        except QueryTimeoutError as e:
            t_sql_end = time.perf_counter()
            if timing_stats is not None:
                timing_stats["num_timeout"] = timing_stats.get("num_timeout", 0) + 1
            if timing_writer is not None:
                timing_writer.writerow({
                    "template_name": template_name or "",
                    "question_index": generated_count,
                    "sql_execution_seconds": round(t_sql_end - t_sql_start, 4),
                    "entity_sampling_seconds": round(entity_sampling_seconds, 4),
                    "total_seconds": round(t_sql_end - t_sql_start + entity_sampling_seconds, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": False,
                })
                if timing_file is not None:
                    timing_file.flush()
            print(f"sql timeout, skipping ({e})")
            continue
        except Exception as e:
            t_sql_end = time.perf_counter()
            if timing_stats is not None:
                timing_stats["num_failed"] = timing_stats.get("num_failed", 0) + 1
            if timing_writer is not None:
                timing_writer.writerow({
                    "template_name": template_name or "",
                    "question_index": generated_count,
                    "sql_execution_seconds": round(t_sql_end - t_sql_start, 4),
                    "entity_sampling_seconds": round(entity_sampling_seconds, 4),
                    "total_seconds": round(t_sql_end - t_sql_start + entity_sampling_seconds, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": False,
                })
                if timing_file is not None:
                    timing_file.flush()
            # Skip bad samples (e.g., invalid topology on random geometries) and continue
            print(f"sql sample failed, retrying ({e})")
            continue
        t_sql_end = time.perf_counter()
        sql_execution_seconds = t_sql_end - t_sql_start

        if len(answers) == 0:
            continue
        if all(len(a) == 0 for a in answers):
            continue
        sql_sig = normalize_sql(sql)
        if sql_sig in seen_sql:
            continue
        for i in range(len(answers)):
            if "geometry" in answers[i]:
                answers[i]["geometry"] = shapely.to_wkt(shapely.from_wkb(answers[i]["geometry"]))
        q = {
            "question": t,
            "sql": sql,
            "answers": answers,
            "answer_type": answer_type,
            "question_entities": selected_entities,
        }
        if writer is None:
            generated_questions.append(q)
            seen_sql.add(sql_sig)
        else:
            writer.write(
                json.dumps(
                    q,
                    default=lambda o: float(o) if isinstance(o, Decimal) else str(o),
                )
                + "\n"
            )
            writer.flush()
            generated_count += 1
            seen_sql.add(sql_sig)

        if timing_writer is not None:
            timing_writer.writerow({
                "template_name": template_name or "",
                "question_index": generated_count,
                "sql_execution_seconds": round(sql_execution_seconds, 4),
                "entity_sampling_seconds": round(entity_sampling_seconds, 4),
                "total_seconds": round(sql_execution_seconds + entity_sampling_seconds, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True,
            })
            if timing_file is not None:
                timing_file.flush()
            if timing_stats is not None:
                timing_stats.setdefault("sql_times", []).append(sql_execution_seconds)

        progress.update(1)
    progress.close()
    return generated_questions if writer is None else generated_count


def check_db():
    try:
        run_sql_select("SELECT 1;")
        print("DB connection: OK")
    except Exception as e:
        print(f"DB connection: FAILED ({e})")
        raise
    try:
        dem_count = run_sql_select("SELECT COUNT(*) FROM public.dem_us;")[0][0]
        print(f"public.dem_us rows: {dem_count}")
    except Exception as e:
        print(f"public.dem_us check FAILED ({e})")
        raise
    try:
        pois_count = run_sql_select("SELECT COUNT(*) FROM pois;")[0][0]
        print(f"pois rows: {pois_count}")
    except Exception as e:
        print(f"pois check FAILED ({e})")
        raise


def load_template_file(path):
    lines = [l.rstrip("\n") for l in open(path, "r").readlines()]
    text_templates = []
    sql_lines = []
    in_sql = False
    answer_type = "unknown"
    for line in lines:
        if line.strip().startswith("-- Answer type:"):
            answer_type = line.split(":", 1)[1].strip()
        if line.strip() == "-- SQL_TEMPLATE_START":
            in_sql = True
            continue
        if line.strip() == "-- SQL_TEMPLATE_END":
            in_sql = False
            continue
        if in_sql:
            if not line.strip().startswith("--"):
                sql_lines.append(line)
        else:
            if line and not line.strip().startswith("--"):
                text_templates.append(line)
    sql_template = "\n".join(sql_lines).strip()
    return text_templates, sql_template, answer_type


TEMPLATE_DIR = os.path.join(BASE_DIR, "templates", "raster_templates", "extend_gs_qa")
TEMPLATE_DIR_VECTOR_ONLY = os.path.join(BASE_DIR, "templates", "raster_templates", "raster_only")
TEMPLATE_DIR_RASTER_VECTOR = os.path.join(BASE_DIR, "templates", "raster_templates", "raster+vector")
TEMPLATE_DIR_EXTENDED_LOCAL = os.path.join(TEMPLATE_DIR, "local")
TEMPLATE_DIR_EXTENDED_FOCAL = os.path.join(TEMPLATE_DIR, "focal")
TEMPLATE_DIR_EXTENDED_ROUTE = os.path.join(TEMPLATE_DIR, "route_based")
TEMPLATE_DIR_EXTENDED_ZONAL = os.path.join(TEMPLATE_DIR, "zonal")
TEMPLATE_DIR_RASTER_ONLY_LOCAL = os.path.join(TEMPLATE_DIR_VECTOR_ONLY, "local")
TEMPLATE_DIR_RASTER_ONLY_FOCAL = os.path.join(TEMPLATE_DIR_VECTOR_ONLY, "focal")
TEMPLATE_DIR_RASTER_ONLY_GLOBAL = os.path.join(TEMPLATE_DIR_VECTOR_ONLY, "global")
TEMPLATE_DIR_RASTER_ONLY_ROUTE = os.path.join(TEMPLATE_DIR_VECTOR_ONLY, "route_based")
TEMPLATE_DIR_RASTER_ONLY_ZONAL = os.path.join(TEMPLATE_DIR_VECTOR_ONLY, "zonal")
TEMPLATE_DIR_RASTER_VECTOR_LOCAL = os.path.join(TEMPLATE_DIR_RASTER_VECTOR, "local")
TEMPLATE_DIR_RASTER_VECTOR_FOCAL = os.path.join(TEMPLATE_DIR_RASTER_VECTOR, "focal")
TEMPLATE_DIR_RASTER_VECTOR_ZONAL = os.path.join(TEMPLATE_DIR_RASTER_VECTOR, "zonal")

TEMPLATES = {
    "range+name+elevation_condition.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "distance_m"), ("[3]", "poi"), ("[4]", "elevation_threshold")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "append_an"),
            ("[2]", "[2] distance"),
            ("[2_text]", "[2] text"),
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
            ("[4]", "[4] elevation"),
        ],
        "output": "range+name+elevation_condition.jsonl",
        "dir": TEMPLATE_DIR_EXTENDED_LOCAL,
    },
    "range+loc+elevation_condition.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "distance_m"), ("[3]", "poi"), ("[4]", "elevation_threshold")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "append_an"),
            ("[2]", "[2] distance"),
            ("[2_text]", "[2] text"),
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
            ("[4]", "[4] elevation"),
        ],
        "output": "range+loc+elevation_condition.jsonl",
        "dir": TEMPLATE_DIR_EXTENDED_LOCAL,
    },
    "range+count+elevation_condition.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "distance_m"), ("[3]", "poi"), ("[4]", "elevation_threshold")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "safe_pluralize"),
            ("[2]", "[2] distance"),
            ("[2_text]", "[2] text"),
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
            ("[4]", "[4] elevation"),
        ],
        "output": "range+count+elevation_condition.jsonl",
        "dir": TEMPLATE_DIR_EXTENDED_LOCAL,
    },
    "range+distance+elevation.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "distance_m"), ("[3]", "poi")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "append_an"),
            ("[2]", "[2] distance"),
            ("[2_text]", "[2] text"),
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "range+distance+elevation.jsonl",
        "dir": TEMPLATE_DIR_EXTENDED_LOCAL,
    },
    "knn+distance+slope.txt": {
        "variable_types": [("[1]", "road_slope_type"), ("[2]", "poi")],
        "template_tokens": [
            ("[1_sql]", "[1] sub_category"),
            ("[1]", "[1] sub_category_label"),
            ("[2]", "[2] display_name"),
            ("[2_wkt]", "[2] geo_wkt"),
        ],
        "output": "knn+distance+slope.jsonl",
        "dir": TEMPLATE_DIR_EXTENDED_ROUTE,
    },
    "intersects:length_max+name+slope.txt": {
        "variable_types": [("[1]", "road_only"), ("[2]", "region_small")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1_table]", "[1] table"),
            ("[1]", "[1] sub_category_label"),
            ("[2]", "[2] region_name"),
            ("[2_wkt]", "[2] geo_wkt"),
        ],
        "output": "intersects:length_max+name+slope.jsonl",
        "dir": TEMPLATE_DIR_EXTENDED_ROUTE,
    },
    "intersects:area_total+area+avg_slope.txt": {
        "variable_types": [("[1]", "park_lake"), ("[2]", "region_small")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1_table]", "[1] table"),
            ("[1]", "[1] sub_category_label", "pluralize"),
            ("[2]", "[2] region_name"),
            ("[2_wkt]", "[2] geo_wkt"),
        ],
        "output": "intersects:area_total+area+avg_slope.jsonl",
        "dir": TEMPLATE_DIR_EXTENDED_FOCAL,
    },
    "intersects:area_max+name+max_elevation.txt": {
        "variable_types": [("[1]", "park_lake"), ("[2]", "region_small")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1_table]", "[1] table"),
            ("[1]", "[1] sub_category_label"),
            ("[2]", "[2] region_name"),
            ("[2_wkt]", "[2] geo_wkt"),
        ],
        "output": "intersects:area_max+name+max_elevation.jsonl",
        "dir": TEMPLATE_DIR_EXTENDED_ZONAL,
    },
}

TEMPLATES_VECTOR_ONLY = {
    "elevation+poi.txt": {
        "variable_types": [("[3]", "poi")],
        "template_tokens": [
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "elevation+poi.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_LOCAL,
    },
    "elevation_max+region.txt": {
        "variable_types": [("[3]", "region")],
        "template_tokens": [
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "elevation_max+region.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ZONAL,
    },
    "elevation_min+region.txt": {
        "variable_types": [("[3]", "region")],
        "template_tokens": [
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "elevation_min+region.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ZONAL,
    },
    "elevation_mean+region.txt": {
        "variable_types": [("[3]", "region")],
        "template_tokens": [
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "elevation_mean+region.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ZONAL,
    },
    "elevation_range+region.txt": {
        "variable_types": [("[3]", "region")],
        "template_tokens": [
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "elevation_range+region.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ZONAL,
    },
    "elevation+coverage.txt": {
        "variable_types": [("[2]", "elevation_threshold"), ("[3]", "region")],
        "template_tokens": [
            ("[2]", "[2] elevation"),
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "elevation+coverage.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ZONAL,
    },
    "avg_slope+area.txt": {
        "variable_types": [("[3]", "region_small")],
        "template_tokens": [
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "avg_slope+area.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ZONAL,
    },
    "min_slope+area.txt": {
        "variable_types": [("[3]", "region_small")],
        "template_tokens": [
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "min_slope+area.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ZONAL,
    },
    "max_slope+area.txt": {
        "variable_types": [("[3]", "region_small")],
        "template_tokens": [
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "max_slope+area.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ZONAL,
    },
    "slope+route.txt": {
        "variable_types": [("[3]", "road_slope")],
        "template_tokens": [
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "slope+route.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ROUTE,
    },
    "slope+compare_route.txt": {
        "variable_types": [("[3]", "road_slope"), ("[4]", "road_slope")],
        "template_tokens": [
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
            ("[4]", "[4] display_name"),
            ("[4_wkt]", "[4] geo_wkt"),
        ],
        "output": "slope+compare_route.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_ROUTE,
    },
}

TEMPLATES_RASTER_VECTOR = {
    "elevation_average+name.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "region_small")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "append_an"),
            ("[2]", "[2] region_name"),
            ("[2_wkt]", "[2] geo_wkt"),
        ],
        "output": "elevation_average+name.jsonl",
        "dir": TEMPLATE_DIR_RASTER_VECTOR_ZONAL,
    },

    "elevation_proximity_zone+name.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "distance_proximity_m"), ("[3]", "elevation_threshold_high"), ("[4]", "region_small")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "safe_pluralize"),
            ("[2]", "[2] distance"),
            ("[3]", "[3] elevation"),
            ("[4]", "[4] region_name"),
            ("[4_wkt]", "[4] geo_wkt"),
        ],
        "output": "elevation_proximity_zone+name.jsonl",
        "dir": TEMPLATE_DIR_RASTER_VECTOR_ZONAL,
    },

    "slope_steepest_n_poi+name.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "region_small"), ("[3]", "top_n")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "safe_pluralize"),
            ("[2]", "[2] region_name"),
            ("[2_wkt]", "[2] geo_wkt"),
            ("[3]", "[3] value"),
        ],
        "output": "slope_steepest_n_poi+name.jsonl",
        "dir": TEMPLATE_DIR_RASTER_VECTOR_FOCAL,
    },

    "elevation_count_threshold+count.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "region_small"), ("[3]", "elevation_threshold_high_realistic")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "safe_pluralize"),
            ("[2]", "[2] region_name"),
            ("[2_wkt]", "[2] geo_wkt"),
            ("[3]", "[3] elevation"),
        ],
        "output": "elevation_count_threshold+count.jsonl",
        "dir": TEMPLATE_DIR_RASTER_VECTOR_LOCAL,
    },
    "elevation_compare_categories+name.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "poi"), ("[3]", "region_small")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "safe_pluralize"),
            ("[2_type]", "[2] main_category"),
            ("[2]", "[2] sub_category", "safe_pluralize"),
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "elevation_compare_categories+name.jsonl",
        "dir": TEMPLATE_DIR_RASTER_VECTOR_LOCAL,
    },
    "slope_condition+name.txt": {
        "variable_types": [("[1]", "poi"), ("[2]", "slope_threshold_high_deg"), ("[3]", "region_small")],
        "template_tokens": [
            ("[1_type]", "[1] main_category"),
            ("[1]", "[1] sub_category", "safe_pluralize"),
            ("[2]", "[2] value"),
            ("[3]", "[3] region_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "slope_condition+name.jsonl",
        "dir": TEMPLATE_DIR_RASTER_VECTOR_FOCAL,
    },
}

TEMPLATES_RASTER_POINT = {
    "elevation_threshold+poi.txt": {
        "variable_types": [("[2]", "elevation_threshold"), ("[3]", "poi_named")],
        "template_tokens": [
            ("[2]", "[2] elevation"),
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "elevation_threshold+poi.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_LOCAL,
    },
    "elevation_compare+two_pois.txt": {
        "variable_types": [("[3]", "poi_named"), ("[4]", "poi_named")],
        "template_tokens": [
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
            ("[4]", "[4] display_name"),
            ("[4_wkt]", "[4] geo_wkt"),
        ],
        "output": "elevation_compare+two_pois.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_LOCAL,
    },
    "elevation_diff+two_pois.txt": {
        "variable_types": [("[3]", "poi_named"), ("[4]", "poi_named")],
        "template_tokens": [
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
            ("[4]", "[4] display_name"),
            ("[4_wkt]", "[4] geo_wkt"),
        ],
        "output": "elevation_diff+two_pois.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_LOCAL,
    },
    "slope+poi.txt": {
        "variable_types": [("[3]", "poi_named")],
        "template_tokens": [
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "slope+poi.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_FOCAL,
    },
    "aspect+poi.txt": {
        "variable_types": [("[3]", "poi_named")],
        "template_tokens": [
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "aspect+poi.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_FOCAL,
    },
    "ruggedness+poi.txt": {
        "variable_types": [("[3]", "poi_named")],
        "template_tokens": [
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "ruggedness+poi.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_FOCAL,
    },
    "nearest_high_terrain+poi.txt": {
        "variable_types": [("[2]", "elevation_threshold_high"), ("[3]", "poi_named")],
        "template_tokens": [
            ("[2]", "[2] elevation"),
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "nearest_high_terrain+poi.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_GLOBAL,
    },
    "nearest_low_terrain+poi.txt": {
        "variable_types": [("[2]", "elevation_threshold"), ("[3]", "poi_named")],
        "template_tokens": [
            ("[2]", "[2] elevation"),
            ("[3]", "[3] display_name"),
            ("[3_wkt]", "[3] geo_wkt"),
        ],
        "output": "nearest_low_terrain+poi.jsonl",
        "dir": TEMPLATE_DIR_RASTER_ONLY_GLOBAL,
    },
}


def _run_template_group(template_dict, output_dir, category, qt_writer, qt_file, ts_writer, ts_file):
    """Run generation for one group of templates, writing timing CSVs as we go."""
    for filename, cfg in template_dict.items():
        tmpl_dir = cfg.get("dir", TEMPLATE_DIR)
        path = os.path.join(tmpl_dir, filename)
        out_path = os.path.join(output_dir, cfg["output"])
        existing_count = 0
        existing_sql = set()
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    existing_count += 1
                    try:
                        obj = json.loads(line)
                        if "sql" in obj:
                            existing_sql.add(normalize_sql(obj["sql"]))
                    except Exception:
                        pass
        if existing_count >= N:
            print(f"skip complete ({existing_count}/{N}) -> {out_path}")
            continue
        print(f"start {filename}")
        text_templates, sql_template, answer_type = load_template_file(path)
        remaining = N - existing_count
        mode = "a" if existing_count > 0 else "w"
        timing_stats = {"num_timeout": 0, "num_failed": 0, "sql_times": []}
        t_template_start = time.perf_counter()
        with open(out_path, mode) as f:
            wrote = question_generator(
                text_templates=text_templates,
                variable_types=cfg["variable_types"],
                template_tokens=cfg["template_tokens"],
                sql_template=sql_template,
                answer_type=answer_type,
                n=remaining,
                writer=f,
                seen_sql_seed=existing_sql,
                timing_writer=qt_writer,
                timing_file=qt_file,
                template_name=filename.removesuffix(".txt"),
                category=category,
                timing_stats=timing_stats,
                sql_timeout_ms=cfg.get("sql_timeout_ms"),
            )
        t_template_end = time.perf_counter()
        sql_times = timing_stats.get("sql_times", [])
        ts_writer.writerow({
            "template_name": filename.removesuffix(".txt"),
            "category": category,
            "num_generated": wrote,
            "num_failed": timing_stats.get("num_failed", 0),
            "num_timeout": timing_stats.get("num_timeout", 0),
            "avg_sql_seconds": round(sum(sql_times) / len(sql_times), 4) if sql_times else 0,
            "max_sql_seconds": round(max(sql_times), 4) if sql_times else 0,
            "min_sql_seconds": round(min(sql_times), 4) if sql_times else 0,
            "total_seconds": round(t_template_end - t_template_start, 4),
        })
        ts_file.flush()
        print(f"wrote {wrote} (now {existing_count + wrote}/{N}) -> {out_path}")


def main():
    check_db()

    os.makedirs(TIMING_DIR, exist_ok=True)
    qt_path = os.path.join(TIMING_DIR, "question_times.csv")
    ts_path = os.path.join(TIMING_DIR, "template_summary.csv")
    qt_is_new = not os.path.exists(qt_path) or os.path.getsize(qt_path) == 0
    ts_is_new = not os.path.exists(ts_path) or os.path.getsize(ts_path) == 0

    with (
        open(qt_path, "a", newline="") as qt_f,
        open(ts_path, "a", newline="") as ts_f,
    ):
        qt_writer = csv.DictWriter(qt_f, fieldnames=QUESTION_TIMES_COLS)
        ts_writer = csv.DictWriter(ts_f, fieldnames=TEMPLATE_SUMMARY_COLS)
        if qt_is_new:
            qt_writer.writeheader()
        if ts_is_new:
            ts_writer.writeheader()

        # Extended templates keep their category-specific source directories
        extended_dict = {k: dict(v, dir=v.get("dir", TEMPLATE_DIR)) for k, v in TEMPLATES.items()}
        _run_template_group(extended_dict, output_folder_extended, "extended", qt_writer, qt_f, ts_writer, ts_f)

        # Raster-only templates
        _run_template_group(TEMPLATES_VECTOR_ONLY, output_folder_vector_only, "raster_only", qt_writer, qt_f, ts_writer, ts_f)

        # Raster+vector templates
        _run_template_group(TEMPLATES_RASTER_VECTOR, output_folder_raster_vector, "raster_vector", qt_writer, qt_f, ts_writer, ts_f)

        # Raster-point templates are raster-only outputs with POIs used as coordinate anchors
        _run_template_group(TEMPLATES_RASTER_POINT, output_folder_vector_only, "raster_only", qt_writer, qt_f, ts_writer, ts_f)


if __name__ == "__main__":
    main()