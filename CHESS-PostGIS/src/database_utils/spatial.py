import math
import re
import sqlite3
import struct
from typing import Optional, Sequence, Tuple


_POINT_RE = re.compile(
    r"^\s*POINT\s*\(\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*\)\s*$",
    flags=re.IGNORECASE,
)


def _as_bytes(blob_value) -> Optional[bytes]:
    if blob_value is None:
        return None
    if isinstance(blob_value, bytes):
        return blob_value
    if isinstance(blob_value, memoryview):
        return blob_value.tobytes()
    return None


def _parse_wkb_point(blob_value) -> Optional[Tuple[float, float]]:
    data = _as_bytes(blob_value)
    if not data or len(data) < 5:
        return None

    def read_uint32(offset: int, endian: str) -> Tuple[int, int]:
        return struct.unpack(f"{endian}I", data[offset:offset + 4])[0], offset + 4

    def read_point(offset: int, endian: str) -> Tuple[Tuple[float, float], int]:
        x = struct.unpack(f"{endian}d", data[offset:offset + 8])[0]
        y = struct.unpack(f"{endian}d", data[offset + 8:offset + 16])[0]
        return (x, y), offset + 16

    def parse_geom(offset: int) -> Tuple[Optional[Tuple[float, float]], int]:
        if offset + 5 > len(data):
            return None, offset
        byte_order = data[offset]
        if byte_order == 1:
            endian = "<"
        elif byte_order == 0:
            endian = ">"
        else:
            return None, offset

        geom_type, offset = read_uint32(offset + 1, endian)
        base_type = geom_type % 1000

        if base_type == 1:
            if offset + 16 > len(data):
                return None, offset
            point, offset = read_point(offset, endian)
            return point, offset

        if base_type == 2:
            n_points, offset = read_uint32(offset, endian)
            if n_points <= 0:
                return None, offset
            first = None
            for i in range(n_points):
                pt, offset = read_point(offset, endian)
                if i == 0:
                    first = pt
            return first, offset

        if base_type == 3:
            n_rings, offset = read_uint32(offset, endian)
            lon_sum = 0.0
            lat_sum = 0.0
            count = 0
            for _ in range(n_rings):
                n_points, offset = read_uint32(offset, endian)
                for _ in range(n_points):
                    pt, offset = read_point(offset, endian)
                    lon_sum += pt[0]
                    lat_sum += pt[1]
                    count += 1
            if count == 0:
                return None, offset
            return (lon_sum / count, lat_sum / count), offset

        if base_type in {4, 5, 6, 7}:
            n_geoms, offset = read_uint32(offset, endian)
            lon_sum = 0.0
            lat_sum = 0.0
            count = 0
            for _ in range(n_geoms):
                pt, offset = parse_geom(offset)
                if pt is None:
                    continue
                lon_sum += pt[0]
                lat_sum += pt[1]
                count += 1
            if count == 0:
                return None, offset
            return (lon_sum / count, lat_sum / count), offset

        return None, offset

    try:
        point, _ = parse_geom(0)
        return point
    except (struct.error, IndexError):
        return None


def _parse_point_text(value: str) -> Optional[Tuple[float, float]]:
    if not value:
        return None
    match = _POINT_RE.match(value)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def parse_geometry_to_lon_lat(geometry_value) -> Optional[Tuple[float, float]]:
    coords = _parse_wkb_point(geometry_value)
    if coords is not None:
        return coords

    if isinstance(geometry_value, str):
        text = geometry_value.strip()
        coords = _parse_point_text(text)
        if coords is not None:
            return coords
        try:
            hex_bytes = bytes.fromhex(text)
        except ValueError:
            return None
        return _parse_wkb_point(hex_bytes)

    return None


def _haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    earth_radius_m = 6371008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return 2.0 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def st_point(lon, lat):
    if lon is None or lat is None:
        return None
    try:
        return f"POINT({float(lon)} {float(lat)})"
    except (TypeError, ValueError):
        return None


def geom_from_wkb(blob_value, _srid=None):
    coords = parse_geometry_to_lon_lat(blob_value)
    if coords is None:
        return None
    lon, lat = coords
    return f"POINT({lon} {lat})"


def geom_from_text(text, _srid=None):
    if text is None:
        return None
    coords = _parse_point_text(str(text))
    if coords is None:
        return str(text)
    lon, lat = coords
    return f"POINT({lon} {lat})"


def st_distance(geom_a, geom_b):
    a = parse_geometry_to_lon_lat(geom_a)
    b = parse_geometry_to_lon_lat(geom_b)
    if a is None or b is None:
        return None
    return _haversine_meters(a[0], a[1], b[0], b[1])


def register_spatial_functions(conn: sqlite3.Connection) -> None:
    conn.create_function("ST_Distance", 2, st_distance)
    conn.create_function("ST_Point", 2, st_point)
    conn.create_function("GeomFromWKB", 2, geom_from_wkb)
    conn.create_function("ST_GeomFromWKB", 2, geom_from_wkb)
    conn.create_function("GeomFromText", 2, geom_from_text)
    conn.create_function("ST_GeomFromText", 2, geom_from_text)


def _get_table_names(conn: sqlite3.Connection) -> Sequence[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    return [row[0] for row in rows]


def _get_columns(conn: sqlite3.Connection, table_name: str) -> Sequence[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info(`{table_name}`)").fetchall()]


def ensure_lon_lat_columns(db_path: str, table_names: Optional[Sequence[str]] = None, batch_size: int = 10000) -> None:
    conn = sqlite3.connect(db_path, timeout=120)
    try:
        tables = list(table_names) if table_names else list(_get_table_names(conn))
        for table_name in tables:
            columns = _get_columns(conn, table_name)
            if "geometry" not in columns:
                continue
            if "longitude" not in columns:
                conn.execute(f"ALTER TABLE `{table_name}` ADD COLUMN longitude REAL")
            if "latitude" not in columns:
                conn.execute(f"ALTER TABLE `{table_name}` ADD COLUMN latitude REAL")
            conn.commit()

            last_rowid = 0
            while True:
                rows = conn.execute(
                    f"""
                    SELECT rowid, geometry
                    FROM `{table_name}`
                    WHERE geometry IS NOT NULL
                      AND (longitude IS NULL OR latitude IS NULL)
                      AND rowid > ?
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    (last_rowid, batch_size),
                ).fetchall()
                if not rows:
                    break

                updates = []
                for rowid, geometry in rows:
                    coords = parse_geometry_to_lon_lat(geometry)
                    if coords is None:
                        continue
                    updates.append((coords[0], coords[1], rowid))

                if updates:
                    conn.executemany(
                        f"UPDATE `{table_name}` SET longitude = ?, latitude = ? WHERE rowid = ?",
                        updates,
                    )
                    conn.commit()
                last_rowid = rows[-1][0]
    finally:
        conn.close()
