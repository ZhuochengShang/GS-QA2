import glob
import json
import os

import ijson

root = "/home/zshan011/OsmData"


def get_prop(feat, key):
    props = feat.get("properties") or {}
    v = props.get(key)
    if v is not None and str(v).strip() != "":
        return v
    tags = props.get("tagsMap") or {}
    if isinstance(tags, dict):
        v = tags.get(key)
        if v is not None and str(v).strip() != "":
            return v
    return None


def norm(value):
    return " ".join(str(value).lower().split())


def find(layer, target, limit=3):
    targetn = norm(target)
    hits = []
    files = sorted(glob.glob(os.path.join(root, layer, "part-*.geojson")))
    for path in files:
        with open(path, "rb") as f:
            for feat in ijson.items(f, "features.item"):
                name = get_prop(feat, "name") or get_prop(feat, "uname")
                if name and (norm(name) == targetn or targetn in norm(name)):
                    geom = feat.get("geometry") or {}
                    hits.append(
                        {
                            "file": path,
                            "name": name,
                            "addr_state": get_prop(feat, "addr_state"),
                            "amenity": get_prop(feat, "amenity"),
                            "cuisine": get_prop(feat, "cuisine"),
                            "geometry_type": geom.get("type"),
                        }
                    )
                    if len(hits) >= limit:
                        return hits
    return hits


for target in [
    "Charlotte's Quest Nature Center",
    "Turtle Island Park",
    "Hebrew Senior Care",
    "Norman Rockwell Museum",
]:
    print("\nTARGET", target)
    for layer in ["pois", "parks", "lakes"]:
        print(layer, json.dumps(find(layer, target), default=str)[:1200])
