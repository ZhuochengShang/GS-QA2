from pathlib import Path

p = Path("SpatialAnalysisAgent_helper.py")
s = p.read_text()
old = '''            "- Stream the GeoJSON shards with ijson and glob using the exact part-*.geojson patterns from the prompt.\\n"
            "- Convert GeoJSON geometries with shapely.shape only after filtering/selecting candidate features.\\n"
'''
new = '''            "- Set data_root from os.environ.get('SPATIALAGENT_DATA_PATH') or os.environ.get('GS_DATA_ROOT'), falling back only to the first Runtime data path above; never hard-code './OsmData' or '/home/zshan011/OsmData'.\\n"
            "- Stream the GeoJSON shards with ijson and glob using os.path.join(data_root, layer, 'part-*.geojson') for the runtime data_root.\\n"
            "- Convert GeoJSON geometries with shapely.shape only after filtering/selecting candidate features.\\n"
'''
if old not in s:
    raise SystemExit("target prompt block not found")
p.write_text(s.replace(old, new))
print("patched prompt data_root requirement")
