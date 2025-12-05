import os
import re
import glob
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# === Paths ===
ROOT = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
DS_DIR = os.path.join(OUT_DIR, "datasets")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DS_DIR, exist_ok=True)

# Candidate region files (shapefile or geojson)
CANDIDATES = [
    os.path.join(DATA_DIR, "delhi_ncr.shp"),
    os.path.join(DATA_DIR, "delhi_ncr_region.geojson"),
    os.path.join(DATA_DIR, "delhi_ncr.geojson"),
    os.path.join(DATA_DIR, "delhi_region.geojson")
]

REGION_PATH = None
for p in CANDIDATES:
    if os.path.exists(p):
        REGION_PATH = p
        break

if REGION_PATH is None:
    raise FileNotFoundError(
        "No Delhi region file found. Put delhi_ncr.shp (with .shx/.dbf/.prj) OR delhi_ncr_region.geojson "
        f"in {DATA_DIR} and re-run."
    )

# optional image centers CSV
IMAGE_CSV = os.path.join(DATA_DIR, "image_centers.csv")  # columns: filename,latitude,longitude
RGB_DIR = os.path.join(DATA_DIR, "rgb")  # optional folder with .png RGB tiles

# === 1. Load region file and reproject to EPSG:32644 ===
print("Reading Delhi region file:", REGION_PATH)
region_gdf = gpd.read_file(REGION_PATH)
print("Original CRS:", region_gdf.crs)
# if no CRS set, assume EPSG:4326 as per assignment
if region_gdf.crs is None:
    print("Warning: region file has no CRS. Assuming EPSG:4326.")
    region_gdf.set_crs(epsg=4326, inplace=True)

region_gdf = region_gdf.to_crs("EPSG:32644")  # projection for gridding
print("Reprojected region to EPSG:32644.")

# === 2. Build uniform 60x60 km grid ===
cell_size_m = 60000  # 60 km in meters
minx, miny, maxx, maxy = region_gdf.total_bounds
# expand bounds a bit to ensure coverage
pad = cell_size_m
xs = np.arange(minx - pad, maxx + pad + 1, cell_size_m)
ys = np.arange(miny - pad, maxy + pad + 1, cell_size_m)

cells = []
for x in xs:
    for y in ys:
        cells.append(box(x, y, x + cell_size_m, y + cell_size_m))

grid_gdf = gpd.GeoDataFrame({"geometry": cells}, crs="EPSG:32644")
# keep only grid cells that intersect the region polygon
grid_gdf = grid_gdf[grid_gdf.geometry.intersects(region_gdf.unary_union)].reset_index(drop=True)
print(f"Created grid: {len(grid_gdf)} cells intersecting region.")

# === 3. Plot static map (matplotlib) with corners & centers ===
print("Plotting static map...")
fig, ax = plt.subplots(figsize=(10, 10))
region_gdf.boundary.plot(ax=ax, linewidth=1, edgecolor="black")
grid_gdf.boundary.plot(ax=ax, color="red", linewidth=0.6)
ax.set_title("Delhi region and 60×60 km grid (EPSG:32644)")

# collect corners & centers
corner_points = []
center_points = []
for geom in grid_gdf.geometry:
    x0, y0, x1, y1 = geom.bounds
    corner_points.extend([(x0, y0), (x0, y1), (x1, y0), (x1, y1)])
    center_points.append(((x0 + x1) / 2, (y0 + y1) / 2))

corners_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([p[0] for p in corner_points], [p[1] for p in corner_points]), crs="EPSG:32644")
centers_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([p[0] for p in center_points], [p[1] for p in center_points]), crs="EPSG:32644")

corners_gdf.plot(ax=ax, color="blue", markersize=4, label="corners")
centers_gdf.plot(ax=ax, color="green", markersize=8, label="centers")
ax.legend()
plt.savefig(os.path.join(FIG_DIR, "grid_matplotlib.png"), dpi=300)
plt.savefig(os.path.join(FIG_DIR, "grid_with_points.png"), dpi=300)
plt.close(fig)
print("Saved grid_matplotlib.png and grid_with_points.png to", FIG_DIR)

# === 4. Load image centers (CSV) OR try to parse from filenames in data/rgb ===
images_gdf = None
if os.path.exists(IMAGE_CSV):
    print("Found image_centers.csv — loading centers from CSV.")
    df = pd.read_csv(IMAGE_CSV)
    expected_cols = {"filename", "latitude", "longitude"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"{IMAGE_CSV} must contain columns: {expected_cols}")
    images_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    images_gdf = images_gdf.to_crs("EPSG:32644")
else:
    # Attempt to parse lat/lon from filenames in rgb folder
    if os.path.isdir(RGB_DIR):
        print("No image_centers.csv found. Attempting to parse coordinates from filenames in data/rgb/ ...")
        candidates = glob.glob(os.path.join(RGB_DIR, "*.png")) + glob.glob(os.path.join(RGB_DIR, "*.jpg"))
        parsed = []
        # Common patterns: lat_lon.*, lat-lon.*, lat_lon_name.png, e.g., 28.67_77.2.png or 28.67-77.2_tile.png
        regexes = [
            re.compile(r"([+-]?\d+\.\d+)[_,-]([+-]?\d+\.\d+)"),       # 12.34_56.78 or 12.34-56.78
            re.compile(r"([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"),         # "12.34 56.78"
            re.compile(r"lat[_-]?([+-]?\d+\.\d+)[_,-]?lon[_-]?([+-]?\d+\.\d+)", re.IGNORECASE),
        ]
        for p in candidates:
            fname = os.path.basename(p)
            found = False
            for rx in regexes:
                m = rx.search(fname)
                if m:
                    lat = float(m.group(1))
                    lon = float(m.group(2))
                    parsed.append({"filename": fname, "latitude": lat, "longitude": lon, "path": p})
                    found = True
                    break
            if not found:
                # try reversed order (lon_lat)
                for rx in regexes:
                    m = rx.search(fname[::-1])  # hacky; less reliable — skip
                # continue
        if len(parsed) > 0:
            dfp = pd.DataFrame(parsed)
            images_gdf = gpd.GeoDataFrame(dfp, geometry=gpd.points_from_xy(dfp.longitude, dfp.latitude), crs="EPSG:4326")
            images_gdf = images_gdf.to_crs("EPSG:32644")
            print(f"Parsed coordinates from {len(parsed)} filenames in rgb folder.")
        else:
            print("Could not parse coordinates from filenames in data/rgb/.")
            print("Either provide data/image_centers.csv or use a filename pattern containing lat/lon.")
    else:
        print("No data/rgb/ folder found. Skipping image center filtering.")

# === 5. If we have image centers, spatially join with grid to filter ===
if images_gdf is None:
    print("No image centers available — skipping filtering step. Grid and figures saved.")
else:
    # ensure images are in same CRS as grid (EPSG:32644)
    images_gdf = images_gdf.to_crs("EPSG:32644")
    before_count = len(images_gdf)
    print("Images before filtering:", before_count)

    # spatial join: which images fall within any grid cell
    # add grid id
    grid_indexed = grid_gdf.reset_index().rename(columns={"index": "grid_id"})
    joined = gpd.sjoin(images_gdf, grid_indexed, how="inner", predicate="within")
    after_count = len(joined)
    print("Images after filtering (centers within grid):", after_count)

    # save filtered list
    out_geojson = os.path.join(DS_DIR, "filtered_images.geojson")
    joined.to_file(out_geojson, driver="GeoJSON")
    print("Saved filtered images to", out_geojson)

    # also save a CSV summary
    summary_csv = os.path.join(DS_DIR, "filtered_images_summary.csv")
    joined.drop(columns="geometry").to_csv(summary_csv, index=False)
    print("Saved filtered images summary CSV to", summary_csv)

# === 6. Optional interactive geemap (satellite basemap) ===
try:
    import geemap
    print("Creating interactive geemap (saved as HTML)...")
    # create a base map (must supply center in lat/lon)
    # compute approx center (transform region back to 4326)
    region_4326 = region_gdf.to_crs("EPSG:4326")
    centroid = region_4326.unary_union.centroid
    center_latlon = [centroid.y, centroid.x]
    m = geemap.Map(center=center_latlon, zoom=9)
    m.add_basemap("SATELLITE")
    # add region, grid, and filtered points (convert to 4326)
    m.add_gdf(region_4326)
    m.add_gdf(grid_gdf.to_crs("EPSG:4326"))
    if images_gdf is not None:
        m.add_gdf(images_gdf.to_crs("EPSG:4326"))
    html_out = os.path.join(FIG_DIR, "grid_geemap.html")
    m.to_html(html_out)
    print("Saved interactive map to", html_out)
except Exception as e:
    print("Interactive geemap not created (geemap not installed or failed):", e)
    print("To create interactive map install geemap: pip install geemap")

print("All done.")
