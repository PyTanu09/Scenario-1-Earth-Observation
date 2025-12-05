import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------
# Directory Setup
# ---------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
DATA_PATH = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")
DATASET_PATH = os.path.join(OUTPUT_PATH, "datasets")
FIG_PATH = os.path.join(OUTPUT_PATH, "figures")
PATCH_PATH = os.path.join(DATASET_PATH, "patches")

os.makedirs(PATCH_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# ✅ FIXED LANDCOVER FILE NAME
LANDCOVER_PATH = os.path.join(DATA_PATH, "worldcover_bbox_delhi_ncr_2021.tif")

FILTERED_POINTS = os.path.join(DATASET_PATH, "filtered_images.geojson")
LABEL_CSV = os.path.join(DATASET_PATH, "labels.csv")

ESA_LABEL_MAP = {
    10: "Tree",
    20: "Shrub",
    30: "Grass",
    40: "Cropland",
    50: "Built-up",
    60: "BareSoil",
    70: "SnowIce",
    80: "Water",
    90: "Wetland",
    95: "MossLichen",
    100: "Rock"
}

PATCH_SIZE = 128  # pixels

# ----------------------------------------------------------
# Load filtered image center points
# ----------------------------------------------------------

print("Loading filtered geojson points...")
points_gdf = gpd.read_file(FILTERED_POINTS)

if points_gdf.crs is None:
    raise ValueError("ERROR: 'filtered_images.geojson' does not contain CRS information.")

points_gdf = points_gdf.to_crs("EPSG:4326")

# ----------------------------------------------------------
# Open LandCover Raster
# ----------------------------------------------------------

print("Opening landcover raster...")
raster = rasterio.open(LANDCOVER_PATH)
print("Raster CRS  :", raster.crs)
print("Raster NoData value:", raster.nodata)

nodata_val = raster.nodata
labeled_records = []

# ----------------------------------------------------------
# Patch Extraction Loop
# ----------------------------------------------------------

print("Extracting landcover patches...")
for idx, row in points_gdf.iterrows():

    img_name = row.get("filename") if "filename" in row else f"image_{idx}"

    lon, lat = row.geometry.x, row.geometry.y

    try:
        col, row_idx = raster.index(lon, lat)
    except Exception:
        continue

    start_c = col - PATCH_SIZE // 2
    start_r = row_idx - PATCH_SIZE // 2

    if (start_c < 0 or start_r < 0 or
        start_c + PATCH_SIZE > raster.width or
        start_r + PATCH_SIZE > raster.height):
        continue

    window = Window(start_c, start_r, PATCH_SIZE, PATCH_SIZE)
    patch = raster.read(1, window=window)

    valid_mask = (patch != nodata_val) if nodata_val is not None else np.ones_like(patch, bool)

    if valid_mask.sum() < 0.5 * (PATCH_SIZE * PATCH_SIZE):
        continue

    values = patch[valid_mask]

    if values.size == 0:
        continue

    mode_result = stats.mode(values, keepdims=False)
    mode_code = int(mode_result.mode)
    mode_count = int(mode_result.count)
    dominance = mode_count / values.size

    label_name = ESA_LABEL_MAP.get(mode_code, "Other")
    is_mixed = dominance < 0.40

    patch_file = os.path.join(PATCH_PATH, f"{img_name}_lc_patch.npy")
    np.save(patch_file, patch.astype(np.int16))

    labeled_records.append({
        "filename": img_name,
        "lon": lon,
        "lat": lat,
        "esa_code": mode_code,
        "label": label_name,
        "dominance_pct": dominance,
        "mixed": is_mixed,
        "patch_path": patch_file
    })

# ----------------------------------------------------------
# Convert to DataFrame and Save
# ----------------------------------------------------------

df = pd.DataFrame(labeled_records)

df["split"] = np.random.choice(["train", "test"], len(df), p=[0.6, 0.4])
df.to_csv(LABEL_CSV, index=False)

print(f"✓ Saved {len(df)} labeled samples → {LABEL_CSV}")

plt.figure(figsize=(8, 5))
df["label"].value_counts().plot(kind="bar")
plt.title("Label Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "class_distribution.png"), dpi=300)
plt.close()

df["label"].value_counts().to_csv(os.path.join(DATASET_PATH, "class_counts.csv"))

print("✓ Class distribution plot saved.")
print("✓ Patch extraction completed successfully!")
