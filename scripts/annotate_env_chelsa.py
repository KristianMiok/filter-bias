"""
annotate_env_chelsa.py
======================
Environmental annotation WITHOUT a bulk download, using CHELSA V2.1 bioclim served as
Cloud-Optimized GeoTIFFs on the Swiss academic cloud (different host from the blocked
geodata.ucdavis.edu). GDAL/rasterio reads only the German window of each COG over HTTP via
/vsicurl/, so only a few MB move across the network instead of 19 x ~110 MB.

The diagnostic measures standardized differences, which are invariant to CHELSA's integer
scaling/offset, so raw stored values are used directly (no unscaling needed).

Output: reports/odonata_de_annotated.csv with bio_1..bio_19, ready for qfbias_diagnose.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol

IN_CSV = "reports/odonata_de_occurrences.csv"
OUT_CSV = "reports/odonata_de_annotated.csv"
LON_COL, LAT_COL = "decimalLongitude", "decimalLatitude"
BASE = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio"
BUFFER_DEG = 0.5

# GDAL settings that make COG-over-HTTP reads fast and resilient
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_TIMEOUT="60",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="2",
)


def vsicurl(n):
    return f"/vsicurl/{BASE}/CHELSA_bio{n}_1981-2010_V.2.1.tif"


def window_sample(src, lon, lat, minlon, minlat, maxlon, maxlat):
    """Read only the bounding-box window of an open raster and sample points within it.
    Returns a float array (NaN where outside window or nodata)."""
    win = from_bounds(minlon, minlat, maxlon, maxlat, src.transform)
    win = win.round_offsets().round_lengths()
    arr = src.read(1, window=win)
    wt = src.window_transform(win)
    rows, cols = rowcol(wt, lon, lat)
    rows = np.asarray(rows).astype(int)
    cols = np.asarray(cols).astype(int)
    h, w = arr.shape
    inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    vals = np.full(len(lon), np.nan, dtype="float64")
    vals[inb] = arr[rows[inb], cols[inb]].astype("float64")
    nd = src.nodata
    if nd is not None:
        vals = np.where(vals == nd, np.nan, vals)
    vals = np.where((vals < -3.0e38) | (vals >= 65535), np.nan, vals)
    return vals


def sample_chelsa(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    minlon, maxlon = lon.min() - BUFFER_DEG, lon.max() + BUFFER_DEG
    minlat, maxlat = lat.min() - BUFFER_DEG, lat.max() + BUFFER_DEG
    print(f"German window: lon[{minlon:.2f},{maxlon:.2f}] lat[{minlat:.2f},{maxlat:.2f}]")
    out = {}
    with rasterio.Env(**GDAL_ENV):
        for n in range(1, 20):
            url = vsicurl(n)
            try:
                with rasterio.open(url) as src:
                    out[f"bio_{n}"] = window_sample(src, lon, lat, minlon, minlat, maxlon, maxlat)
                print(f"  bio_{n:>2} sampled")
            except Exception as e:
                raise SystemExit(f"failed reading {url}\n  {e}\n"
                                 f"  (if this is a network/host block, tell me and we switch source)")
    return pd.DataFrame(out)


def main():
    df = pd.read_csv(IN_CSV)
    print(f"occurrences: {len(df)} rows from {IN_CSV}")
    df = df.dropna(subset=[LON_COL, LAT_COL]).reset_index(drop=True)

    env = sample_chelsa(df[LON_COL].values, df[LAT_COL].values)
    annotated = pd.concat([df, env], axis=1)
    bio_cols = list(env.columns)
    good = annotated[bio_cols].notna().all(axis=1)
    print(f"\nrecords with complete bioclim: {int(good.sum())} "
          f"(dropped {len(annotated) - int(good.sum())} on nodata/border)")
    annotated = annotated[good].reset_index(drop=True)

    annotated.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(annotated)} annotated records -> {OUT_CSV}")
    lab = annotated["accuracy_class"].notna()
    print(f"  of which {int(lab.sum())} have an accuracy label (diagnostic-ready)")
    print(f"  env feature columns: bio_1 ... bio_19 ({len(bio_cols)} vars)")


if __name__ == "__main__":
    main()
