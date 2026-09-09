"""
survey2_annotate.py
===================
SURVEY, step 2 — CHELSA annotation for every pulled group, generalizing the
proven annotate_env_chelsa.py (COG-over-HTTP windowed reads; no bulk download).

Reads the German window of CHELSA V2.1 bio1..bio19 ONCE, caches the stack to
data/chelsa_de_window.npz, then annotates every data/survey/<slug>_occurrences.csv
to <slug>_annotated.csv. The cached stack is reused by survey3 for disk-mean
calibration and the layer variogram, so this network step runs exactly once.

Run:   python scripts/survey2_annotate.py
Requires rasterio (already used by annotate_env_chelsa.py).
"""
import glob
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "survey")
CACHE = os.path.join(ROOT, "data", "chelsa_de_window.npz")

BASE = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio"
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = 4.7, 46.7, 16.0, 55.6   # DE + buffer
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2")


def vsicurl(n):
    return f"/vsicurl/{BASE}/CHELSA_bio{n}_1981-2010_V.2.1.tif"


def fetch_stack():
    import rasterio
    from rasterio.windows import from_bounds
    layers, meta = [], None
    with rasterio.Env(**GDAL_ENV):
        for n in range(1, 20):
            t0 = time.time()
            with rasterio.open(vsicurl(n)) as src:
                win = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
                win = win.round_offsets().round_lengths()
                arr = src.read(1, window=win).astype(np.float32)
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                wt = src.window_transform(win)
                if meta is None:
                    meta = dict(x0=wt.c, dx=wt.a, y0=wt.f, dy=wt.e)  # north-up: dy < 0
                layers.append(arr)
            print(f"  bio{n:2d}  {arr.shape}  {time.time()-t0:.1f}s")
    stack = np.stack(layers, axis=0)
    np.savez_compressed(CACHE, stack=stack, **meta)
    return stack, meta


def load_stack():
    z = np.load(CACHE)
    return z["stack"], dict(x0=float(z["x0"]), dx=float(z["dx"]),
                            y0=float(z["y0"]), dy=float(z["dy"]))


def sample_points(stack, meta, lon, lat):
    """Nearest-cell sample of every layer at (lon, lat); NaN outside window."""
    col = np.floor((lon - meta["x0"]) / meta["dx"]).astype(int)
    row = np.floor((lat - meta["y0"]) / meta["dy"]).astype(int)   # dy negative
    H, W = stack.shape[1:]
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    out = np.full((len(lon), stack.shape[0]), np.nan, dtype=np.float32)
    if ok.any():
        out[ok] = stack[:, row[ok], col[ok]].T
    return out


def main():
    if os.path.exists(CACHE):
        print(f"cache found: {CACHE}")
        stack, meta = load_stack()
    else:
        print("fetching CHELSA DE window (runs once) ...")
        stack, meta = fetch_stack()
    print(f"stack: {stack.shape}   NaN frac: {np.isnan(stack).mean():.3f}")

    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*_occurrences.csv"))):
        slug = os.path.basename(path).replace("_occurrences.csv", "")
        out = os.path.join(RAW_DIR, f"{slug}_annotated.csv")
        if os.path.exists(out):
            print(f"[{slug}] annotated exists, skipping"); continue
        d = pd.read_csv(path, low_memory=False)
        vals = sample_points(stack, meta,
                             d.decimalLongitude.values.astype(float),
                             d.decimalLatitude.values.astype(float))
        for i in range(19):
            d[f"bio_{i+1}"] = vals[:, i]
        n_ok = int(np.isfinite(vals).all(axis=1).sum())
        d.to_csv(out, index=False)
        print(f"[{slug}] {len(d)} records, complete bio for {n_ok} ({n_ok/len(d):.1%}) -> {out}")


if __name__ == "__main__":
    main()
