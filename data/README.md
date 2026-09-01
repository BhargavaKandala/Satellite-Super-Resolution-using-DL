# Data directories

Contents are git-ignored — satellite scenes are far too large to version.

| Directory | Contents |
|---|---|
| `raw/` | Input Sentinel-2 GeoTIFFs. Bands must be ordered as `data.band_indices` in `configs/config.yaml`. `scripts/prepare_dataset.py --synthetic` writes demo scenes here. |
| `reference/` | Optional co-registered high-resolution reference imagery, for the `full_resolution` evaluation protocol. |
| `processed/` | Intermediate products and dashboard uploads. |
| `patches/` | `patches.npy` (memory-mapped HR patch store) + `manifest.json`. Regenerate with `scripts/prepare_dataset.py`; never edit by hand. |

## Getting real Sentinel-2 data

Level-2A (bottom-of-atmosphere reflectance) products are available free from the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/). You need the four 10 m
bands — B04 (red), B03 (green), B02 (blue), B08 (NIR) — stacked into a single GeoTIFF in that
order:

```bash
gdal_merge.py -separate -o scene.tif \
  T43QGV_..._B04_10m.jp2 T43QGV_..._B03_10m.jp2 \
  T43QGV_..._B02_10m.jp2 T43QGV_..._B08_10m.jp2
```

The pipeline reads scenes windowed, so a full 10980×10980 tile is fine — it is never loaded
into memory in one piece.

## A note on training references

`prepare_dataset.py` treats scenes in `raw/` as the **high-resolution reference** and synthesises
the low-resolution input by degradation. For a scientifically meaningful model you therefore want
reference imagery at `data.target_resolution_m` (2.5 m by default) — e.g. PlanetScope or
WorldView co-registered to your Sentinel-2 footprint.

Training on 10 m scenes as the "reference" still produces a working model, but it learns the
40 m → 10 m step and its performance at 10 m → 2.5 m is an extrapolation. Whichever you do, the
evaluation reports state which protocol produced their numbers.
