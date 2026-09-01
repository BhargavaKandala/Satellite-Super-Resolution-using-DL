# Adding real Sentinel-2 data

This is the single highest-value change available to the project. Everything
currently in `data/raw/` is synthetic, so every metric measures whether the
pipeline is wired correctly — not whether the science works. Real data converts
those numbers from plumbing checks into evidence.

Budget about an hour, most of it download time.

---

## 1. Download an L2A product

**[Copernicus Browser](https://browser.dataspace.copernicus.eu/)** — free, needs
a registration.

1. Search → **Sentinel-2** → **L2A** (not L1C: L2A is atmospherically corrected
   surface reflectance, which is what the project assumes)
2. Filter **cloud cover < 10%**
3. Pick an area with visible structure — a city edge, farmland with field
   boundaries, a coastline. Uniform forest or open ocean gives the model nothing
   to learn and makes an unconvincing demo.
4. Download the full product (`.SAFE`, ~1 GB zipped)

Good Indian scenes: Hyderabad, Pune, the Krishna delta, Chandigarh's grid.

> **L2A, not L1C.** L1C is top-of-atmosphere radiance. The pipeline treats input
> as surface reflectance, and mixing the two silently changes what every spectral
> metric means.

---

## 2. Import it

```bash
python scripts/import_sentinel2.py --input S2A_MSIL2A_20230115T051121_....SAFE --size 4096
```

A full tile is 10980², which is more than you need — `--size` takes a centre
crop. 4096² at 10 m is ~41 km across and plenty for training.

Check what it found before committing:

```bash
python scripts/import_sentinel2.py --input <product>.SAFE --list
```

Output:

```
band order (from config, NOT filename order):
  1. B04   <- T43PGQ_20230115T051121_B04_10m.jp2
  2. B03   <- T43PGQ_20230115T051121_B03_10m.jp2
  3. B02   <- T43PGQ_20230115T051121_B02_10m.jp2
  4. B08   <- T43PGQ_20230115T051121_B08_10m.jp2

radiometric offset: -1000 DN
  source: MTD_MSIL2A.xml (baseline >= 04.00)
```

### The two things this exists to prevent

Both produce a file that opens fine and looks plausible. Neither is visible in a
preview image. Do not stack the bands by hand.

**Band order.** The pipeline reads bands *positionally* — `data.band_indices`
maps position to meaning. Sentinel-2's filenames sort **B02, B03, B04, B08**
(blue, green, red, NIR) while `config.yaml` asks for **B04, B03, B02, B08**
(red, green, blue, NIR). Stack them in filename order and red and blue are
swapped: NDVI still computes, the image still looks like a satellite image, and
every spectral metric is quietly measuring the wrong thing.

**The L2A radiometric offset.** From processing baseline 04.00 — products
acquired after **2022-01-25** — L2A carries `BOA_ADD_OFFSET = -1000`:

```
reflectance = (DN - 1000) / 10000        NOT   DN / 10000
```

Miss it and every reflectance is 0.1 too high. Clipping at 1.0 hides the
overflow. The importer reads the offset from `MTD_MSIL2A.xml` and bakes it into
the output, so the rest of the pipeline keeps the plain `DN / 10000` convention
and `data/raw` holds one harmonised representation. What it did is recorded in
the GeoTIFF tags (`BOA_ADD_OFFSET`, `OFFSET_APPLIED`).

---

## 3. Train and evaluate

Clear the synthetic scenes first — mixing real and synthetic data would make the
metrics meaningless:

```bash
mkdir -p data/archive && mv data/raw/synthetic_*.tif data/archive/
rm -rf data/patches/*

python scripts/prepare_dataset.py
python scripts/train.py --epochs 40
python scripts/evaluate.py --downstream
```

Expect **more patches and slower epochs** than the synthetic demo — a 4096²
scene yields far more than 363 patches. Start with `--epochs 40`; if it's too
slow on CPU, this is the point where the DGX earns its keep
([`gpu-runbook.md`](gpu-runbook.md)).

---

## What changes scientifically

This is the part to understand before you present it.

**What the model actually learns.** There is no free 2.5 m reference
co-registered with Sentinel-2, so training uses **Wald's protocol**: your real
10 m observation becomes the high-resolution reference, it is degraded to 40 m,
and the model learns **40 m → 10 m**. Ground truth is a genuine satellite
observation.

**What it is then asked to do.** At inference the same 4× model is applied to
10 m input to produce 2.5 m. That relies on **scale invariance** — the
assumption that the relationship between 40 m and 10 m also holds between 10 m
and 2.5 m.

That assumption is standard in the super-resolution literature, and it is an
assumption rather than a measurement. `evaluate.py` already prints it:

```
CAVEATS:
  - Metrics come from Wald's reduced-resolution protocol: the model was
    evaluated on a 4x step at coarser scale, not on the actual 10 m -> 2.5 m product.
  - Performance at the operational scale is assumed, not measured.
```

**Say this out loud to the judges.** It is the difference between a team that
understands remote sensing and one that ran a model.

**What improves the moment you switch:** your metrics stop describing synthetic
textures and start describing real land cover, atmospheric effects, sensor noise
and genuine spatial statistics. The numbers will probably get *worse* — real
imagery is harder than synthetic. That is a good sign, not a bad one.

---

## Going further

**A high-resolution reference** unlocks the strongest validation tier
(`--protocol full_resolution`), which measures the actual 10 m → 2.5 m task
instead of assuming it. Options: PlanetScope (3 m, free education licence via
Planet's NICFI or education programme), or an aerial orthophoto from Bhuvan /
your state GIS portal. It must be co-registered and close in acquisition date.

**Land-cover labels** turn the downstream experiment from unsupervised k-means
into a quantitatively valid accuracy comparison. Bhuvan publishes LULC for
India; ESA WorldCover is global at 10 m. Point `application.labels_path` at a
rasterised label GeoTIFF on the same grid.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `band B03 not found` | L1C, or an incomplete download | Confirm it is L2A with an `R10m` folder |
| `not Sentinel-2 10 m bands` | `data.bands` lists B11/B12 etc. | Only B02/B03/B04/B08 are native 10 m |
| `over half the scene is nodata` | Granule edge | Re-crop with `--size`, or pick another granule |
| Reflectance looks 0.1 too high | Offset not applied | Check the `OFFSET_APPLIED` tag |
| `bands must share the 10 m grid` | Mixed resolutions | Point at the `R10m` folder specifically |
| Training much slower | Real scenes yield far more patches | Lower `patches.max_patches`, or use the GPU |
