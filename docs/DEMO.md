# Live demo runbook

Exact commands for demonstrating the working prototype to judges, plus what to
say while each one runs and what to do when something breaks.

Speaking script for the slides is in [`PITCH.md`](PITCH.md).

---

## Before you leave for the venue

Do all of this the night before. **The demo must never train live.**

```bash
python scripts/prepare_dataset.py --synthetic --synthetic-scenes 3 --synthetic-size 1024
python scripts/train.py --epochs 12
python scripts/evaluate.py --downstream
python scripts/inference.py --input sample.tif
```

That produces, and you must confirm all four exist:

| Artefact | Path |
|---|---|
| Trained checkpoint | `checkpoints/best.pth` |
| Super-resolved GeoTIFF | `outputs/sample_sr.tif` |
| Uncertainty GeoTIFF | `outputs/sample_uncertainty.tif` |
| Metrics report | `outputs/evaluation/` |

**Pre-flight checklist:**

- [ ] All four artefacts above exist
- [ ] `streamlit run app/dashboard.py` opens in your browser
- [ ] A GeoTIFF opens correctly in QGIS (your offline fallback)
- [ ] Screen-record the whole demo once — your insurance if the laptop misbehaves
- [ ] Screenshots of every panel saved to a folder you can open instantly
- [ ] Laptop on **mains power** (CPU inference throttles hard on battery)
- [ ] Terminal font size raised to ~18pt — judges are sitting 3 metres away
- [ ] Browser zoom at 110–125%

---

## On this machine (WSL)

Windows Smart App Control blocks `torch` and `rasterio` from importing natively,
so everything runs inside WSL. **Open a WSL terminal first and stay in it** — do
not use the `wsl -- bash -lc` wrapper during a live demo, the quoting is fragile.

```bash
wsl -d Ubuntu
cd "/mnt/c/Users/BHARGAVA/OneDrive/Desktop/SIH project/sih142-satellite-sr"
alias py=/opt/sih-venv/bin/python
```

Then `py scripts/...` everywhere below. Set this up **before** the judges arrive.

---

## The demo — 4 minutes

### Step 1 · Show the input is real geospatial data (20 s)

```bash
py -c "
from src.data.geotiff import read_info
print(read_info('sample.tif').summary())
"
```

> "This is a georeferenced GeoTIFF — coordinate reference system, affine
> transform, four spectral bands at 10 metres. Not a PNG."

### Step 2 · Run super-resolution (60 s)

```bash
py scripts/inference.py --input sample.tif
```

While it runs:

> "It's tiling the scene, running each tile through the network with padded
> context so there are no seams, and reassembling. Watch the bottom of the
> output — those are the geospatial validation checks."

The output ends with:

```
  [PASS] has_crs
  [PASS] crs_matches_source
  [PASS] dimensions_scaled
  [PASS] resolution_scaled
  [PASS] bounds_preserved
  [PASS] transform_not_identity
```

> "Six automated checks. The important one is `bounds_preserved` — the output
> covers **exactly** the same ground as the input. Four times the pixels, same
> footprint. Getting that wrong is the most common bug in super-resolution
> pipelines, and it produces a file that opens fine and is silently four times
> too large on the ground."

**This is your strongest 20 seconds. It's verifiable, on screen, live.**

### Step 3 · Prove it in QGIS (30 s, optional but powerful)

Drag `sample.tif` and `outputs/sample_sr.tif` into QGIS. They overlay perfectly.

> "Same coordinates, four times the detail. This is a real GIS product, not a
> picture."

*If any judge has a remote-sensing background, this is the moment they believe
you. Worth the 30 seconds.*

### Step 4 · Show the numbers (45 s)

```bash
py scripts/evaluate.py --downstream
```

> "Bicubic interpolation is our control — it's what you get without any AI.
> We beat it on all five metrics, including both spectral metrics, which means
> we got sharper without distorting the physics.
>
> And then the part we care about most —"

*point at the downstream block*

> "— land-cover classification accuracy, on the original versus on our product.
> It went up. That's evidence the reconstructed detail carries real information,
> not just that it looks nicer."

### Step 5 · The dashboard (90 s)

```bash
py -m streamlit run app/dashboard.py
```

Opens at **http://localhost:8501** in about 2 seconds.

Walk through, in this order:

1. **Imagery tab** — original / SR / difference side by side
2. **Uncertainty tab** — the confidence map
   > "Green is where the model agrees with itself. Red is where it doesn't.
   > This is an uncalibrated relative indicator, not a probability — we're
   > explicit about that."
3. **Geospatial tab** — CRS, transform, bounds preserved
4. **Export tab** — SR GeoTIFF, uncertainty GeoTIFF, metrics JSON

Finish on the warning banner and read it aloud:

> **"Super-resolved imagery contains AI-inferred information and should not be
> interpreted as direct high-resolution observation without validation."**

> "We put that in the product itself, not just in the slides."

---

## Command reference

| Purpose | Command |
|---|---|
| Generate demo data | `py scripts/prepare_dataset.py --synthetic --synthetic-scenes 3 --synthetic-size 1024` |
| Prepare from real scenes | `py scripts/prepare_dataset.py` *(reads `data/raw/*.tif`)* |
| Train | `py scripts/train.py --epochs 12` |
| Train on the DGX | `py scripts/train.py --profile dgx_b200` |
| Evaluate + downstream | `py scripts/evaluate.py --downstream` |
| Super-resolve one scene | `py scripts/inference.py --input sample.tif` |
| Full Sentinel-2 tile | `py scripts/inference.py --input tile.tif --stream` |
| Baseline only (no model) | `py scripts/inference.py --input sample.tif --baseline` |
| Dashboard | `py -m streamlit run app/dashboard.py` |
| Test suite | `py -m pytest -q` |

**Verified timings** (laptop CPU, 8 threads, 256×256 demo scene):

| Step | Time |
|---|---|
| Dataset preparation | ~15 s |
| Training, 12 epochs | ~4 min (17–20 s/epoch) |
| Inference + uncertainty | 3–10 s |
| Evaluation + downstream | ~35 s |
| Dashboard startup | ~2 s |
| Full test suite | ~24 s |

**Retraining moves your numbers by roughly ±0.3 dB PSNR** — cuDNN autotuning and
worker-process augmentation make runs reproducible in distribution, not bitwise.
Re-run `evaluate.py` after your final training run and put *those* figures on the
slide. The bicubic column is deterministic and should never move; if it does,
something is wrong with your data, not your model.

---

## If something breaks

**Rule: never debug in front of judges.** Say "I have the output from an earlier
run" and move to the fallback. Nobody remembers a smooth fallback; everybody
remembers a stack trace.

| Problem | Fallback |
|---|---|
| Anything at all | Your screen recording. Have it open in a background tab. |
| Dashboard won't start | Show `outputs/` GeoTIFFs in QGIS |
| Inference errors | `--baseline` still runs and demonstrates the geospatial pipeline |
| No checkpoint | Dashboard runs in "bicubic baseline only" mode |
| Port 8501 busy | `py -m streamlit run app/dashboard.py --server.port 8502` |
| WSL won't start | Screenshots folder |
| Laptop dies | Have the deck and screenshots on a phone |

**The tests are a fallback too.** If a demo fails, `py -m pytest -q` finishing
with 318 passing in ~24 seconds is a genuinely strong recovery — it shows the
system works and that you engineered it properly.

---

## Three things not to do

**Don't train live.** It takes minutes, it can fail, and it demonstrates nothing
a pre-trained checkpoint doesn't.

**Don't show raw code** unless asked. Judges are evaluating whether the system
works and whether you understand it. If they ask, open `src/models/generator.py`
— the docstring explains all three satellite-specific design decisions.

**Don't oversell the numbers.** If asked whether this is real Sentinel-2 data,
say no, it's synthetic, and it validates the pipeline rather than the science.
That answer is stronger than bluffing and infinitely stronger than being caught.
See [`PITCH.md`](PITCH.md) for the full wording.
