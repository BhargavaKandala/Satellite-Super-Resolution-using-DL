# Project Context — SIH 2026, Problem Statement 142

**Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries**

This document holds the *why* of the project: the thesis, the framing, the roadmap
and the language we hold ourselves to. It is deliberately separate from the code.
`README.md` explains how the system works; this explains what it is for and what
we refuse to claim about it.

---

## 1. The thesis

> Convert 10 m Sentinel-2 imagery into a scientifically reliable **sub-4 m**
> product containing useful, geographically consistent fine-scale information —
> while explicitly indicating which details are **observed** and which are
> **AI-inferred**.

This is not an image-enhancement project. If it were, an off-the-shelf ESRGAN
checkpoint would already have solved it.

### Why ordinary upscaling is not enough

A single Sentinel-2 pixel covers 10 m × 10 m of ground. That footprint may
contain a building, a tree and a stretch of road all at once, integrated into one
reflectance value. A generic super-resolution network will happily invent a crisp
boundary between them.

The result *looks* excellent and may be **geographically wrong**. In remote
sensing that is not a cosmetic flaw — a hallucinated road edge in a flood map is
a wrong answer delivered confidently.

So the framing is:

| Not this | This |
| --- | --- |
| Image enhancement | AI-based information reconstruction under uncertainty |
| "Make it sharper" | "What fine-scale information can we defensibly reconstruct, and how confident are we?" |
| Judged on visual appeal | Judged on spectral fidelity, geospatial fidelity, and downstream task performance |

### The worked example

A village during a flood. Sentinel-2 at 10 m detects *that* the region is
flooded. Disaster response needs to know *which roads* are impassable, *which
buildings* are surrounded, and *where the boundary actually lies*.

A conventional upscaler produces a beautiful picture. What is actually needed is
a product that carries its own reliability with it:

```
Sentinel-2 10 m
      ↓  pre-processing
      ↓  AI super-resolution
   < 4 m reconstruction
      ↓  geospatial consistency check
      ↓  spectral consistency check
      ↓  uncertainty estimation
      ↓
  ┌───────────┬──────────────┬─────────────────┐
  │ SR image  │ Confidence   │ Classification  │
  │           │ map          │ / change map    │
  └───────────┴──────────────┴─────────────────┘
```

---

## 2. The five components

### ① Super-resolution
Baseline CNN first, then Transformer, then (only if there is time and reason)
diffusion. Starting with the most complicated model is the classic way to end up
with something that cannot be trained, validated or demonstrated. The model
registry (`src/models/__init__.py`) exists so the architecture can be swapped
without touching the pipeline.

### ② Spectral consistency
Sentinel-2 is not RGB. The network must not produce visually attractive pixels
whose band ratios have drifted — NDVI computed on the SR product has to mean the
same thing it meant on the input. Enforced by a SAM term in the loss and measured
by SAM / ERGAS / per-band RMSE.

### ③ Geospatial consistency
A road at a given coordinate must stay at that coordinate. CRS, affine transform,
bounds and pixel alignment are preserved end to end; output is a GeoTIFF that
opens correctly in QGIS, never a PNG.

### ④ Uncertainty
The most important differentiator. The system must not merely assert "there is a
building here". It must attach a spread, and label the map honestly as an
*uncalibrated relative indicator of model disagreement* — not a probability.

### ⑤ Validation on a real task
Does super-resolution actually improve a downstream analysis? Land-cover
classification accuracy on the 10 m input versus on the SR product is far
stronger evidence than any side-by-side image.

---

## 3. Validation strategy

Three levels, weakest to strongest:

1. **Image quality** — PSNR, SSIM, RMSE. Necessary, not sufficient. High PSNR is
   compatible with confidently invented detail.
2. **Spectral fidelity** — SAM, ERGAS, per-band RMSE. Catches the failure mode
   that PSNR misses.
3. **Downstream task performance** — classification accuracy, mIoU, kappa. The
   only evidence that answers "is this *useful*?"

The reference problem is real: there is usually no true 2.5 m image to compare
against. Three protocols, and every report states which one produced its numbers:

| Protocol | Requires | Gives |
| --- | --- | --- |
| Reduced-resolution (Wald) | Nothing beyond the input | Quantitative metrics against a real observation |
| Full-resolution | A co-registered high-res reference | The strongest evidence available |
| Reference-free | Nothing | Consistency checks only — **no quality metrics** |

---

## 4. Scientific constraints

These are binding on code, documentation, the dashboard and the pitch.

**Never claim:**
- ❌ "The AI creates real 4 m satellite information."
- ❌ "Every generated building is real."
- ❌ "The output is equivalent to native 4 m satellite imagery."
- ❌ "The uncertainty map is ground truth probability."

**Say instead:**
- ✅ "AI-generated super-resolved imagery containing reconstructed fine-scale information."

**Always:**
- Distinguish observed from inferred information.
- Never fabricate results. Where an experiment needs data we do not have, build
  the pipeline and mark the experiment `NOT MEASURED`.
- Never present the confidence map as a calibrated probability, because it is not.
- Carry the warning: *"Super-resolved imagery contains AI-inferred information
  and should not be interpreted as direct high-resolution observation without
  validation."*

Scientific honesty is not a constraint we tolerate — it is the differentiator.
Any team can show a sharper picture. Showing a sharper picture *and* stating
precisely how much of it to trust is the harder and more valuable result.

---

## 5. Roadmap

| Phase | Focus | Status |
| --- | --- | --- |
| 1 | Data loading, preprocessing, patch extraction | ✅ done |
| 2 | Bicubic baseline + PSNR/SSIM/RMSE | ✅ done |
| 3 | First trained SR model vs. baseline | ✅ done (EDSR-Lite) |
| 4 | Spectral loss, geospatial preservation, uncertainty | ✅ done |
| 5 | Downstream application (land cover) | ✅ pipeline done |
| 6 | Dashboard | ✅ done |
| 7 | Presentation | pending |

**The roadmap is not the bottleneck.** Every phase above runs, with 319 passing
tests. The gap is data: v1 trains and evaluates on *synthetic* scenes, which
demonstrate that the plumbing is correct and prove nothing scientific.

### The actual critical path

1. **Real Sentinel-2 L2A data** — B04/B03/B02/B08 stacked at 10 m. Until this
   exists, every metric in the README measures the pipeline, not the science.
2. **A co-registered high-resolution reference** — unlocks full-resolution
   evaluation, the strongest validation tier.
3. **Labels for the downstream task** — turns the land-cover experiment from
   unsupervised k-means into a quantitatively valid accuracy comparison.
4. **GPU access** — see `docs/gpu-runbook.md`. Necessary for a model larger than
   EDSR-Lite, but *not* a substitute for items 1–3.

Item 4 is the tempting one and the least important. A B200 will train the wrong
model on synthetic data very quickly.

---

## 6. The demonstration

Not "before → after". Four panels:

```
┌──────────────────┬──────────────────┐
│ Sentinel-2 10 m  │ AI SR  < 4 m     │
├──────────────────┼──────────────────┤
│ Reference HR     │ Uncertainty map  │
└──────────────────┴──────────────────┘
```

Then the part that actually wins the argument — the downstream comparison:

```
Land-cover classification
  10 m input : XX.X %
  SR product : YY.Y %
```

Those numbers come from `scripts/evaluate.py --downstream`. They are reported
from measurement, never estimated, and never quoted without stating whether the
underlying data was real or synthetic.

---

## 7. One-sentence USP

> We do not claim to recover information the satellite never observed. We
> reconstruct plausible fine-scale information, quantify its uncertainty,
> preserve spectral and geospatial consistency, and measure whether it actually
> improves downstream remote-sensing analysis.
