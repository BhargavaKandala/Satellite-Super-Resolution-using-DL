# Presentation script — SIH 2026, PS-142

Slide-by-slide speaking script, then the Q&A preparation. Live demo commands are
in [`DEMO.md`](DEMO.md).

**Timing:** 12 slides. ~6 minutes of speaking + 2 minutes of live demo, leaving
room for questions. If you are cut to 5 minutes, drop slides 4, 8 and 11.

**The one thing to get right:** every other team will show a sharper picture.
Your differentiator is that you *measured whether the sharper picture was
actually useful*, and that you say out loud which parts of it are invented. Lead
with that, close with that.

---

## Slide 1 — Title

> **Deep Learning Based Super Resolution Mapping from Medium Resolution Satellite Imageries**
> SIH 2026 · Problem Statement 142

**Say:**

> "Sentinel-2 gives us free satellite imagery of the whole planet every five
> days. But its sharpest bands are 10 metres per pixel. We're going to show you
> how to get below 4 metres — and, more importantly, how to know how much of
> that result you can trust."

---

## Slide 2 — The problem

**Visual:** one 10 m pixel drawn as a large square, containing a building, a
tree and a road.

**Say:**

> "This is one Sentinel-2 pixel. Ten metres by ten metres of real ground. Inside
> it there might be a building, a tree, and a road — and the satellite records
> all three as a single averaged number.
>
> For disaster response, urban planning, agriculture, that's not enough
> resolution. Which roads are flooded? Where exactly is the field boundary?
> Commercial sub-metre imagery answers those, but it's expensive and infrequent."

---

## Slide 3 — Why this is not an image-sharpening problem

**Visual:** two panels — "Looks right" (a crisp, confident, wrong road) vs
"Is right".

**Say:**

> "Here's the trap. If you run any off-the-shelf super-resolution model on this,
> you get a beautiful, sharp image. And it may be geographically wrong.
>
> The AI invents a road edge that isn't there. It looks completely convincing.
> In a flood map, that's not a cosmetic flaw — that's a wrong answer delivered
> confidently to someone making an evacuation decision.
>
> So our project isn't 'make satellite images sharper'. It's **information
> reconstruction under uncertainty**: what fine-scale detail can we defensibly
> recover, and how much of it should anyone trust?"

*This is the slide that separates you from the field. Do not rush it.*

---

## Slide 4 — Four specific failure modes

**Visual:** four short rows.

| Failure | Why it matters | What we do |
|---|---|---|
| Radiometry drift | Reflectance is a *measurement*, not a colour | No batch normalisation anywhere |
| Band ratios break | NDVI/NDWI depend on ratios, not levels | Spectral loss term + SAM/ERGAS reported |
| Wrong georeferencing | Output silently 4× too large on the ground | Transform derived, not copied; tested |
| Unlabelled invention | Reconstruction presented as observation | Uncertainty map + explicit labelling |

**Say:**

> "Photo super-resolution models fail on satellite data in four specific ways.
> The one people miss is the second: a model can lower its pixel error by
> trading error between bands. PSNR goes up, and NDVI is destroyed. That's why
> we put a spectral term in the loss and report spectral metrics alongside
> image-quality metrics."

---

## Slide 5 — System architecture

**Visual:** the Mermaid diagram from `README.md` §3.

```
Sentinel-2 GeoTIFF (10 m, B04 B03 B02 B08)
        │
        ├─► Preprocessing ─► Patch extraction ─► Training
        │                                            │
        └─► Tiled inference ◄────── checkpoint ──────┘
                    │
        ┌───────────┼──────────────┐
        ▼           ▼              ▼
   SR GeoTIFF   Uncertainty    Validation
    (2.5 m)        map      (quality/spectral/
                              geospatial)
                                   │
                                   ▼
                        Downstream land-cover test
```

**Say:**

> "Input is a multi-band GeoTIFF — red, green, blue and near-infrared at 10
> metres. Output is a 2.5-metre GeoTIFF that opens correctly in QGIS, plus an
> uncertainty map, plus a metrics report. Every stage is driven by one config
> file, so any run is reproducible from config, seed and checkpoint alone."

---

## Slide 6 — The model

**Visual:** the block diagram below.

```
LR input (4 bands)
   │
   ├──────────── bicubic upsample ──────────────┐  (low frequencies, untouched)
   │                                            │
   ▼                                            ▼
 Conv3×3 ─► 12 × ResidualBlock ─► Conv3×3 ─► PixelShuffle ×4 ─► Conv3×3 ─►(+)─► SR output
                (no BatchNorm)                                                   (2.5 m)
```

**EDSR-Lite** — a residual CNN in the EDSR family.

| | |
|---|---|
| Parameters | **1,223,300** |
| Residual blocks | 12, 64 features, `res_scale` 0.1 |
| Upsampling | PixelShuffle (sub-pixel), ×4 |
| Normalisation | **None** — deliberately |
| Receptive field | 55 px in LR space |
| Input / output | 4 bands → 4 bands |

**Say:**

> "We use EDSR-Lite, a 1.2-million-parameter residual CNN. Three choices are
> specific to satellite data rather than copied from photo super-resolution.
>
> **One — no batch normalisation.** Batch norm rescales activations using batch
> statistics, which destroys the absolute radiometry that makes reflectance
> physically meaningful.
>
> **Two — the network predicts only the residual on top of a bicubic upsample,
> and its output layer starts at zero.** So at step zero the model is *exactly*
> the bicubic baseline, and training can only improve on it. The low-frequency
> content that carries the spectral signature passes through untouched by
> construction.
>
> **Three — all convolutions run at low resolution and we upsample once at the
> end** using PixelShuffle. That's about 16× less compute than upsampling first,
> and it avoids the checkerboard artefacts of transposed convolutions — which in
> a satellite product would be indistinguishable from real fine structure."

**If asked why not a Transformer or diffusion model:**

> "We built a strong baseline first, deliberately. A model registry sits behind
> this — SwinIR or a diffusion model registers with one decorator and becomes
> selectable from the config, with no pipeline changes. But a model you can't
> train, validate and demonstrate isn't worth anything on a hackathon timeline."

---

## Slide 7 — How we train it

**Visual:** the loss equation.

```
Total = 1.00 · L1(pixel)
      + 0.15 · (1 − SSIM)      structural
      + 0.30 · (1 − cos θ)     spectral (SAM)
      + 0.05 · |∇pred − ∇ref|  gradient
```

**Say:**

> "We can't buy paired 10-metre and 2.5-metre imagery, so we use **Wald's
> protocol**: take a real observation, degrade it by the scale factor with a
> sensor-like blur, and train the model to invert that. The real observation is
> the ground truth.
>
> The loss has four terms. Pixel accuracy dominates, but the spectral term is
> weighted second-highest at 0.30 — that's what stops the model from trading
> error between bands to flatter its PSNR."

---

## Slide 8 — The trust layer

**Visual:** SR image beside its uncertainty map (green/yellow/red).

**Say:**

> "This is the part we think matters most. The system doesn't just say 'there's
> a building here.' It attaches a per-pixel spread showing where the model
> disagrees with itself.
>
> We generate this three ways — Monte-Carlo dropout, geometric self-ensembling
> over the eight symmetries of the square, and a reprojection consistency
> residual that checks whether our output degrades back to the input we started
> from.
>
> And we say clearly: **this is an uncalibrated relative indicator, not a
> probability.** Calling it a probability would require calibration we haven't
> done. The code, the file metadata and the UI all state that."

*Judges reward this. Do not soften it.*

---

## Slide 9 — Results

**Visual:** the two tables.

| | PSNR ↑ | SSIM ↑ | RMSE ↓ | SAM° ↓ | ERGAS ↓ |
|---|---|---|---|---|---|
| Bicubic baseline | 34.97 | 0.845 | 0.0178 | 2.791 | 2.772 |
| **AI super-resolution** | **36.40** | **0.871** | **0.0151** | **2.622** | **2.294** |

*Re-run `evaluate.py` the night before and use **your** numbers — a retrain moves
PSNR by ±0.3 dB. Never quote a figure from a checkpoint you no longer have.*

**Say:**

> "Against the bicubic control, we improve on all five metrics — including both
> spectral metrics, which means we got sharper *without* distorting the physics."

---

## Slide 10 — The slide that actually matters

**Visual:** big, single table.

```
Land-cover classification accuracy
  10 m input  : 97.1 %
  SR product  : 98.8 %

  mIoU        : 0.895  →  0.953
```

**Say:**

> "Anyone can show you a sharper image. Here's what we think is the real
> question: **did it help?**
>
> We ran land-cover classification on the original and on our super-resolved
> product, with identical settings. Accuracy improved, and IoU improved more.
> That's evidence the reconstructed detail carries real information — not just
> that it looks nicer.
>
> And I want to be straight with you about these numbers: they come from
> synthetic scenes. They prove our pipeline is correct end to end. They are
> **not** a measurement on real Sentinel-2 imagery. Real-data ingestion is the
> same code path and it's our immediate next step."

*Say that last paragraph. See the Q&A section for why.*

---

## Slide 11 — Engineering

**Say:**

> "Some things we had to get right that aren't visible in a demo.
>
> A full Sentinel-2 tile super-resolved 4× is about 30 terabytes as float32 — it
> can never be held in memory. So we use context-padded, non-overlapping tiled
> writes: each output block is written once, from a prediction that saw real
> context on every side. Peak memory is one tile, independent of scene size.
>
> Georeferencing: the affine transform is *derived* — pixel size divided by four,
> origin fixed — never copied. Copying it is the most common bug in SR pipelines
> and produces a file that opens fine and is silently four times too large on the
> ground. Six automated checks verify this on every run.
>
> 337 tests, including an end-to-end smoke test."

---

## Slide 12 — Close

**Say:**

> "We don't claim to recover information the satellite never observed. We
> reconstruct plausible fine-scale information, quantify its uncertainty,
> preserve spectral and geospatial consistency, and measure whether it actually
> improves downstream analysis.
>
> That last part is what makes this a remote-sensing product rather than a photo
> filter."

Then go to the live demo — [`DEMO.md`](DEMO.md).

---

# Anticipated questions

## The four they will definitely ask

**Q: "Is the AI making up detail?"**

> "Yes — and that's inherent to super-resolution, not a flaw in our
> implementation. The information isn't in the 10-metre observation, so anything
> finer is reconstruction. That's precisely why we built the uncertainty map. We
> call it reconstructed information, never observed information, and the
> dashboard labels every region as Observed, Reconstructed, or Uncertain."

**Q: "Is your confidence map a real probability?"**

> "No, and we're careful not to claim it is. It's a relative indicator of model
> disagreement. A calibrated probability would require a calibration study
> against ground truth that we haven't done. We state that in the code, in the
> GeoTIFF metadata, and in the UI."

**Q: "How can you evaluate without high-resolution reference imagery?"**

> "Wald's protocol. We degrade the real observation by the scale factor,
> super-resolve it back, and score against the original — which is a genuine
> measurement, not a synthetic target. We support three protocols and every
> report we generate states which one produced its numbers, so nobody can
> accidentally compare across protocols."

**Q: "Is this real Sentinel-2 data?"** ← *the one that decides your outcome*

> "Not yet. The numbers on that slide come from synthetic scenes, and they
> validate the pipeline, not the science. Real Sentinel-2 L2A goes through the
> exact same code path — it's our immediate next step, and it's the single
> highest-value thing we can do."

**Answer this honestly.** It is far stronger than bluffing, and infinitely
stronger than being caught. Judges have seen a hundred teams quote numbers they
can't defend. A team that knows exactly what its numbers do and don't prove reads
as the one that actually understands remote sensing. Rehearse this answer until
it comes out calm rather than apologetic.

## Technical follow-ups

**Q: "Why a CNN and not a Transformer / GAN / diffusion model?"**

> "Deliberate sequencing. A strong, honest baseline first — one we can train,
> validate and explain. The architecture sits behind a registry, so SwinIR
> registers with a single decorator and becomes selectable from the config with
> no other changes. We'd rather show a model we fully understand than a bigger
> one we can't defend."

**Q: "How do you know it isn't just hallucinating plausible texture?"**

> "We can't rule it out, and we say so. Our reprojection check catches
> inconsistent output — but we've documented its limitation: invented texture
> with zero local mean passes it. That's an actual test in our suite that asserts
> the failure. The honest defence is the downstream experiment: if the detail were
> pure noise, classification accuracy would not improve."

*That answer wins respect. Volunteering a limitation you've tested for is the
strongest possible signal of rigour.*

**Q: "What's your scale factor and why 2.5 m?"**

> "4×, so 10 m becomes 2.5 m — comfortably below the 4-metre requirement. The
> factor is validated at config load: the model's upsampling factor must equal
> the factor the training data was degraded by, or it refuses to run."

**Q: "Which bands?"**

> "The four Sentinel-2 10-metre bands: red, green, blue and NIR. The channel
> count derives from the config band list, so adding B05 or B11 is a config
> change, not a code change."

**Q: "Would this run on a full Sentinel-2 tile?"**

> "Yes — `--stream` mode writes tiles straight to disk. Memory stays at one tile
> regardless of scene size. Note that uncertainty estimation currently needs the
> scene in memory; a windowed uncertainty pass is on our list rather than
> half-built."

**Q: "How long does training take?"**

> "About 17 seconds per epoch on a laptop CPU for our demo dataset. We have a
> profile for our college's NVIDIA DGX B200. Honestly though — this model is 1.2
> million parameters and a B200 has 180 GB of memory. The GPU matters for a
> *bigger* model on *real* data, not for this one."

**Q: "What's your accuracy?"** (vague — clarify before answering)

> "Which one? Image reconstruction is 36.4 dB PSNR. Spectral angle is 2.62
> degrees. Downstream land-cover accuracy is 98.8% versus 97.1% for the baseline.
> We report all three because reconstruction quality alone doesn't tell you
> whether the product is useful."

## Uncomfortable questions

**Q: "Couldn't you just use an existing pretrained super-resolution model?"**

> "You could produce an image that way. It would use batch normalisation, be
> trained on RGB photographs, output three channels not four, discard the
> georeferencing, and give you no way to know which details were invented. The
> super-resolution model is maybe 20% of this project. The other 80% is what
> makes the output trustworthy enough to use."

**Q: "What's genuinely novel here?"**

> "We're not claiming a novel architecture — we'd rather be honest than
> impressive. What we've built is the reliability layer: spectral validation in
> the loss and in the metrics, exact geospatial preservation with automated
> tests, uncertainty quantification that's labelled honestly, and a downstream
> experiment that measures whether super-resolution actually helps. Most SR work
> stops at PSNR. Ours doesn't."

**Q: "What doesn't work / what would you do with more time?"**

> "Three things, in order. One: real Sentinel-2 data — everything we've shown is
> pipeline validation, not science. Two: a co-registered high-resolution
> reference, which unlocks the strongest validation tier. Three: calibrating the
> uncertainty map so we could legitimately call it a probability. We'd also swap
> in SwinIR, but that's fourth — it's the fun one, not the important one."

*Having a ranked, specific list of your own weaknesses is a strong signal. Vague
answers here read as not knowing.*

---

# Delivery notes

**Do say:** "AI-generated super-resolved imagery containing reconstructed
fine-scale information."

**Never say:**
- ❌ "The AI creates real 4-metre satellite information."
- ❌ "Every generated building is real."
- ❌ "Equivalent to native 4-metre imagery."
- ❌ "The confidence map shows probability."

**If the demo breaks:** you have pre-generated outputs in `outputs/`. Open the
GeoTIFF in QGIS instead and show the georeferencing. Never debug live — say
"I have the output from an earlier run" and move on.

**If you're running short:** cut slides 4, 8 and 11. Never cut slide 3 (why
sharpening isn't enough) or slide 10 (downstream results). Those two carry the
argument.
