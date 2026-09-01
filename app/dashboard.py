#!/usr/bin/env python
"""Phase 10 + 11: the evaluation and explainability dashboard.

    streamlit run app/dashboard.py

Design intent
-------------
The dashboard's job is not to make the output look impressive — it is to let a
reviewer decide how much of what they are seeing was *observed* and how much was
*inferred*. Every view is built around that distinction, and the disclaimer is
rendered before any imagery, not tucked into a footer.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, set_seed  # noqa: E402
from src.data.geotiff import read_info, validate_geospatial, write_superres  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    DegradationConfig,
    bicubic_upsample,
    degrade,
    to_rgb,
)
from src.evaluation.metrics import compute_metrics, sam_map  # noqa: E402
from src.inference.predict import (  # noqa: E402
    describe_device,
    load_checkpoint,
    read_scene,
    resolve_device,
    write_uncertainty,
)
from src.models import build_model  # noqa: E402

DISCLAIMER = (
    "Super-resolved imagery contains AI-inferred information and should not be "
    "interpreted as direct high-resolution observation without validation."
)

st.set_page_config(
    page_title="SIH 142 — Satellite Super-Resolution",
    page_icon="🛰️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_config(path: str | None = None):
    return load_config(path)


@st.cache_resource(show_spinner="Loading model…")
def get_model(checkpoint: str, device_str: str):
    """Load a checkpoint once per session — reloading per rerun would dominate runtime."""
    device = resolve_device(device_str)
    model, payload = load_checkpoint(checkpoint, device)
    return model, payload, device


@st.cache_resource(show_spinner=False)
def get_baseline_model(_cfg, device_str: str):
    return build_model(_cfg.merge({"model": {"name": "bicubic"}}))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_rgb(array: np.ndarray, cfg, max_dim: int = 1024) -> np.ndarray:
    """Band-select, contrast-stretch and downsample for browser display.

    The stretch is cosmetic and never touches the arrays used for metrics.
    """
    rgb = to_rgb(
        array,
        list(cfg.data.bands),
        list(cfg.data.rgb_bands),
        stretch=True,
        percentiles=tuple(cfg.dashboard.stretch_percentiles),
    )
    return _downsample(rgb, max_dim)


def _downsample(image: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = image.shape[:2]
    factor = max(1, int(np.ceil(max(height, width) / max_dim)))
    return image[::factor, ::factor] if factor > 1 else image


def colourise(values: np.ndarray, max_dim: int = 1024, cmap: str = "magma") -> np.ndarray:
    """Apply a perceptually uniform colour map to a scalar field."""
    import matplotlib

    data = _downsample(np.asarray(values, dtype=np.float32), max_dim)
    finite = data[np.isfinite(data)]
    if finite.size:
        lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
        data = np.clip((data - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(data)
    else:
        data = np.zeros_like(data)
    return (matplotlib.colormaps[cmap](data)[..., :3] * 255).astype(np.uint8)


def crop(array: np.ndarray, roi: tuple[int, int, int] | None, scale: int = 1) -> np.ndarray:
    """Extract the selected region of interest, scaled into the array's own grid."""
    if roi is None:
        return array
    row, col, size = (v * scale for v in roi)
    return array[..., row : row + size, col : col + size]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(cfg):
    st.sidebar.title("🛰️ SIH 2026 — PS 142")
    st.sidebar.caption("Deep-learning super-resolution mapping from medium-resolution imagery")

    st.sidebar.subheader("Model")
    checkpoint_dir = cfg.get_path("training.checkpoint_dir")
    checkpoints = sorted(checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []
    options = ["bicubic baseline only"] + [str(p) for p in checkpoints]
    default = next((i for i, o in enumerate(options) if o.endswith("best.pth")), 0)
    checkpoint = st.sidebar.selectbox("Checkpoint", options, index=default)

    device_choice = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    device_str = None if device_choice == "auto" else device_choice
    st.sidebar.caption(f"Resolved: `{describe_device(resolve_device(device_str))}`")

    st.sidebar.subheader("Uncertainty")
    methods = ["mc_dropout", "ensemble", "reprojection", "none"]
    method = st.sidebar.selectbox(
        "Method", methods, index=methods.index(str(cfg.uncertainty.method))
    )
    passes = st.sidebar.slider("Passes", 2, 16, int(cfg.uncertainty.passes))

    st.sidebar.subheader("Evaluation")
    protocol = st.sidebar.radio(
        "Protocol",
        ["reduced_resolution", "reference_free"],
        help=(
            "Reduced-resolution (Wald's protocol) degrades the scene, "
            "super-resolves it back, and scores against the real observation — "
            "the only way to get quantitative metrics without a higher-resolution "
            "reference image."
        ),
    )
    max_dim = st.sidebar.select_slider(
        "Preview resolution", [512, 768, 1024, 1536, 2048], value=int(cfg.dashboard.preview_max_dim)
    )
    return checkpoint, device_str, method, passes, protocol, max_dim


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    scene_path, cfg, checkpoint, device_str, method, passes, protocol, report=None
):
    """Preprocess → super-resolve → estimate uncertainty → score.

    ``report`` is an optional progress callback taking a markdown string. It is
    injected rather than calling Streamlit directly so the whole pipeline stays
    runnable — and testable — outside a Streamlit session.
    """
    from src.uncertainty.uncertainty import estimate

    report = report or (lambda _message: None)
    scale = int(cfg.patches.scale)
    band_indices = list(cfg.data.band_indices)
    timings: dict[str, float] = {}

    report("**Preprocessing** — reading GeoTIFF, normalising reflectance, masking nodata")
    started = time.perf_counter()
    observed, valid_mask, info = read_scene(scene_path, band_indices, float(cfg.data.dn_scale))
    timings["preprocess"] = time.perf_counter() - started

    if protocol == "reduced_resolution":
        _, height, width = observed.shape
        height, width = height - height % scale, width - width % scale
        reference = observed[:, :height, :width]
        valid_mask = valid_mask[:height, :width]
        lr = degrade(reference, scale, DegradationConfig.from_config(cfg.patches.degradation))
        report(
            f"**Wald's protocol** — degraded {width}×{height} to "
            f"{width // scale}×{height // scale}; the original observation is the ground truth"
        )
    else:
        reference = None
        lr = observed
        report("**Reference-free** — no ground truth; metrics will be omitted")

    device = resolve_device(device_str)
    use_model = checkpoint != "bicubic baseline only"
    if use_model:
        model, payload, device = get_model(checkpoint, device_str or "")
    else:
        model = get_baseline_model(cfg, device_str or "")
        payload = {}
        method = "none"

    report(f"**Inference** — {type(model).__name__} on `{describe_device(device)}`")
    started = time.perf_counter()
    run_cfg = cfg.merge({"uncertainty": {"method": method, "passes": passes}})
    tiling = dict(
        tile_size=int(cfg.inference.tile_size),
        overlap=int(cfg.inference.tile_overlap),
        batch_size=int(cfg.inference.batch_size),
        amp=bool(cfg.training.amp),
        channels_last=bool(cfg.training.channels_last),
    )
    result = estimate(model, lr, scale, run_cfg, device=device, **tiling)
    timings["inference"] = time.perf_counter() - started

    report("**Baseline** — bicubic interpolation for comparison")
    baseline = bicubic_upsample(lr, scale)

    metrics: dict = {}
    if reference is not None:
        report("**Metrics** — PSNR / SSIM / RMSE / SAM / ERGAS")
        started = time.perf_counter()
        kwargs = dict(
            data_range=float(cfg.evaluation.data_range),
            ratio=float(cfg.evaluation.ergas_ratio),
            valid_mask=valid_mask,
            border_crop=int(cfg.evaluation.border_crop),
            band_names=list(cfg.data.bands),
            metrics=list(cfg.evaluation.metrics),
        )
        metrics["bicubic"] = compute_metrics(baseline, reference, **kwargs)
        if use_model:
            metrics["ai_sr"] = compute_metrics(result.prediction, reference, **kwargs)
        timings["metrics"] = time.perf_counter() - started

    return {
        "info": info,
        "lr": lr,
        "sr": result.prediction,
        "baseline": baseline,
        "reference": reference,
        "uncertainty": result,
        "metrics": metrics,
        "timings": timings,
        "scale": scale,
        "checkpoint": payload,
        "used_model": use_model,
        "protocol": protocol,
    }


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def view_imagery(state, cfg, max_dim):
    st.subheader("Imagery")
    scale = state["scale"]
    lr, sr, baseline = state["lr"], state["sr"], state["baseline"]

    with st.expander("Zoom to a region", expanded=False):
        st.caption(
            "Select a window in low-resolution pixel coordinates. The same ground "
            "area is shown from every product, so the comparison stays like-for-like."
        )
        _, height, width = lr.shape
        size = st.slider("Window size (LR px)", 32, min(512, min(height, width)), min(256, min(height, width)), step=32)
        col_a, col_b = st.columns(2)
        row = col_a.slider("Row offset", 0, max(0, height - size), 0, step=8)
        col = col_b.slider("Column offset", 0, max(0, width - size), 0, step=8)
        zoom = st.checkbox("Apply zoom", value=False)
    roi = (row, col, size) if zoom else None

    lr_view = crop(lr, roi, 1)
    sr_view = crop(sr, roi, scale)
    base_view = crop(baseline, roi, scale)

    columns = st.columns(3)
    columns[0].image(
        render_rgb(lr_view, cfg, max_dim),
        caption=f"Observed — {cfg.data.source_resolution_m:g} m Sentinel-2 input",
        use_container_width=True,
    )
    columns[1].image(
        render_rgb(base_view, cfg, max_dim),
        caption=f"Bicubic baseline — {cfg.data.target_resolution_m:g} m (no new information)",
        use_container_width=True,
    )
    columns[2].image(
        render_rgb(sr_view, cfg, max_dim),
        caption=f"AI super-resolved — {cfg.data.target_resolution_m:g} m (contains inferred detail)",
        use_container_width=True,
    )

    if state["reference"] is not None:
        st.markdown("---")
        ref_view = crop(state["reference"], roi, scale)
        columns = st.columns(3)
        columns[0].image(
            render_rgb(ref_view, cfg, max_dim),
            caption="Reference (ground truth for this protocol)",
            use_container_width=True,
        )
        error = np.abs(sr_view - ref_view).max(axis=0)
        columns[1].image(
            colourise(error, max_dim, "inferno"),
            caption="Error map — |AI SR − reference|, max across bands",
            use_container_width=True,
        )
        columns[2].image(
            colourise(sam_map(sr_view, ref_view), max_dim, "viridis"),
            caption="Spectral angle map (degrees) — where band ratios shifted",
            use_container_width=True,
        )


def view_metrics(state):
    st.subheader("Metrics")
    metrics = state["metrics"]
    if not metrics:
        st.warning(
            "No ground truth was available under the reference-free protocol, so "
            "no accuracy metrics can be computed. Switch to the reduced-resolution "
            "protocol in the sidebar for quantitative numbers."
        )
        return

    st.caption(
        "Reduced-resolution protocol: the scene was degraded by the scale factor "
        "and reconstructed, then scored against the original observation."
        if state["protocol"] == "reduced_resolution"
        else "Scored against the supplied reference image."
    )

    if "ai_sr" in metrics:
        base, ai = metrics["bicubic"], metrics["ai_sr"]
        st.markdown("##### AI super-resolution vs bicubic baseline")
        columns = st.columns(5)
        specs = [
            ("PSNR", "psnr", "dB", True),
            ("SSIM", "ssim", "", True),
            ("RMSE", "rmse", "", False),
            ("SAM", "sam", "°", False),
            ("ERGAS", "ergas", "", False),
        ]
        for column, (label, key, unit, higher) in zip(columns, specs):
            if key not in ai:
                continue
            delta = ai[key] - base[key]
            column.metric(
                f"{label}{f' ({unit})' if unit else ''}",
                f"{ai[key]:.4f}",
                f"{delta:+.4f}",
                delta_color="normal" if higher else "inverse",
            )
        st.caption(
            "Deltas are relative to the bicubic baseline. Green means the AI model "
            "improved the metric. PSNR and SSIM: higher is better. RMSE, SAM and "
            "ERGAS: lower is better."
        )

    import pandas as pd

    rows = []
    for method, values in metrics.items():
        row = {"method": method}
        row.update({k: v for k, v in values.items() if not isinstance(v, dict)})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("method"), use_container_width=True)

    st.markdown("##### Spectral consistency — per-band RMSE")
    st.caption(
        "Spectral fidelity is the metric that decides whether the product is "
        "scientifically usable. A model can raise PSNR while distorting band "
        "ratios enough to break every downstream index."
    )
    band_rows = {
        method: values["per_band_rmse"]
        for method, values in metrics.items()
        if "per_band_rmse" in values
    }
    if band_rows:
        st.dataframe(pd.DataFrame(band_rows), use_container_width=True)


def view_uncertainty(state, cfg, max_dim):
    st.subheader("Uncertainty & explainability")
    result = state["uncertainty"]

    st.error(f"**{DISCLAIMER}**", icon="⚠️")

    st.markdown("##### What you are looking at")
    columns = st.columns(3)
    columns[0].info(
        "**Observed**\n\nInformation genuinely recorded by Sentinel-2 at "
        f"{cfg.data.source_resolution_m:g} m. Every low-frequency structure in the "
        "output traces back to a real measurement.",
        icon="🛰️",
    )
    columns[1].warning(
        "**Reconstructed**\n\nFine-scale detail below the sensor's resolving power. "
        "The network produced it from patterns learnt during training. It is "
        "plausible, not measured — individual small features may not exist.",
        icon="🧠",
    )
    columns[2].error(
        "**Uncertain**\n\nAreas where the model's own predictions disagree. Treat "
        "detail here as unreliable. Low uncertainty is not proof of correctness.",
        icon="❓",
    )

    st.markdown("---")
    columns = st.columns(2)
    columns[0].image(
        colourise(result["uncertainty_normalised"], max_dim, "magma"),
        caption=f"Model uncertainty ({result['method']}, {result['passes']} passes) — "
        "bright = less stable reconstruction",
        use_container_width=True,
    )
    columns[1].image(
        colourise(result["reprojection_residual"], max_dim, "cividis"),
        caption="Reprojection residual — disagreement with the observed pixels "
        "after degrading the output back to the input resolution",
        use_container_width=True,
    )

    summary = result["statistics"]
    columns = st.columns(4)
    columns[0].metric("Mean uncertainty", f"{summary['mean']:.5f}")
    columns[1].metric("Median", f"{summary['median']:.5f}")
    columns[2].metric("95th percentile", f"{summary['p95']:.5f}")
    columns[3].metric("Reprojection MAE", f"{summary['reprojection_mae']:.5f}")

    st.info(
        "**How to read these numbers.** They are in reflectance units and measure "
        "*relative* model disagreement. They are **not calibrated probabilities** — "
        "no calibration set exists for detail that was never observed. High values "
        "reliably flag unstable reconstruction. Low values do not guarantee the "
        "detail is real: a model can be confidently wrong. The reprojection "
        "residual is the one signal anchored to real measurements, since it "
        "compares against pixels the sensor actually recorded.",
        icon="📐",
    )
    for note in result["notes"]:
        st.warning(note)


def view_geospatial(state):
    st.subheader("Geospatial information")
    info = state["info"]
    scale = state["scale"]
    summary = info.summary()

    columns = st.columns(2)
    with columns[0]:
        st.markdown("##### Input")
        bounds = summary["bounds"]
        st.json(
            {
                "CRS": summary["crs"],
                "EPSG": summary["epsg"],
                "dimensions": f"{summary['width']} × {summary['height']}",
                "bands": summary["bands"],
                "resolution_m": [summary["resolution_x"], summary["resolution_y"]],
                "bounds": bounds,
                "nodata": summary["nodata"],
            }
        )
    with columns[1]:
        st.markdown("##### Super-resolved output")
        st.json(
            {
                "CRS": summary["crs"],
                "EPSG": summary["epsg"],
                "dimensions": f"{summary['width'] * scale} × {summary['height'] * scale}",
                "bands": summary["bands"],
                "resolution_m": [
                    summary["resolution_x"] / scale,
                    summary["resolution_y"] / scale,
                ],
                "bounds": summary["bounds"],
                "note": "Identical footprint; pixel size divided by the scale factor",
            }
        )

    timings = state["timings"]
    st.markdown("##### Processing time")
    columns = st.columns(len(timings) or 1)
    for column, (stage, seconds) in zip(columns, timings.items()):
        column.metric(stage.replace("_", " ").title(), f"{seconds:.2f} s")


def view_export(state, cfg, scene_path):
    st.subheader("Export")
    st.caption(
        "Outputs are GeoTIFFs carrying the source CRS and a correctly rescaled "
        "affine transform, so they open aligned in QGIS or ArcGIS."
    )

    info = state["info"]
    scale = state["scale"]
    stem = Path(scene_path).stem
    out_dir = REPO_ROOT / "outputs" / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    if st.button("Generate export files", type="primary"):
        with st.spinner("Writing GeoTIFFs…"):
            sr_path = out_dir / f"{stem}_sr.tif"
            payload = np.clip(
                state["sr"] * float(cfg.inference.output_dn_scale), 0, 65535
            ).astype("uint16")
            write_superres(
                sr_path, payload, info, scale, nodata=None, compress=str(cfg.inference.compress)
            )

            unc_path = out_dir / f"{stem}_uncertainty.tif"
            write_uncertainty(
                unc_path,
                state["uncertainty"]["uncertainty"],
                info,
                scale,
                tags={"SR_UNCERTAINTY_METHOD": state["uncertainty"]["method"]},
            )

            validation = validate_geospatial(sr_path, info, scale)
            report = {
                "scene": stem,
                "protocol": state["protocol"],
                "scale": scale,
                "metrics": state["metrics"],
                "uncertainty": state["uncertainty"]["statistics"],
                "geospatial_validation": validation,
                "timings_seconds": state["timings"],
                "disclaimer": DISCLAIMER,
            }
            report_path = out_dir / f"{stem}_metrics.json"
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        st.session_state["exports"] = {
            "sr": sr_path,
            "uncertainty": unc_path,
            "report": report_path,
            "validation": validation,
        }

    exports = st.session_state.get("exports")
    if not exports:
        return

    validation = exports["validation"]
    if validation["valid"]:
        st.success("Geospatial validation passed — CRS, transform and bounds are consistent.")
    else:
        st.error(f"Geospatial validation FAILED: {validation['checks']}")

    columns = st.columns(3)
    for column, (label, key, mime) in zip(
        columns,
        [
            ("⬇️ SR GeoTIFF", "sr", "image/tiff"),
            ("⬇️ Uncertainty GeoTIFF", "uncertainty", "image/tiff"),
            ("⬇️ Metrics JSON", "report", "application/json"),
        ],
    ):
        path = exports[key]
        column.download_button(
            label, path.read_bytes(), file_name=path.name, mime=mime, use_container_width=True
        )

    if state["metrics"]:
        import pandas as pd

        rows = []
        for method, values in state["metrics"].items():
            row = {"method": method}
            row.update({k: v for k, v in values.items() if not isinstance(v, dict)})
            row.update({f"rmse_{b}": v for b, v in values.get("per_band_rmse", {}).items()})
            rows.append(row)
        st.download_button(
            "⬇️ Metrics CSV",
            pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
            file_name=f"{stem}_metrics.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = get_config()
    set_seed(int(cfg.project.seed))

    st.title("Satellite Image Super-Resolution Platform")
    st.caption(
        "SIH 2026 · Problem Statement 142 — Deep-learning-based super-resolution "
        f"mapping: {cfg.data.source_resolution_m:g} m Sentinel-2 → "
        f"{cfg.data.target_resolution_m:g} m"
    )
    st.error(f"**{DISCLAIMER}**", icon="⚠️")

    checkpoint, device_str, method, passes, protocol, max_dim = sidebar(cfg)

    st.markdown("### 1 · Input")
    raw_dir = cfg.get_path("data.raw_dir")
    samples = sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff"))

    tab_upload, tab_sample = st.tabs(["Upload a GeoTIFF", "Use a scene from data/raw"])
    scene_path = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Sentinel-2 GeoTIFF",
            type=["tif", "tiff"],
            help=(
                f"Bands must be ordered as configured: {list(cfg.data.bands)} "
                f"at indices {list(cfg.data.band_indices)}."
            ),
        )
        if uploaded is not None:
            upload_dir = REPO_ROOT / "data" / "processed"
            upload_dir.mkdir(parents=True, exist_ok=True)
            scene_path = upload_dir / uploaded.name
            scene_path.write_bytes(uploaded.getbuffer())
            st.success(f"Uploaded `{uploaded.name}` ({uploaded.size / 1e6:.1f} MB)")

    with tab_sample:
        if samples:
            choice = st.selectbox("Scene", [p.name for p in samples])
            if st.checkbox("Use this scene", value=not bool(scene_path)):
                scene_path = raw_dir / choice
        else:
            st.info(
                f"No scenes in `{raw_dir}`. Generate demo data with:\n\n"
                "```\npython scripts/prepare_dataset.py --synthetic\n```"
            )

    if scene_path is None:
        st.stop()

    try:
        info = read_info(scene_path)
    except Exception as exc:  # rasterio raises a variety of driver errors
        st.error(f"Could not read `{Path(scene_path).name}` as a GeoTIFF: {exc}")
        st.stop()

    columns = st.columns(4)
    columns[0].metric("Dimensions", f"{info.width} × {info.height}")
    columns[1].metric("Bands", info.count)
    columns[2].metric("Pixel size", f"{info.resolution[0]:g} m")
    columns[3].metric("CRS", info.crs.to_string() if info.crs else "none")

    if info.count < max(cfg.data.band_indices):
        st.error(
            f"This file has {info.count} band(s) but the configuration expects at "
            f"least {max(cfg.data.band_indices)}. Adjust `data.band_indices` in "
            "`configs/config.yaml`."
        )
        st.stop()
    if not info.is_georeferenced:
        st.warning("This file has no CRS or transform — the output will not be geospatially usable.")

    st.markdown("### 2 · Process")
    if st.button("Run super-resolution", type="primary", use_container_width=True):
        st.session_state.pop("exports", None)
        status = st.status("Running pipeline…", expanded=True)
        st.session_state["result"] = run_pipeline(
            scene_path,
            cfg,
            checkpoint,
            device_str,
            method,
            passes,
            protocol,
            report=status.write,
        )
        status.update(label="Pipeline complete", state="complete", expanded=False)
        st.session_state["scene_path"] = str(scene_path)

    state = st.session_state.get("result")
    if state is None:
        st.info("Configure the model in the sidebar, then run the pipeline.")
        st.stop()

    if not state["used_model"]:
        st.warning(
            "Running the **bicubic baseline only** — no trained checkpoint was "
            "selected. Train one with `python scripts/train.py` to see the AI "
            "comparison this dashboard is built around."
        )

    st.markdown("### 3 · Results")
    tabs = st.tabs(
        ["🖼️ Imagery", "📊 Metrics", "❓ Uncertainty", "🌍 Geospatial", "💾 Export"]
    )
    with tabs[0]:
        view_imagery(state, cfg, max_dim)
    with tabs[1]:
        view_metrics(state)
    with tabs[2]:
        view_uncertainty(state, cfg, max_dim)
    with tabs[3]:
        view_geospatial(state)
    with tabs[4]:
        view_export(state, cfg, st.session_state["scene_path"])


if __name__ == "__main__":
    main()
