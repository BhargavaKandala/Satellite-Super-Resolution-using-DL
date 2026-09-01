#!/usr/bin/env python
"""Import a real Sentinel-2 L2A product into ``data/raw`` as a stacked GeoTIFF.

    python scripts/import_sentinel2.py --input S2A_MSIL2A_....SAFE
    python scripts/import_sentinel2.py --input ./downloaded_bands --size 2048
    python scripts/import_sentinel2.py --input S2A_....SAFE --list

Two things this exists to get right, because both fail *silently* if done by
hand:

**Band order.** The pipeline reads bands positionally — ``data.band_indices``
maps position to meaning. Sentinel-2 ships one JP2 per band with names that sort
B02, B03, B04, B08 (blue, green, red, NIR), while ``config.yaml`` asks for
B04, B03, B02, B08 (red, green, blue, NIR). Stacking in filename order swaps red
and blue: NDVI still computes, the image still looks plausible, and every
spectral metric is quietly wrong.

**The L2A radiometric offset.** From processing baseline 04.00 (products taken
after 2022-01-25) L2A carries ``BOA_ADD_OFFSET = -1000``, so reflectance is
``(DN - 1000) / 10000``, not ``DN / 10000``. Ignore it and every reflectance is
0.1 too high — clipping at 1.0 hides the overflow, and the error is invisible in
a preview image. This script reads the offset from ``MTD_MSIL2A.xml`` and bakes
it into the output, so everything downstream keeps the simple ``DN / 10000``
convention and ``data/raw`` holds one harmonised representation.

The result is written at **10 m**, which is Sentinel-2's native resolution and
therefore a real observation. Training then learns 40 m -> 10 m under Wald's
protocol, against genuine ground truth rather than a synthetic target.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

import _bootstrap  # noqa: F401
from src.config import load_config

#: Sentinel-2 bands available at 10 m ground sampling distance.
TEN_METRE_BANDS = ("B02", "B03", "B04", "B08")

QUANTIFICATION = 10000.0
#: Baseline 04.00 introduced this; earlier products have no offset.
LEGACY_BASELINE_OFFSET = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, help=".SAFE directory, or a folder of band files"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None, help="output GeoTIFF (default: data/raw/<name>.tif)")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="centre-crop to SIZE x SIZE pixels (a full tile is 10980 square)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="override BOA_ADD_OFFSET instead of reading it from the metadata",
    )
    parser.add_argument("--list", action="store_true", help="report what was found, write nothing")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_band_files(root: Path, bands: list[str]) -> dict[str, Path]:
    """Locate one raster per requested band under ``root``.

    Handles the SAFE layout (``GRANULE/*/IMG_DATA/R10m/*_B04_10m.jp2``), the L1C
    layout without the resolution subfolder, and a plain directory of files
    downloaded individually. Band identity comes from the filename, never from
    sort order.
    """
    if root.is_file():
        raise NotADirectoryError(
            f"{root} is a file; pass the .SAFE directory or the folder of bands"
        )

    candidates = [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in (".jp2", ".tif", ".tiff") and p.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no .jp2/.tif files found under {root}")

    found: dict[str, Path] = {}
    for band in bands:
        # Word-boundary match so B08 never matches B8A, and B02 never matches
        # a filename that merely contains "B02" as part of a longer token.
        pattern = re.compile(rf"(?<![0-9A-Za-z]){re.escape(band)}(?![0-9A-Za-z])")
        matches = [p for p in candidates if pattern.search(p.name)]
        if not matches:
            raise FileNotFoundError(
                f"band {band} not found under {root}. "
                f"Looked at {len(candidates)} file(s); is this an L2A product "
                "containing the 10 m bands?"
            )
        # Prefer an explicit 10 m product when several resolutions are present.
        ten_metre = [p for p in matches if "10m" in p.name or "R10m" in str(p.parent)]
        found[band] = sorted(ten_metre or matches)[0]
    return found


def read_boa_offset(root: Path, bands: list[str]) -> tuple[float, str]:
    """Read ``BOA_ADD_OFFSET`` from the product metadata.

    Returns ``(offset, provenance)``. Absent metadata means a pre-04.00 product,
    which genuinely has no offset — so 0.0 is the correct answer, not a guess.
    """
    metadata = next(iter(sorted(root.rglob("MTD_MSIL2A.xml"))), None)
    if metadata is None:
        return LEGACY_BASELINE_OFFSET, "no MTD_MSIL2A.xml found; assuming baseline < 04.00"

    try:
        tree = ET.parse(metadata)
    except ET.ParseError as exc:
        return LEGACY_BASELINE_OFFSET, f"{metadata.name} could not be parsed ({exc})"

    offsets = {
        int(node.attrib["band_id"]): float(node.text)
        for node in tree.iter("BOA_ADD_OFFSET")
        if node.text is not None and "band_id" in node.attrib
    }
    if not offsets:
        return LEGACY_BASELINE_OFFSET, f"{metadata.name} declares no BOA_ADD_OFFSET"

    unique = set(offsets.values())
    if len(unique) > 1:
        raise ValueError(
            f"per-band BOA_ADD_OFFSET values differ ({sorted(unique)}); this "
            "importer bakes a single offset into all bands. Pass --offset to "
            "override, or import bands separately."
        )
    return unique.pop(), f"{metadata.name} (baseline >= 04.00)"


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------
def _centre_window(width: int, height: int, size: int) -> Window:
    if size >= min(width, height):
        raise ValueError(
            f"--size {size} is not smaller than the scene ({width}x{height}); omit it"
        )
    return Window((width - size) // 2, (height - size) // 2, size, size)


def stack(
    band_files: dict[str, Path], bands: list[str], offset: float, size: int | None
) -> tuple[np.ndarray, dict]:
    """Read each band in *config* order and harmonise the digital numbers."""
    profile: dict | None = None
    window: Window | None = None
    planes: list[np.ndarray] = []

    for band in bands:
        with rasterio.open(band_files[band]) as src:
            if profile is None:
                profile = src.profile.copy()
                if size is not None:
                    window = _centre_window(src.width, src.height, size)
                    profile.update(
                        width=size,
                        height=size,
                        transform=src.window_transform(window),
                    )
            elif (src.width, src.height) != (profile["width"], profile["height"]) and window is None:
                raise ValueError(
                    f"band {band} is {src.width}x{src.height}, but the first band "
                    f"is {profile['width']}x{profile['height']}. All bands must "
                    "share the 10 m grid — check you selected the R10m folder."
                )
            planes.append(src.read(1, window=window).astype(np.int32))

    assert profile is not None
    stacked = np.stack(planes)

    if offset:
        # Fold the offset into the stored DNs so the rest of the pipeline keeps
        # the plain DN/10000 convention. Clamp at 1 because 0 is the nodata
        # value: a genuinely dark pixel must not become "no observation".
        stacked = stacked + int(offset)
    stacked = np.clip(stacked, 1, 65535).astype(np.uint16)

    profile.update(
        driver="GTiff",
        count=len(bands),
        dtype="uint16",
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=0,
    )
    return stacked, profile


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    bands = list(cfg.data.bands)

    root = Path(args.input)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 1

    unsupported = [b for b in bands if b not in TEN_METRE_BANDS]
    if unsupported:
        print(
            f"error: {unsupported} are not Sentinel-2 10 m bands. This importer "
            f"handles {', '.join(TEN_METRE_BANDS)}; other bands need resampling "
            "onto the 10 m grid first.",
            file=sys.stderr,
        )
        return 2

    print("=" * 70)
    print("  IMPORT SENTINEL-2 L2A")
    print("=" * 70)

    try:
        band_files = find_band_files(root, bands)
        offset, provenance = (
            (args.offset, "--offset override")
            if args.offset is not None
            else read_boa_offset(root, bands)
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nband order (from config, NOT filename order):")
    for position, band in enumerate(bands, start=1):
        print(f"  {position}. {band:5s} <- {band_files[band].name}")

    print(f"\nradiometric offset: {offset:+g} DN")
    print(f"  source: {provenance}")
    if offset:
        print(
            f"  reflectance = (DN {offset:+g}) / {QUANTIFICATION:.0f}; "
            "baked into the output so downstream stays DN/10000"
        )

    if args.list:
        print("\n--list given; nothing written")
        return 0

    try:
        data, profile = stack(band_files, bands, offset, args.size)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        out = Path(args.output)
    else:
        stem = root.name.replace(".SAFE", "") or "sentinel2"
        out = cfg.get_path("data.raw_dir") / f"{stem}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = tuple(bands)
        dst.update_tags(
            SOURCE=str(root),
            BOA_ADD_OFFSET=str(offset),
            OFFSET_APPLIED="true" if offset else "false",
            QUANTIFICATION_VALUE=str(QUANTIFICATION),
            BAND_ORDER=",".join(bands),
            NOTE="Real Sentinel-2 L2A observation at 10 m. Harmonised to DN/10000.",
        )

    valid = float((data > 0).mean())
    print(f"\nwrote {out}")
    print(f"  size:   {profile['width']} x {profile['height']}, {profile['count']} bands")
    print(f"  pixel:  {abs(profile['transform'].a):g} m")
    print(f"  CRS:    {profile['crs']}")
    print(f"  valid:  {valid:.1%} of pixels")
    print(f"  refl:   min {data.min() / QUANTIFICATION:.4f}  "
          f"mean {data.mean() / QUANTIFICATION:.4f}  "
          f"max {data.max() / QUANTIFICATION:.4f}")

    if valid < 0.5:
        print(
            "\nwarning: over half the scene is nodata. Crop to a populated area "
            "with --size, or pick a different granule."
        )

    print("\nnext:")
    print("  python scripts/prepare_dataset.py     # patches from real data")
    print("  python scripts/train.py")
    print("  python scripts/evaluate.py --downstream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
