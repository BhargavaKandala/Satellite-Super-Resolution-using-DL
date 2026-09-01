"""Importing a real Sentinel-2 L2A product.

The two failures guarded here are both *silent*: swapped bands and an unapplied
radiometric offset each produce a file that opens fine and looks plausible, and
each corrupts every spectral metric downstream.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import import_sentinel2 as imp
from src.config import load_config
from src.data.geotiff import read_info

BANDS = ["B04", "B03", "B02", "B08"]
#: A distinct constant per band, so a mis-ordered stack is unmistakable. All sit
#: comfortably above the 1000 DN offset, so none of them hits the nodata clamp —
#: that boundary gets its own test.
FILL = {"B02": 1500, "B03": 2000, "B04": 3000, "B08": 4000}


def _write_band(path, value, size=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype="uint16",
        crs="EPSG:32643",
        transform=from_origin(300000.0, 2000000.0, 10.0, 10.0),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((size, size), value, dtype=np.uint16), 1)


@pytest.fixture
def safe_dir(tmp_path):
    """A minimal SAFE tree: 10 m bands plus metadata declaring the offset."""
    root = tmp_path / "S2A_MSIL2A_20230115T051121_N0509_R019_T43PGQ.SAFE"
    img = root / "GRANULE" / "L2A_T43PGQ" / "IMG_DATA" / "R10m"
    for band, value in FILL.items():
        _write_band(img / f"T43PGQ_20230115T051121_{band}_10m.jp2.tif", value)

    (root / "MTD_MSIL2A.xml").write_text(
        """<?xml version="1.0"?>
        <Level-2A_User_Product>
          <General_Info>
            <Product_Image_Characteristics>
              <BOA_ADD_OFFSET_VALUES_LIST>
                <BOA_ADD_OFFSET band_id="0">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="1">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="2">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="3">-1000</BOA_ADD_OFFSET>
              </BOA_ADD_OFFSET_VALUES_LIST>
            </Product_Image_Characteristics>
          </General_Info>
        </Level-2A_User_Product>
        """,
        encoding="utf-8",
    )
    return root


@pytest.fixture
def legacy_dir(tmp_path):
    """A pre-baseline-04.00 product: bands, no offset metadata."""
    root = tmp_path / "old_product"
    for band, value in FILL.items():
        _write_band(root / f"{band}.tif", value)
    return root


# ---------------------------------------------------------------------------
# Band discovery and ordering
# ---------------------------------------------------------------------------
def test_bands_are_found_by_name(safe_dir):
    found = imp.find_band_files(safe_dir, BANDS)
    assert set(found) == set(BANDS)
    for band, path in found.items():
        assert band in path.name


def test_output_follows_config_order_not_filename_order(safe_dir, tmp_path):
    """Filenames sort B02,B03,B04,B08; the config wants B04,B03,B02,B08."""
    out = tmp_path / "stacked.tif"
    assert imp.main(["--input", str(safe_dir), "--output", str(out)]) == 0

    with rasterio.open(out) as src:
        values = [int(src.read(i + 1)[0, 0]) for i in range(src.count)]
        assert list(src.descriptions) == BANDS

    # Offset -1000 was applied, so each band is its fill minus 1000.
    assert values == [FILL[b] - 1000 for b in BANDS]
    assert values[0] > values[2], "band 1 must be red (B04), not blue (B02)"


def test_b08_is_not_confused_with_b8a(tmp_path):
    """Word-boundary matching: a B8A file must never satisfy a B08 request."""
    root = tmp_path / "p"
    for band, value in FILL.items():
        _write_band(root / f"{band}.tif", value)
    _write_band(root / "B8A.tif", 9999)

    found = imp.find_band_files(root, ["B08"])
    assert found["B08"].name == "B08.tif"


def test_a_missing_band_is_reported_by_name(tmp_path):
    root = tmp_path / "p"
    _write_band(root / "B04.tif", 3000)
    with pytest.raises(FileNotFoundError, match="band B03 not found"):
        imp.find_band_files(root, ["B04", "B03"])


def test_passing_a_file_instead_of_a_directory_is_caught(safe_dir):
    band = next(safe_dir.rglob("*B04*"))
    with pytest.raises(NotADirectoryError, match="pass the .SAFE directory"):
        imp.find_band_files(band, BANDS)


# ---------------------------------------------------------------------------
# The radiometric offset
# ---------------------------------------------------------------------------
def test_offset_is_read_from_product_metadata(safe_dir):
    offset, provenance = imp.read_boa_offset(safe_dir, BANDS)
    assert offset == -1000.0
    assert "MTD_MSIL2A.xml" in provenance


def test_absent_metadata_means_no_offset_not_a_guess(legacy_dir):
    """Pre-04.00 products genuinely have no offset; 0.0 is correct, not a default."""
    offset, provenance = imp.read_boa_offset(legacy_dir, BANDS)
    assert offset == 0.0
    assert "baseline < 04.00" in provenance


def test_offset_changes_the_resulting_reflectance(safe_dir, legacy_dir, tmp_path):
    """The whole point: ignoring the offset shifts reflectance by +0.1."""
    modern, legacy = tmp_path / "m.tif", tmp_path / "l.tif"
    assert imp.main(["--input", str(safe_dir), "--output", str(modern)]) == 0
    assert imp.main(["--input", str(legacy_dir), "--output", str(legacy)]) == 0

    with rasterio.open(modern) as a, rasterio.open(legacy) as b:
        shift = (b.read(1)[0, 0] - a.read(1)[0, 0]) / 10000.0
    assert shift == pytest.approx(0.1)


def test_offset_can_be_overridden(safe_dir, tmp_path):
    out = tmp_path / "o.tif"
    assert imp.main(["--input", str(safe_dir), "--output", str(out), "--offset", "0"]) == 0
    with rasterio.open(out) as src:
        assert int(src.read(1)[0, 0]) == FILL["B04"]


def test_dark_pixels_never_become_nodata(tmp_path):
    """DN below the offset must clamp to 1, not to 0 — 0 means 'no observation'."""
    root = tmp_path / "dark"
    for band in FILL:
        _write_band(root / f"{band}.tif", 500)  # below the 1000 offset

    data, profile = imp.stack(imp.find_band_files(root, BANDS), BANDS, -1000.0, None)
    assert data.min() >= 1
    assert profile["nodata"] == 0


# ---------------------------------------------------------------------------
# Output product
# ---------------------------------------------------------------------------
def test_output_is_a_valid_ten_metre_geotiff(safe_dir, tmp_path):
    out = tmp_path / "s2.tif"
    assert imp.main(["--input", str(safe_dir), "--output", str(out)]) == 0

    info = read_info(out)
    assert info.resolution == (10.0, 10.0)
    assert info.crs is not None
    assert info.count == len(BANDS)
    assert info.dtype == "uint16"
    assert info.band_descriptions == tuple(BANDS)


def test_provenance_is_recorded_in_the_tags(safe_dir, tmp_path):
    out = tmp_path / "s2.tif"
    imp.main(["--input", str(safe_dir), "--output", str(out)])
    tags = read_info(out).tags
    assert tags["BOA_ADD_OFFSET"] == "-1000.0"
    assert tags["OFFSET_APPLIED"] == "true"
    assert tags["BAND_ORDER"] == ",".join(BANDS)


def test_centre_crop_reduces_size_and_keeps_georeferencing(safe_dir, tmp_path):
    out = tmp_path / "crop.tif"
    assert imp.main(["--input", str(safe_dir), "--output", str(out), "--size", "32"]) == 0

    info = read_info(out)
    assert (info.width, info.height) == (32, 32)
    assert info.resolution == (10.0, 10.0)
    assert info.crs is not None


def test_an_oversized_crop_is_rejected(safe_dir, tmp_path):
    code = imp.main(
        ["--input", str(safe_dir), "--output", str(tmp_path / "x.tif"), "--size", "9999"]
    )
    assert code == 1


def test_list_mode_writes_nothing(safe_dir, tmp_path):
    out = tmp_path / "none.tif"
    assert imp.main(["--input", str(safe_dir), "--output", str(out), "--list"]) == 0
    assert not out.exists()


def test_a_missing_input_exits_nonzero(tmp_path):
    assert imp.main(["--input", str(tmp_path / "nope")]) != 0


# ---------------------------------------------------------------------------
# Integration with the rest of the pipeline
# ---------------------------------------------------------------------------
def test_imported_scene_is_readable_by_the_pipeline(safe_dir, tmp_path):
    """The importer's output must satisfy the reader the pipeline actually uses."""
    from src.data.geotiff import read_raster
    from src.data.preprocessing import normalize_reflectance

    out = tmp_path / "s2.tif"
    imp.main(["--input", str(safe_dir), "--output", str(out)])

    cfg = load_config()
    array, info = read_raster(out, list(cfg.data.band_indices))
    assert array.shape[0] == len(cfg.data.bands)

    reflectance = normalize_reflectance(array, dn_scale=float(cfg.data.dn_scale))
    assert 0.0 <= reflectance.min() and reflectance.max() <= 1.0
    # B04 filled at 3000, offset -1000 -> 2000 DN -> 0.20 reflectance.
    assert reflectance[0].mean() == pytest.approx(0.20, abs=1e-4)


def test_non_ten_metre_bands_are_refused(safe_dir, tmp_path):
    """B11/B12 are 20 m; silently stacking them onto a 10 m grid would be wrong."""
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    data = load_config().to_dict()
    data["data"]["bands"] = ["B04", "B03", "B02", "B11"]
    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    code = imp.main(["--input", str(safe_dir), "--config", str(cfg_path)])
    assert code == 2
