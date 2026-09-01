"""Paired LR/HR patch datasets (Phase 2).

Storage design
--------------
Prepared patches live in **one memory-mapped ``.npy`` array per scene set**,
plus a JSON manifest, rather than thousands of small files. This matters:
a 20k-patch dataset as individual files costs 20k open/close syscalls per
epoch and is the single biggest throughput sink on Windows and on network
storage. A memmap slice is effectively a page-cache read, and it composes
correctly with multiple DataLoader workers because each worker opens its own
read-only view.

Only the **HR reference** is stored. The LR input is synthesised on the fly by
:func:`~src.data.preprocessing.degrade` inside the worker process. That halves
disk usage, keeps the degradation model a live config knob (changing it does
not invalidate the prepared dataset), and costs a few hundred microseconds per
patch on a worker thread that would otherwise be idle.

Split design
------------
Patches are split **spatially**, not randomly. With a stride smaller than the
patch size, neighbouring patches overlap; a random split would put overlapping
pixels in both train and validation and inflate validation metrics. Whole
spatial blocks are assigned to one split, so no validation pixel is ever seen
during training.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import DegradationConfig, degrade

MANIFEST_NAME = "manifest.json"
ARRAY_NAME = "patches.npy"

# Patches are stored as uint16 DN to halve the memmap footprint versus float32.
STORE_DTYPE = "uint16"
STORE_SCALE = 10000.0


@dataclass(frozen=True)
class PatchRecord:
    """Where a stored patch came from — needed for the spatial split."""

    index: int
    scene: str
    row_off: int
    col_off: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "scene": self.scene,
            "row_off": self.row_off,
            "col_off": self.col_off,
        }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(directory: Path, manifest: dict[str, Any]) -> Path:
    path = Path(directory) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_manifest(directory: Path) -> dict[str, Any]:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no prepared dataset at {directory} — run scripts/prepare_dataset.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def _block_hash(key: str) -> int:
    """Uniform 64-bit hash of a block key.

    BLAKE2b rather than ``zlib.crc32``: CRC-32 is a linear checksum with poor
    avalanche on short structured keys, and using it here made the split
    degenerate into column stripes that barely responded to the seed. A
    cryptographic hash costs microseconds and only runs once per patch at
    preparation time. ``hash()`` is not an option — it is salted per process
    and would break reproducibility across runs.
    """
    return int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big"
    )


def spatial_split(
    records: Sequence[PatchRecord],
    val_fraction: float,
    patch_size: int,
    seed: int = 42,
    block_multiple: int = 4,
) -> tuple[list[int], list[int]]:
    """Assign whole spatial blocks to train/val so overlapping patches cannot leak.

    A block is ``block_multiple`` patches on a side. Block assignment is a
    deterministic hash of ``(seed, scene, block_row, block_col)`` — reproducible
    across machines and stable if patches are later appended.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    block_px = patch_size * block_multiple
    train: list[int] = []
    val: list[int] = []
    threshold = val_fraction * float(1 << 64)

    for rec in records:
        key = f"{seed}|{rec.scene}|{rec.row_off // block_px}|{rec.col_off // block_px}"
        (val if _block_hash(key) < threshold else train).append(rec.index)

    # A dataset small enough to occupy a single block lands entirely on one
    # side. Rather than silently training with no validation (or no training
    # data at all), fall back to a deterministic index split.
    if val_fraction > 0 and (not val or not train) and len(records) > 1:
        ordered = sorted(train + val)
        cut = min(len(ordered) - 1, max(1, round(len(ordered) * val_fraction)))
        train, val = ordered[:-cut], ordered[-cut:]
    return train, val


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    """LR/HR pairs backed by a memory-mapped HR patch array.

    Yields ``{"lr": (C, h, w), "hr": (C, H, W)}`` float32 tensors in
    reflectance units, where ``H = h * scale``.
    """

    def __init__(
        self,
        directory: str | Path,
        indices: Sequence[int] | None = None,
        *,
        scale: int = 4,
        degradation: DegradationConfig | None = None,
        augment: bool = False,
        seed: int = 42,
    ):
        self.directory = Path(directory)
        self.manifest = read_manifest(self.directory)
        self.array_path = self.directory / self.manifest.get("array", ARRAY_NAME)
        if not self.array_path.exists():
            raise FileNotFoundError(f"patch array missing: {self.array_path}")

        self.scale = scale
        self.degradation = degradation or DegradationConfig()
        self.augment = augment
        self.seed = seed
        self.bands: list[str] = list(self.manifest["bands"])
        self.patch_size: int = int(self.manifest["patch_size"])

        if self.patch_size % scale:
            raise ValueError(
                f"patch_size {self.patch_size} is not divisible by scale {scale}"
            )

        total = int(self.manifest["count"])
        self.indices = list(range(total)) if indices is None else list(indices)
        bad = [i for i in self.indices if not 0 <= i < total]
        if bad:
            raise IndexError(f"patch indices out of range for a dataset of {total}: {bad[:5]}")

        # Opened lazily so the memmap handle is created inside each worker
        # process rather than inherited across a fork/spawn boundary.
        self._array: np.ndarray | None = None

    # -- memmap handling --------------------------------------------------
    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.array_path, mmap_mode="r")
        return self._array

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_array"] = None  # memmaps are not picklable to workers
        return state

    # -- Dataset protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        index = self.indices[position]
        hr = np.asarray(self.array[index], dtype=np.float32) / np.float32(STORE_SCALE)

        if self.augment:
            hr = _augment_d4(hr, self._rng(position))

        hr = np.ascontiguousarray(hr)
        lr = degrade(hr, self.scale, self.degradation, rng=self._rng(position, salt=1))

        return {
            "lr": torch.from_numpy(np.ascontiguousarray(lr)),
            "hr": torch.from_numpy(hr),
        }

    def _rng(self, position: int, salt: int = 0) -> np.random.Generator:
        """Per-sample generator: reproducible, and independent across workers.

        Seeding from ``(seed, patch index, salt)`` rather than from global
        state means two runs with the same config produce the same augmented
        batches regardless of worker count.
        """
        return np.random.default_rng(
            (self.seed * 1_000_003 + self.indices[position] * 31 + salt * 7) % (2**32)
        )


def _augment_d4(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random element of the dihedral group D4 (flips + 90 deg rotations).

    Valid for nadir-looking satellite imagery: unlike natural photographs there
    is no canonical "up", so all eight orientations are physically plausible
    and the augmentation introduces no distribution shift.
    """
    if rng.random() < 0.5:
        array = array[:, ::-1, :]
    if rng.random() < 0.5:
        array = array[:, :, ::-1]
    k = int(rng.integers(0, 4))
    if k:
        array = np.rot90(array, k=k, axes=(1, 2))
    return array


class SceneDataset(Dataset):
    """Tiles of a single scene, for windowed inference over a large raster.

    Used by the inference pipeline so a full Sentinel-2 tile is processed in
    bounded memory. Yields the tile plus the window offsets needed to stitch
    the result back together.
    """

    def __init__(
        self,
        path: str | Path,
        band_indices: Sequence[int],
        tile_size: int,
        overlap: int,
        dn_scale: float = 10000.0,
    ):
        from .geotiff import iter_windows, read_info

        self.path = Path(path)
        self.band_indices = list(band_indices)
        self.dn_scale = dn_scale
        self.info = read_info(self.path)
        self.windows = list(iter_windows(self.info, tile_size, overlap=overlap))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from .geotiff import read_raster
        from .preprocessing import normalize_reflectance

        window = self.windows[index]
        array, _ = read_raster(self.path, self.band_indices, window=window)
        tile = normalize_reflectance(array, dn_scale=self.dn_scale)
        return {
            "tile": torch.from_numpy(np.ascontiguousarray(tile)),
            "row_off": int(window.row_off),
            "col_off": int(window.col_off),
            "height": int(window.height),
            "width": int(window.width),
        }


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------
def build_dataloaders(
    directory: str | Path,
    cfg,
    *,
    scale: int | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    """Build train/val loaders from a prepared patch directory and the config."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    scale = scale if scale is not None else int(cfg.patches.scale)
    seed = int(cfg.project.seed)

    records = [PatchRecord(**r) for r in manifest["records"]]
    train_idx, val_idx = spatial_split(
        records,
        val_fraction=float(cfg.patches.val_fraction),
        patch_size=int(manifest["patch_size"]),
        seed=seed,
    )

    degradation = DegradationConfig.from_config(cfg.patches.degradation)
    common = dict(scale=scale, degradation=degradation, seed=seed)

    train_ds = PatchDataset(
        directory, train_idx, augment=bool(cfg.training.augment), **common
    )
    workers = int(cfg.training.num_workers)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        drop_last=len(train_ds) > int(cfg.training.batch_size),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
    )

    if not val_idx:
        return train_loader, None

    val_ds = PatchDataset(directory, val_idx, augment=False, **common)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    return train_loader, val_loader
