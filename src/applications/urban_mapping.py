"""Phase 9: does super-resolution improve an actual remote-sensing task?

The question
------------
PSNR going up does not mean a product is more useful. This module runs a
land-cover classification on the bicubic baseline and on the AI-SR output and
reports whether the classification changed for the better — the only evidence
that matters for the problem statement.

Experiment design
-----------------
The classifier is fitted **once, on the reference image**, and the resulting
class centroids are then applied unchanged to every input. This is deliberate:
if each image were clustered independently, the cluster labels would be
arbitrary and the comparison would reduce to a label-matching exercise. With
shared centroids, class ``k`` means the same thing everywhere and accuracy
against the reference map is directly meaningful.

Under Wald's protocol the reference map is derived from the **original observed
scene**, which is real data — so the accuracy numbers are genuine, with one
stated caveat: the reference *labels* come from unsupervised clustering of that
observation, not from field survey. They measure "does SR recover the map you
would get from truly finer imagery", not "does SR recover ground truth land
cover". Supplying ``application.labels_path`` replaces the clustering with real
labels and removes that caveat.

Nothing here fabricates numbers: when no reference is available the module
returns the pipeline output with ``quantitative: false`` and reports only
structural descriptors, explicitly labelled as *not* accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.cluster.vq import kmeans2

EPS = 1e-8


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def normalized_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a - b) / (a + b)`` — the normalised-difference form used by NDVI/NDWI."""
    denominator = a + b
    return np.divide(a - b, denominator, out=np.zeros_like(a), where=np.abs(denominator) > EPS)


def build_features(
    image: np.ndarray,
    band_names: Sequence[str],
    use_ndvi: bool = True,
    use_ndwi: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Stack reflectance bands with spectral indices into a feature cube.

    NDVI and NDWI are included because they separate the classes that matter
    for urban mapping far better than raw reflectance — and because they are
    ratio-based, they are exactly the quantities a spectrally inconsistent SR
    model would corrupt. Including them makes this experiment sensitive to the
    failure mode the whole project is designed to avoid.
    """
    lookup = {name: idx for idx, name in enumerate(band_names)}
    features = [image[i] for i in range(image.shape[0])]
    names = list(band_names)

    if use_ndvi and "B08" in lookup and "B04" in lookup:
        features.append(normalized_index(image[lookup["B08"]], image[lookup["B04"]]))
        names.append("NDVI")
    if use_ndwi and "B03" in lookup and "B08" in lookup:
        features.append(normalized_index(image[lookup["B03"]], image[lookup["B08"]]))
        names.append("NDWI")

    return np.stack(features).astype(np.float32), names


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
@dataclass
class CentroidClassifier:
    """Minimum-distance-to-centroid classifier over standardised features.

    Deliberately simple. A high-capacity classifier could compensate for a poor
    reconstruction with its own learned priors, which would confound the
    measurement — the experiment must isolate the contribution of the imagery,
    not of the classifier.
    """

    centroids: np.ndarray          # (K, F)
    mean: np.ndarray               # (F,)
    std: np.ndarray                # (F,)
    class_names: list[str] = field(default_factory=list)

    @classmethod
    def fit_unsupervised(
        cls,
        features: np.ndarray,
        n_classes: int,
        class_names: Sequence[str] | None = None,
        seed: int = 42,
    ) -> "CentroidClassifier":
        flat, mean, std = _standardise(features)
        centroids, _ = kmeans2(flat, n_classes, minit="++", seed=seed, iter=25)
        names = list(class_names or [])[:n_classes]
        names += [f"class_{i}" for i in range(len(names), n_classes)]
        return cls(centroids=centroids, mean=mean, std=std, class_names=names)

    @classmethod
    def fit_supervised(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: Sequence[str] | None = None,
    ) -> "CentroidClassifier":
        flat, mean, std = _standardise(features)
        flat_labels = labels.reshape(-1)
        classes = np.unique(flat_labels[flat_labels >= 0])
        centroids = np.stack([flat[flat_labels == c].mean(axis=0) for c in classes])
        names = list(class_names or [])[: len(classes)]
        names += [f"class_{int(c)}" for c in classes[len(names) :]]
        return cls(centroids=centroids, mean=mean, std=std, class_names=names)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Classify an ``(F, H, W)`` feature cube into an ``(H, W)`` label map.

        Standardisation uses the statistics learnt on the reference image, so
        every input is projected into the same feature space.
        """
        _, height, width = features.shape
        flat = features.reshape(features.shape[0], -1).T
        flat = (flat - self.mean) / self.std

        # (N, K) squared distances via the expansion |x|^2 - 2x.c + |c|^2 —
        # avoids materialising an (N, K, F) difference tensor, which for a
        # full scene would be tens of gigabytes.
        distances = (
            (flat**2).sum(axis=1, keepdims=True)
            - 2.0 * flat @ self.centroids.T
            + (self.centroids**2).sum(axis=1)
        )
        return distances.argmin(axis=1).reshape(height, width).astype(np.int16)

    @property
    def n_classes(self) -> int:
        return int(self.centroids.shape[0])


def _standardise(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = features.reshape(features.shape[0], -1).T.astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < EPS] = 1.0
    return (flat - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def confusion_matrix(truth: np.ndarray, predicted: np.ndarray, n_classes: int) -> np.ndarray:
    valid = (truth >= 0) & (truth < n_classes) & (predicted >= 0) & (predicted < n_classes)
    return np.bincount(
        truth[valid].astype(np.int64) * n_classes + predicted[valid].astype(np.int64),
        minlength=n_classes**2,
    ).reshape(n_classes, n_classes)


def classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    n_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Overall accuracy, per-class IoU, mean IoU and Cohen's kappa."""
    cm = confusion_matrix(truth, predicted, n_classes)
    total = cm.sum()
    if total == 0:
        return {"overall_accuracy": None, "mean_iou": None, "kappa": None, "per_class": {}}

    correct = np.trace(cm)
    accuracy = correct / total

    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = np.divide(intersection, union, out=np.full(n_classes, np.nan), where=union > 0)

    expected = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total**2)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 0.0

    names = list(class_names or [])[:n_classes]
    names += [f"class_{i}" for i in range(len(names), n_classes)]

    return {
        "overall_accuracy": float(accuracy),
        "mean_iou": float(np.nanmean(iou)),
        "kappa": float(kappa),
        "per_class": {
            name: {
                "iou": None if np.isnan(iou[i]) else float(iou[i]),
                "support": int(cm[i].sum()),
            }
            for i, name in enumerate(names)
        },
    }


def structural_descriptors(labels: np.ndarray) -> dict[str, float]:
    """Reference-free descriptors of a label map.

    These describe how *detailed* a map is, not how *correct* it is. A model
    that hallucinates texture would score well here, which is precisely why
    they are never reported as accuracy.
    """
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary_density = (horizontal.sum() + vertical.sum()) / labels.size

    counts = np.bincount(labels.reshape(-1).astype(np.int64))
    proportions = counts[counts > 0] / counts.sum()
    entropy = float(-(proportions * np.log2(proportions)).sum())

    return {
        "boundary_density": float(boundary_density),
        "class_entropy_bits": entropy,
        "n_classes_present": int((counts > 0).sum()),
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
@dataclass
class ApplicationResult:
    """Outcome of the downstream experiment, including its honesty flags."""

    quantitative: bool
    task: str
    metrics: dict[str, dict[str, Any]]
    maps: dict[str, np.ndarray]
    class_names: list[str]
    caveats: list[str]
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view — the label maps themselves are excluded."""
        return {
            "task": self.task,
            "quantitative": self.quantitative,
            "class_names": self.class_names,
            "metrics": self.metrics,
            "caveats": self.caveats,
            "verdict": self.verdict,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path


def run_experiment(
    products: dict[str, np.ndarray],
    cfg,
    *,
    reference: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> ApplicationResult:
    """Classify each product and compare the resulting land-cover maps.

    ``products`` maps a method name (``"bicubic"``, ``"ai_sr"``, ...) to a
    ``(C, H, W)`` image on a common grid. ``reference`` is the ground-truth
    image at the same resolution, when one exists.
    """
    band_names = list(cfg.data.bands)
    n_classes = int(cfg.application.n_classes)
    class_names = list(cfg.application.class_names)
    seed = int(cfg.project.seed)

    use_ndvi = bool(cfg.application.use_ndvi)
    use_ndwi = bool(cfg.application.use_ndwi)

    def features(image: np.ndarray) -> np.ndarray:
        return build_features(image, band_names, use_ndvi, use_ndwi)[0]

    caveats: list[str] = []

    # -- fit the classifier on the best available source ------------------
    if labels is not None and reference is not None:
        classifier = CentroidClassifier.fit_supervised(
            features(reference), labels, class_names
        )
        truth = labels.astype(np.int16)
        quantitative = True
    elif reference is not None:
        classifier = CentroidClassifier.fit_unsupervised(
            features(reference), n_classes, class_names, seed=seed
        )
        truth = classifier.predict(features(reference))
        quantitative = True
        caveats.append(
            "Reference land-cover labels were derived by unsupervised clustering "
            "of the reference imagery, not from field survey. Accuracy therefore "
            "measures agreement with the map obtainable from truly finer imagery, "
            "not agreement with surveyed ground truth."
        )
    else:
        anchor = next(iter(products.values()))
        classifier = CentroidClassifier.fit_unsupervised(
            features(anchor), n_classes, class_names, seed=seed
        )
        truth = None
        quantitative = False
        caveats.append(
            "NO REFERENCE IMAGERY AVAILABLE — this experiment is NOT quantitative. "
            "Only structural descriptors are reported; they describe map detail, "
            "not map accuracy. Supply a co-registered high-resolution reference "
            "(or run under the reduced-resolution protocol) for real numbers."
        )

    # -- classify every product with the same centroids -------------------
    maps = {name: classifier.predict(features(image)) for name, image in products.items()}
    if truth is not None:
        maps["reference"] = truth

    metrics: dict[str, dict[str, Any]] = {}
    for name, label_map in maps.items():
        if name == "reference":
            continue
        entry: dict[str, Any] = {"structural": structural_descriptors(label_map)}
        if truth is not None:
            entry.update(
                classification_metrics(truth, label_map, classifier.n_classes, classifier.class_names)
            )
        metrics[name] = entry

    verdict = _verdict(metrics, quantitative)
    if not quantitative:
        caveats.append("Verdict is inconclusive without a reference.")

    return ApplicationResult(
        quantitative=quantitative,
        task=str(cfg.application.task),
        metrics=metrics,
        maps=maps,
        class_names=classifier.class_names,
        caveats=caveats,
        verdict=verdict,
    )


def _verdict(metrics: dict[str, dict[str, Any]], quantitative: bool) -> str:
    """State plainly whether SR helped, hurt, or could not be judged."""
    if not quantitative:
        return (
            "NOT MEASURED — no reference available. The pipeline ran end to end, "
            "but no claim about downstream benefit is made."
        )
    if "ai_sr" not in metrics or "bicubic" not in metrics:
        return "NOT COMPARED — both the AI-SR and bicubic products are required."

    ai = metrics["ai_sr"].get("overall_accuracy")
    base = metrics["bicubic"].get("overall_accuracy")
    if ai is None or base is None:
        return "NOT COMPARED — accuracy could not be computed."

    delta = ai - base
    direction = "improved" if delta > 0 else ("reduced" if delta < 0 else "left unchanged")
    significant = "" if abs(delta) >= 0.005 else " (within noise; not a meaningful difference)"
    return (
        f"Super-resolution {direction} land-cover classification accuracy by "
        f"{delta:+.4f} ({base:.4f} -> {ai:.4f}) versus the bicubic baseline{significant}."
    )
