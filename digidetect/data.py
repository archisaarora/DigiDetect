"""Datasets: the bundled 6,000-image MNIST sample (works offline) and full MNIST."""
from __future__ import annotations

import gzip
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIDE = 14
DIM = SIDE * SIDE
BUNDLED = Path(__file__).resolve().parent.parent / "data" / "digits.npz"
CACHE = Path(os.environ.get("DIGIDETECT_CACHE", Path.home() / ".cache" / "digidetect"))


@dataclass
class Split:
    x: np.ndarray   # (n, 196) float32 in 0..1
    y: np.ndarray   # (n,) uint8

    @property
    def n(self) -> int:
        return len(self.y)


def load_bundled() -> tuple[Split, Split]:
    """5,000 training and 1,000 held-out test digits, already 14x14."""
    d = np.load(BUNDLED)
    return (Split(d["train_x"].astype(np.float32) / 255, d["train_y"]),
            Split(d["test_x"].astype(np.float32) / 255, d["test_y"]))


# Public mirrors are tried file by file, so one missing file doesn't sink the download.
MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
MNIST_MIRRORS = [
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://cdn.jsdelivr.net/gh/fgnt/mnist@master/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]


def _fetch(filename: str, log=print) -> bytes:
    """Return the decompressed IDX bytes for one MNIST file (cached on disk)."""
    cached = CACHE / filename
    if cached.exists():
        raw = cached.read_bytes()
    else:
        errors = []
        raw = None
        for i, mirror in enumerate(MNIST_MIRRORS, 1):
            log(f"Downloading {filename} · source {i}/{len(MNIST_MIRRORS)}")
            try:
                with urllib.request.urlopen(mirror + filename, timeout=45) as r:
                    raw = r.read()
                if raw[:2] != b"\x1f\x8b":
                    raise ValueError("not a gzip file (often a blocked request or HTML error page)")
                break
            except Exception as e:  # noqa: BLE001 - try the next mirror
                errors.append(f"{mirror.split('/')[2]}: {e}")
                raw = None
        if raw is None:
            raise RuntimeError(f"Could not download {filename}. " + " | ".join(errors))
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(raw)
    return gzip.decompress(raw)


def parse_idx(images: bytes, labels: bytes) -> Split:
    """Parse MNIST IDX files and average-pool each 2x2 block down to 14x14."""
    count, rows, cols = (int.from_bytes(images[o:o + 4], "big") for o in (4, 8, 12))
    if count != int.from_bytes(labels[4:8], "big") or (rows, cols) != (28, 28):
        raise ValueError("The MNIST files have an unexpected format.")
    img = np.frombuffer(images, np.uint8, offset=16).reshape(count, 28, 28).astype(np.float32)
    pooled = img.reshape(count, 14, 2, 14, 2).mean(axis=(2, 4)) / 255
    return Split(pooled.reshape(count, DIM).astype(np.float32),
                 np.frombuffer(labels, np.uint8, offset=8).copy())


def load_full_mnist(log=print) -> tuple[Split, Split]:
    ti, tl, vi, vl = (_fetch(f, log) for f in MNIST_FILES)
    train, test = parse_idx(ti, tl), parse_idx(vi, vl)
    if (train.n, test.n) != (60000, 10000):
        raise ValueError("Expected 60,000 training and 10,000 test images.")
    return train, test
