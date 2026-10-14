"""Reproducible software fixtures, not medical images or anatomical simulations."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from utils.image_utils import png_bytes


def synthetic_sample(sample="A", seed=42, size=128):
    if sample not in ("A", "B", "C"):
        raise ValueError("Sample must be A, B or C.")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    r = (x / 0.78) ** 2 + (y / 0.92) ** 2
    brain = r < 1
    texture = 0.012 * np.cos(21 * x) * np.cos(19 * y)
    noise = gaussian_filter(rng.normal(0, 0.004, (size, size)), 1)
    noise = (noise + noise[:, ::-1]) / 2
    image = np.where(brain, 0.26 + 0.22 * np.exp(-1.8 * r) + texture + noise, 0)
    # Broad, low-contrast paired interior structures and dark ventricles.
    for side in (-1, 1):
        image += 0.055 * np.exp(-((x - side * 0.3) ** 2 / 0.025 + y ** 2 / 0.22)) * brain
        image -= 0.15 * np.exp(-((x - side * 0.11) ** 2 / 0.004 + y ** 2 / 0.045)) * brain
    mask = np.zeros((size, size), bool)
    centers = [] if sample == "A" else [(0.34, -0.24, 0.085)]
    if sample == "C":
        centers.append((-0.34, 0.32, 0.10))
    for cx, cy, radius in centers:
        region = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
        mask |= region
        image += 0.48 * gaussian_filter(region.astype(float), 0.65)
    return np.clip(image, 0, 1).astype(np.float32), mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("sample_data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for sample in "ABC":
        image, mask = synthetic_sample(sample)
        (args.output / f"sample_{sample}.png").write_bytes(png_bytes(image))
        (args.output / f"sample_{sample}_injection_mask.png").write_bytes(png_bytes(mask))
    print(f"Saved synthetic software test images (seed 42) to {args.output}")


if __name__ == "__main__":
    main()
