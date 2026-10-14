import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, opening, closing, dilation, remove_small_objects
from skimage.measure import label, regionprops
from config import IMAGE_SIZE, DEFAULT_NORM_METHOD


def extract_brain_mask(image):
    """Robust anatomical brain mask extraction (skull stripping heuristic).
    
    Disconnects skull/dura using morphological filtering, extracts the largest
    central brain tissue component, and fills internal holes (e.g. ventricles).
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2 or x.size == 0 or not np.isfinite(x).all():
        return np.zeros_like(x, dtype=bool)

    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=bool)

    # Initial head/tissue thresholding
    norm = (x - lo) / (hi - lo)
    initial_fg = norm > 0.05
    if not initial_fg.any():
        return np.zeros_like(x, dtype=bool)

    # Morphological opening to disconnect thin dura/skull connections
    opened = opening(initial_fg, footprint=disk(2))
    cleaned = remove_small_objects(opened, min_size=30)
    filled = binary_fill_holes(cleaned)

    # Extract largest component
    labeled = label(filled)
    props = regionprops(labeled)
    if not props:
        # Fallback to standard foreground heuristic
        return binary_fill_holes(norm > 0.03)

    largest = max(props, key=lambda p: p.area)
    mask = labeled == largest.label

    # Fill internal cavities (ventricles, sulci)
    mask = binary_fill_holes(mask)

    # Mild dilation to recover cortical boundary while excluding skull
    mask = dilation(mask, footprint=disk(1))
    return mask.astype(bool)


def normalize(image, mask=None, method=DEFAULT_NORM_METHOD):
    """Normalize MRI intensities within the brain mask.
    
    Methods:
    - 'percentile': robust to bright skull/outliers using 1st & 99th percentiles within brain
    - 'zscore': standardized (x - mean)/std mapped smoothly to [0, 1]
    - 'minmax': standard linear min-max across the image (sensitive to outliers)
    """
    x = np.asarray(image, dtype=np.float32)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Image must contain finite, nonempty pixel data.")

    if mask is None:
        mask = extract_brain_mask(x) if method in ("percentile", "zscore") else None

    if method == "percentile" and mask is not None and mask.any():
        brain_pixels = x[mask]
        p_lo, p_hi = np.percentile(brain_pixels, [1.0, 99.0])
        if p_hi - p_lo < 1e-6:
            normed = np.zeros_like(x)
        else:
            normed = np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            normed[~mask] = 0.0
        return normed.astype(np.float32)

    elif method == "zscore" and mask is not None and mask.any():
        brain_pixels = x[mask]
        mean = float(brain_pixels.mean())
        std = float(brain_pixels.std())
        if std < 1e-6:
            normed = np.zeros_like(x)
        else:
            # Map +/- 3 standard deviations to [0, 1]
            z = (x - mean) / std
            normed = np.clip((z + 3.0) / 6.0, 0.0, 1.0)
            normed[~mask] = 0.0
        return normed.astype(np.float32)

    # Fallback / Min-max
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def grayscale(image):
    x = np.asarray(image)
    if x.ndim == 3 and x.shape[-1] in (3, 4):
        x = x[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("Select a 2D grayscale or RGB image with at least 2 pixels per dimension.")
    return x


def resize_image(image, size=IMAGE_SIZE):
    """Aspect-preserving resize, centered on a zero-padded square canvas."""
    x = grayscale(image)
    h, w = x.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(x.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), np.float32)
    y, z = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, z:z + nw] = resized
    return canvas


def preprocess(image, size=IMAGE_SIZE, norm_method=DEFAULT_NORM_METHOD, return_mask=False):
    """Preprocess image: aspect-preserving resize, robust brain normalization, and smoothing."""
    resized_raw = resize_image(grayscale(image), size=size)
    mask = extract_brain_mask(resized_raw)
    normalized = normalize(resized_raw, mask=mask, method=norm_method)
    processed = cv2.GaussianBlur(normalized, (3, 3), 0.55)
    processed = np.clip(processed, 0, 1).astype(np.float32)
    if return_mask:
        return normalized, processed, mask
    return normalized, processed
