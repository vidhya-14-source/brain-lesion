import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes
from config import DEFAULT_CALIBRATED_THRESHOLD, DEFAULT_CALIBRATED_PERCENTILE
from processing.preprocessing import extract_brain_mask


def demo_reconstruction(image):
    """Illustrative reference: opening plus a left/right symmetry heuristic."""
    opened = opening(image, footprint=disk(9))
    residual = np.maximum(image - opened, 0)
    return (opened + np.minimum(residual, residual[:, ::-1])).astype(np.float32)


def foreground_mask(image):
    """Real anatomical brain mask with fallback for synthetic images."""
    mask = extract_brain_mask(image)
    if mask.sum() > 50:
        return mask
    return binary_fill_holes(image > 0.03)


def calibrate_threshold(healthy_errors, masks=None, percentile=DEFAULT_CALIBRATED_PERCENTILE):
    """Calibrate anomaly threshold on healthy validation reconstruction errors.
    
    Instead of calculating a per-image threshold which can be distorted by lesions
    or normal variations, computes the empirical percentile (e.g. 95th or 99th)
    across all healthy brain pixels.
    """
    if isinstance(healthy_errors, (list, tuple)):
        all_vals = []
        for i, err in enumerate(healthy_errors):
            m = masks[i] if (masks is not None and i < len(masks)) else foreground_mask(err)
            if m.any():
                all_vals.append(err[m].ravel())
        if not all_vals:
            return float(DEFAULT_CALIBRATED_THRESHOLD)
        combined = np.concatenate(all_vals)
    else:
        m = masks if masks is not None else foreground_mask(healthy_errors)
        combined = healthy_errors[m].ravel() if m.any() else healthy_errors.ravel()

    if combined.size == 0:
        return float(DEFAULT_CALIBRATED_THRESHOLD)
    threshold = float(np.percentile(combined, percentile))
    return max(0.01, min(1.0, threshold))


def anomaly_map(image, reconstruction, mask=None):
    if image.shape != reconstruction.shape:
        raise ValueError("Input and reconstruction dimensions must match.")
    error = np.abs(image.astype(np.float32) - reconstruction.astype(np.float32))
    heat = np.clip(gaussian_filter(error, 0.6), 0, 1)
    fg = foreground_mask(image) if mask is None else mask.astype(bool)
    heat *= fg
    return error, heat.astype(np.float32)


def threshold_map(heat, foreground=None, mode="Automatic", threshold=0.2,
                  min_area=18, calibrated_threshold=DEFAULT_CALIBRATED_THRESHOLD):
    """Threshold anomaly heatmap into binary candidate regions.
    
    Modes:
    - 'Calibrated': Uses threshold derived from healthy validation data (fixed baseline).
    - 'Automatic': Legacy per-image rule max(0.10, mean + 2*std).
    - 'Manual': User-selected fixed threshold [0, 1].
    """
    if mode not in ("Calibrated", "Automatic", "Manual"):
        raise ValueError("Threshold mode must be Automatic or Manual.") if mode not in ("Calibrated", "Automatic", "Manual") else None
    if not 0 <= threshold <= 1 or min_area < 1:
        raise ValueError("Threshold must be 0–1 and minimum area positive.")

    foreground = np.ones_like(heat, dtype=bool) if foreground is None else foreground.astype(bool)
    values = heat[foreground]

    if mode == "Calibrated":
        effective_threshold = float(calibrated_threshold)
    elif mode == "Automatic":
        effective_threshold = min(1.0, max(0.10, float(values.mean() + 2 * values.std()))) if values.size else 0.10
    else:
        effective_threshold = float(threshold)

    mask = (heat > effective_threshold) & foreground
    mask = opening(mask, disk(1))
    mask = closing(mask, disk(1))
    mask = remove_small_objects(mask, min_size=min_area)
    mask = remove_small_holes(mask, area_threshold=12) & foreground
    return mask.astype(bool), float(effective_threshold)


def intensity_threshold_baseline(image, foreground=None, percentile=98.0, min_area=18):
    """Simple non-neural baseline: flags hyperintense brain regions above percentile."""
    fg = foreground_mask(image) if foreground is None else foreground.astype(bool)
    if not fg.any():
        return np.zeros_like(image, dtype=bool), np.zeros_like(image, dtype=np.float32)
    values = image[fg]
    thresh = float(np.percentile(values, percentile))
    mask = (image > thresh) & fg
    mask = opening(mask, disk(1))
    mask = closing(mask, disk(1))
    mask = remove_small_objects(mask, min_size=min_area)
    mask = remove_small_holes(mask, area_threshold=12) & fg
    heat = np.clip((image - thresh) / max(1.0 - thresh, 1e-6), 0, 1) * fg
    return mask.astype(bool), heat.astype(np.float32)


def score_error(error, foreground):
    values = error[foreground]
    mean_error = float(values.mean()) if values.size else 0.0
    score = float(np.clip(mean_error * 100, 0, 100))
    band = "Lower anomaly signal" if score <= 30 else "Moderate anomaly signal" if score <= 60 else "Higher anomaly signal"
    return score, mean_error, band
