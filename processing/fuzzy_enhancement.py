import cv2
import numpy as np
from scipy.ndimage import uniform_filter


def fuzzy_enhance(image, enabled=True):
    """Bounded fuzzy intensification with entropy-adapted local contrast blending.
    
    If enabled=False (ablation mode), returns the unenhanced image directly.
    """
    x = np.clip(np.asarray(image, np.float32), 0, 1)
    if not enabled:
        return x.copy(), x.copy()
    foreground = x > 0.02
    if not foreground.any() or float(x.std()) < 1e-7:
        return x.copy(), x.copy()
    values = x[foreground]
    lo, hi = np.percentile(values, [2, 98])
    membership = np.clip((x - lo) / max(float(hi - lo), 1e-6), 0, 1)
    m = np.clip(membership[foreground], 1e-6, 1 - 1e-6)
    entropy = float(np.mean(-(m * np.log2(m) + (1 - m) * np.log2(1 - m))))
    local_var = np.maximum(uniform_filter(x * x, 7) - uniform_filter(x, 7) ** 2, 0)
    local_std = float(np.sqrt(local_var)[foreground].mean())
    strength = float(np.clip(0.12 + 0.15 * entropy + 0.10 * (1 - values.std()) - local_std, 0.1, 0.32))
    intensified = np.where(membership <= 0.5, 2 * membership ** 2, 1 - 2 * (1 - membership) ** 2)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply((x * 255).astype(np.uint8)) / 255.0
    enhanced = (1 - strength) * x + strength * (0.75 * intensified + 0.25 * clahe)
    enhanced[~foreground] = x[~foreground]
    return x.copy(), np.clip(enhanced, 0, 1).astype(np.float32)
