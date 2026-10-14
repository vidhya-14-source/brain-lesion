import numpy as np
from config import DEFAULT_MODALITY_WEIGHTS


def fuse_maps(maps, weights=None, method="equal"):
    """Fuse anomaly or panel maps across modalities.
    
    Supports:
    - 'equal': unweighted average 1/N (default for standard fusion)
    - 'weighted': uses calibrated validation weights (e.g. FLAIR 0.50, T2 0.30, T1 0.20)
    """
    if not maps:
        raise ValueError("Provide at least one modality.")
    keys = list(maps)
    arrays = [np.asarray(maps[k], np.float32) for k in keys]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Modality maps must share a common shape.")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Modality maps must contain finite values.")

    if weights is not None:
        w = np.array([float(weights.get(k, 1.0)) for k in keys], dtype=float)
    elif method == "weighted":
        # Calibrated weights prioritizing lesion-sensitive sequences
        w = np.array([float(DEFAULT_MODALITY_WEIGHTS.get(k, 1.0)) for k in keys], dtype=float)
    else:
        # Equal weights
        w = np.ones(len(keys), dtype=float)

    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Fusion weights must be finite, nonnegative and have a positive sum.")
    w /= w.sum()
    fused = np.clip(np.average(np.stack(arrays), axis=0, weights=w), 0, 1).astype(np.float32)
    return fused, dict(zip(keys, [float(x) for x in w]))


def fuse_multichannel_residuals(recon_errors, modality_names=("T1", "T2", "FLAIR"), weights=None):
    """Combine multi-channel reconstruction errors from a joint MultiChannelVAE."""
    if weights is None:
        weights = DEFAULT_MODALITY_WEIGHTS
    w = np.array([weights.get(name, 1.0) for name in modality_names], dtype=np.float32)
    w /= w.sum()
    if recon_errors.ndim == 3 and recon_errors.shape[0] == len(modality_names):
        fused = np.sum(recon_errors * w[:, None, None], axis=0)
    else:
        fused = np.mean(recon_errors, axis=0)
    return np.clip(fused, 0, 1).astype(np.float32)
