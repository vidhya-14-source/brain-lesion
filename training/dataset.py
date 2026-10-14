import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.image_utils import read_image, select_slice
from processing.preprocessing import preprocess, extract_brain_mask
from processing.fuzzy_enhancement import fuzzy_enhance
from config import IMAGE_SIZE, DEFAULT_NORM_METHOD


def extract_patient_id(path):
    """Extract patient or subject identifier from path or filename."""
    path = Path(path)
    stem = path.name.lower()
    # Check for common BIDS patterns: sub-XXXX
    bids = re.search(r"(sub-[a-z0-9]+)", stem)
    if bids:
        return bids.group(1)
    # Check for patient_XX or subject_XX
    pat = re.search(r"(patient[_-]?[0-9]+|subject[_-]?[0-9]+|case[_-]?[0-9]+|pat[0-9]+)", stem)
    if pat:
        return pat.group(1)
    # Check parent directory name if indicative
    parent_name = path.parent.name.lower()
    if parent_name not in ("", "train", "val", "validation", "test", "healthy_data", "data"):
        return parent_name
    # Fallback to file stem prefix before first underscore or dot
    prefix = re.split(r"[._-]", path.stem)[0]
    return prefix if prefix else path.stem


class HealthyMRIDataset(Dataset):
    """Multi-slice MRI dataset using all brain-containing axial slices (middle 30-70%).
    
    Skips empty/background slices and guarantees clean patient-level splitting.
    """
    def __init__(self, folder, slice_range=(0.30, 0.70), image_size=IMAGE_SIZE,
                 norm_method=DEFAULT_NORM_METHOD, use_fuzzy=True, min_brain_pixels=50,
                 patient_whitelist=None):
        folder = Path(folder)
        self.folder = folder
        self.image_size = image_size
        self.norm_method = norm_method
        self.use_fuzzy = use_fuzzy
        self.slice_range = slice_range
        self.min_brain_pixels = min_brain_pixels
        self.patient_whitelist = set(patient_whitelist) if patient_whitelist is not None else None

        self.files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".nii", ".nii.gz")))
        if not self.files:
            raise ValueError(f"No supported reference images found in {folder}.")

        self.slice_entries = []  # list of (file_path, slice_idx, patient_id)
        self.patient_to_slices = defaultdict(list)
        self._volume_cache = {}  # small cache for fast sequential volume reads

        self._index_dataset()
        if not self.slice_entries:
            raise ValueError(f"No brain-containing slices found in {folder} within extent {slice_range}.")

    def _index_dataset(self):
        for path in self.files:
            pid = extract_patient_id(path)
            if self.patient_whitelist is not None and pid not in self.patient_whitelist:
                continue

            try:
                array, _ = read_image(path.read_bytes(), path.name)
            except Exception:
                continue

            if array.ndim == 2:
                # 2D image
                mask = extract_brain_mask(array)
                if mask.sum() >= self.min_brain_pixels:
                    entry = (path, None, pid)
                    self.slice_entries.append(entry)
                    self.patient_to_slices[pid].append(entry)
            elif array.ndim == 3:
                # 3D volume: inspect axial slices along dimension 2
                num_slices = array.shape[2]
                brain_counts = []
                for z in range(num_slices):
                    sl = array[:, :, z]
                    # Fast heuristic count
                    lo, hi = float(sl.min()), float(sl.max())
                    if hi - lo > 1e-4:
                        brain_counts.append((sl - lo) / (hi - lo) > 0.05)
                    else:
                        brain_counts.append(np.zeros_like(sl, dtype=bool))

                # Identify brain extent (slices where brain tissue is present)
                valid_z = [z for z, b in enumerate(brain_counts) if b.sum() >= self.min_brain_pixels]
                if valid_z:
                    z_min, z_max = min(valid_z), max(valid_z)
                    extent = z_max - z_min
                    z_start = int(np.floor(z_min + self.slice_range[0] * extent))
                    z_end = int(np.ceil(z_min + self.slice_range[1] * extent))
                    z_start = max(z_min, min(z_start, z_max))
                    z_end = max(z_start, min(z_end, z_max))

                    for z in range(z_start, z_end + 1):
                        if brain_counts[z].sum() >= self.min_brain_pixels:
                            entry = (path, z, pid)
                            self.slice_entries.append(entry)
                            self.patient_to_slices[pid].append(entry)

    def __len__(self):
        return len(self.slice_entries)

    def _load_slice(self, path, slice_idx):
        key = str(path)
        if key in self._volume_cache:
            array = self._volume_cache[key]
        else:
            array, _ = read_image(path.read_bytes(), path.name)
            if len(self._volume_cache) > 4:
                self._volume_cache.clear()
            self._volume_cache[key] = array

        if slice_idx is None:
            return array if array.ndim == 2 else select_slice(array)
        return select_slice(array, slice_idx)

    def __getitem__(self, index):
        path, slice_idx, _ = self.slice_entries[index]
        try:
            sl = self._load_slice(path, slice_idx)
            _, processed = preprocess(sl, size=self.image_size, norm_method=self.norm_method)
            _, enhanced = fuzzy_enhance(processed, enabled=self.use_fuzzy)
            return torch.from_numpy(enhanced)[None]
        except ValueError as exc:
            raise ValueError(f"Invalid training slice {path.name} (slice {slice_idx}): {exc}") from exc


def split_patients(folder, val_ratio=0.2, seed=42):
    """Split folder into patient ID sets to prevent cross-slice data leakage."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name.lower().endswith(
        (".png", ".jpg", ".jpeg", ".nii", ".nii.gz")))
    patients = sorted(list({extract_patient_id(p) for p in files}))
    rng = np.random.default_rng(seed)
    shuffled = list(patients)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_patients = set(shuffled[:n_val])
    train_patients = set(shuffled[n_val:])
    return train_patients, val_patients
