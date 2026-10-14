from io import BytesIO
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 64 * 1024 * 1024
MAX_VOXELS = 32_000_000


def read_image(data, name):
    """Return a raster or canonical 3D NIfTI volume and provenance metadata."""
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Upload a nonempty image smaller than 64 MB.")
    lower = name.lower()
    try:
        if lower.endswith((".nii", ".nii.gz")):
            suffix = ".nii.gz" if lower.endswith(".gz") else ".nii"
            with tempfile.TemporaryDirectory(prefix="brain_mri_") as folder:
                path = Path(folder) / ("upload" + suffix)
                path.write_bytes(data)
                # Disable memory mapping so Windows can remove the temporary file.
                nii = nib.load(str(path), mmap=False, keep_file_open=False)
                if len(nii.shape) not in (2, 3, 4) or np.prod(nii.shape) > MAX_VOXELS:
                    raise ValueError("NIfTI must be 2D, 3D or 4D and contain at most 32 million voxels.")
                if len(nii.shape) == 4:
                    if nii.shape[3] != 1:
                        raise ValueError("Multi-volume 4D scans are unsupported; export one 3D volume first.")
                    nii = nib.Nifti1Image(np.asarray(nii.dataobj)[..., 0], nii.affine)
                nii = nib.as_closest_canonical(nii)
                array = nii.get_fdata(dtype=np.float32)
                metadata = {"kind": "NIfTI", "shape": list(array.shape), "orientation": "RAS+ canonical; axial slice display",
                            "spacing_mm": [float(v) for v in nii.header.get_zooms()[:3]]}
                if array.ndim == 2:
                    array = np.rot90(array).copy()
        elif lower.endswith((".png", ".jpg", ".jpeg")):
            with Image.open(BytesIO(data)) as img:
                if img.width * img.height > MAX_VOXELS:
                    raise ValueError("Raster image is too large (maximum 32 million pixels).")
                img = ImageOps.exif_transpose(img)
                array = np.asarray(img.convert("F") if img.mode in ("I", "F", "I;16") else img.convert("L"), dtype=np.float32)
            metadata = {"kind": "Raster", "shape": list(array.shape), "orientation": "As uploaded; orientation not verified"}
        else:
            raise ValueError("Supported files: PNG, JPG, JPEG, NIfTI .nii and .nii.gz.")
        if not np.isfinite(array).all() or min(array.shape[:2]) < 2:
            raise ValueError("Image contains invalid values or dimensions.")
        return array, metadata
    except (ValueError, UnidentifiedImageError, OSError, nib.filebasedimages.ImageFileError) as exc:
        raise ValueError(f"Unable to read image: {exc}") from exc
    except Exception as exc:
        raise ValueError("Unable to read this file. Check that it is a supported, intact image.") from exc


def select_slice(array, index=None):
    if array.ndim == 2:
        return array.copy()
    if array.ndim != 3:
        raise ValueError("Expected a 2D image or a 3D volume.")
    index = array.shape[2] // 2 if index is None else int(index)
    if not 0 <= index < array.shape[2]:
        raise ValueError("Slice index is outside this volume.")
    return np.rot90(array[:, :, index]).copy()


def png_bytes(image):
    x = np.asarray(image)
    if x.dtype != np.uint8:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(x).save(output, format="PNG")
    return output.getvalue()
