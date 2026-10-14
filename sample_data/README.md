# Synthetic software test images

These are **not medical images**. Seed: 42. Generate again with:

```powershell
python -m utils.synthetic
```

- `sample_A.png`: mostly symmetric reference without injected bright regions.
- `sample_B.png`: one small artificial bright region.
- `sample_C.png`: two artificial bright regions.
- `*_injection_mask.png`: known injection locations, exclusively for software tests.

Do not upload the injection masks as MRI inputs or train on this folder: it contains masks as well as images. The pipeline never reads these masks. Synthetic appearance and successful localization do not establish performance on MRI.
