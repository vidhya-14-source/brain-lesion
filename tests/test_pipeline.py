from io import BytesIO
from pathlib import Path
import subprocess
import sys
import numpy as np
import nibabel as nib
import pytest
import torch
from PIL import Image
from processing.preprocessing import normalize, resize_image, preprocess
from processing.fuzzy_enhancement import fuzzy_enhance
from processing.anomaly import anomaly_map, threshold_map
from processing.localization import localize
from processing.multimodal import fuse_maps
from processing.pipeline import load_models, analyze_image, analyze_multimodal
from models.vae import ConvVAE, vae_loss
from models.feature_extractor import FeatureExtractor
from utils.image_utils import read_image, select_slice, png_bytes
from utils.synthetic import synthetic_sample
from training.dataset import HealthyMRIDataset


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return load_models(tmp_path_factory.mktemp("missing") / "none.pth")


def test_normalization():
    x = normalize(np.array([[-10, 0], [5, 10]]))
    assert x.min() == 0 and x.max() == 1
    assert np.isfinite(normalize(np.ones((4, 4)))).all()
    assert not normalize(np.zeros((4, 4))).any()
    with pytest.raises(ValueError):
        normalize(np.array([[np.nan]]))


def test_resize_preserves_aspect_and_rgb():
    x = resize_image(np.ones((40, 80, 3)))
    assert x.shape == (128, 128)
    assert not x[:32].any() and x[32:96].min() > 0
    assert preprocess(np.zeros((8, 8, 3)))[0].shape == (128, 128)


def test_enhancement():
    x, _ = synthetic_sample("B")
    original, enhanced = fuzzy_enhance(x)
    np.testing.assert_array_equal(original, x)
    assert enhanced.shape == x.shape and 0 <= enhanced.min() <= enhanced.max() <= 1
    assert np.abs(enhanced - x).mean() < 0.15
    assert not fuzzy_enhance(np.zeros((128, 128)))[1].any()


def test_anomaly_fixed_scale():
    x = np.ones((128, 128), np.float32) * 0.4
    error, heat = anomaly_map(x, x - 0.1)
    assert error.shape == heat.shape == x.shape
    np.testing.assert_allclose(heat, 0.1, atol=1e-6)
    assert 0 <= heat.min() <= heat.max() <= 1
    assert not anomaly_map(x, x)[1].any()


def test_threshold_and_components():
    heat = np.zeros((128, 128), np.float32)
    heat[20:35, 20:35] = 0.6
    heat[70:90, 70:90] = 0.9
    heat[2, 2] = 1
    mask, selected = threshold_map(heat, mode="Manual", threshold=0.5)
    regions = localize(mask, heat)
    assert selected == 0.5 and len(regions) == 2
    assert not mask[2, 2]
    assert regions[0]["bbox_xyxy"] == [20, 20, 35, 35]
    assert regions[0]["area_pixels"] > 200
    assert regions[1]["max_anomaly"] == pytest.approx(0.9)
    assert not threshold_map(heat, mode="Manual", threshold=1)[0].any()


def test_empty_mask():
    heat = np.zeros((128, 128), np.float32)
    mask, threshold = threshold_map(heat, np.zeros_like(heat, bool))
    assert not mask.any() and localize(mask, heat) == [] and threshold == 0.1


def test_multimodal_fusion():
    a, b = np.zeros((8, 8)), np.ones((8, 8))
    fused, weights = fuse_maps({"T1": a, "FLAIR": b})
    np.testing.assert_allclose(fused, 0.5)
    assert weights == {"T1": 0.5, "FLAIR": 0.5}
    np.testing.assert_array_equal(fuse_maps({"DWI": b})[0], b)
    np.testing.assert_allclose(fuse_maps({"T1": a, "T2": b}, {"T1": 3, "T2": 1})[0], 0.25)
    for maps, weights in [({}, None), ({"a": a, "b": np.ones((9, 9))}, None), ({"a": a}, {"a": -1})]:
        with pytest.raises(ValueError):
            fuse_maps(maps, weights)


def test_vae_forward_backward_and_determinism():
    torch.set_num_threads(2)
    model = ConvVAE()
    x = torch.rand(2, 1, 128, 128)
    output, mu, logvar = model(x)
    assert output.shape == x.shape and mu.shape == logvar.shape == (2, 32)
    assert output.min() >= 0 and output.max() <= 1
    loss, mse, kl = vae_loss(output, x, mu, logvar)
    loss.backward()
    assert torch.isfinite(loss) and model.mu.weight.grad is not None
    assert mse > 0 and kl >= 0
    model.eval()
    with torch.no_grad():
        np.testing.assert_array_equal(model(x)[0].numpy(), model(x)[0].numpy())


@pytest.mark.parametrize("sample,expected", [("A", 0), ("B", 1), ("C", 2)])
def test_synthetic_behavior(bundle, sample, expected):
    image, injection = synthetic_sample(sample)
    result = analyze_image(image, bundle)
    assert result["demo"] and len(result["regions"]) == expected
    assert result["heat"].shape == (128, 128)
    if injection.any():
        assert (result["mask"] & injection).sum() / injection.sum() > 0.7
        assert result["heat"][injection].mean() > 3 * result["heat"][~injection].mean()


def test_single_and_multimodal(bundle):
    image, _ = synthetic_sample("C")
    single = analyze_image(image, bundle)
    one = analyze_multimodal({"DWI": image}, bundle)
    multiple = analyze_multimodal({"T1": image, "T2": image, "FLAIR": image}, bundle)
    np.testing.assert_allclose(single["heat"], one["heat"])
    np.testing.assert_allclose(single["heat"], multiple["heat"], atol=1e-7)
    assert len(multiple["regions"]) == 2
    with pytest.raises(ValueError):
        analyze_multimodal({}, bundle)


def test_mirrored_candidate_and_different_input_dimensions(bundle):
    image, injection = synthetic_sample("B")
    result = analyze_image(image[:, ::-1].copy(), bundle)
    assert len(result["regions"]) == 1
    assert (result["mask"] & injection[:, ::-1]).sum() / injection.sum() > 0.7
    different = analyze_multimodal({"T1": image, "T2": np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)}, bundle)
    assert different["heat"].shape == (128, 128)
    assert len(different["regions"]) == 1


def test_missing_corrupt_checkpoint(bundle, tmp_path):
    assert bundle.demo and bundle.vae is None
    bad = tmp_path / "corrupt.pth"
    bad.write_bytes(b"not a checkpoint")
    assert load_models(bad).demo


def test_pretrained_download_failure(monkeypatch):
    from torchvision.models import ResNet18_Weights
    def fail(*args, **kwargs):
        raise OSError("Simulated offline connection")
    monkeypatch.setattr(ResNet18_Weights, "get_state_dict", fail)
    with pytest.warns(RuntimeWarning, match="unavailable"):
        model = FeatureExtractor(allow_download=True)
    with torch.no_grad():
        features = model(torch.zeros(1, 1, 128, 128))
    assert not model.pretrained and features.shape == (1, 512)


def test_invalid_image_and_rgb():
    for data, name in [(b"bad image", "scan.png"), (b"", "scan.jpg"), (b"bad", "scan.nii.gz"), (b"abc", "scan.txt")]:
        with pytest.raises(ValueError):
            read_image(data, name)
    buffer = BytesIO()
    Image.new("RGB", (64, 80), "red").save(buffer, "JPEG")
    array, metadata = read_image(buffer.getvalue(), "rgb.jpg")
    assert array.shape == (80, 64) and metadata["kind"] == "Raster"


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_nifti_slices_and_4d(tmp_path, suffix):
    volume = np.arange(12 * 14 * 5, dtype=np.float32).reshape(12, 14, 5)
    path = tmp_path / ("volume" + suffix)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
    data, metadata = read_image(path.read_bytes(), path.name)
    assert data.shape == volume.shape and metadata["kind"] == "NIfTI"
    np.testing.assert_array_equal(select_slice(data, 2), np.rot90(volume[:, :, 2]))
    with pytest.raises(ValueError):
        select_slice(data, 100)
    nib.save(nib.Nifti1Image(np.zeros((12, 14, 5, 2)), np.eye(4)), path)
    with pytest.raises(ValueError, match="Multi-volume"):
        read_image(path.read_bytes(), path.name)
    nib.save(nib.Nifti1Image(np.zeros((12, 14)), np.eye(4)), path)
    assert read_image(path.read_bytes(), path.name)[0].shape == (14, 12)


def test_training_cli_and_checkpoint(tmp_path):
    data = tmp_path / "train"
    validation = tmp_path / "validation"
    data.mkdir()
    validation.mkdir()
    for i in range(2):
        image, _ = synthetic_sample("A", seed=i)
        (data / f"reference_{i}.png").write_bytes(png_bytes(image))
    (validation / "held_out.png").write_bytes(png_bytes(synthetic_sample("A", seed=20)[0]))
    assert HealthyMRIDataset(data)[0].shape == (1, 128, 128)
    output = tmp_path / "smoke_only.pth"
    completed = subprocess.run([sys.executable, "training/train_vae.py", "--data", str(data), "--val-data", str(validation), "--epochs", "1", "--batch-size", "2", "--offline", "--device", "cpu", "--output", str(output), "--data-description", "Synthetic software smoke test; not MRI training"], capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    model = load_models(output)
    assert not model.demo and model.metadata["epochs_trained"] == 1
    assert model.metadata["validation_loss"] is not None
    result = analyze_image(synthetic_sample("B")[0], model)
    assert np.isfinite(result["reconstruction"]).all()


def test_spatial_vae_architecture():
    from models.vae import SpatialVAE, PlainAutoencoder, PlainVAE, MultiChannelVAE
    # Spatial VAE with 16x16 spatial latent map
    model = SpatialVAE(in_channels=1, input_size=128, use_resnet=True)
    x = torch.rand(2, 1, 128, 128)
    features = torch.rand(2, 512)
    recon, mu, logvar = model(x, features)
    assert recon.shape == (2, 1, 128, 128)
    assert mu.shape == (2, 8, 16, 16)
    assert logvar.shape == (2, 8, 16, 16)

    # Spatial VAE at 256x256
    model256 = SpatialVAE(in_channels=1, input_size=256, use_resnet=False)
    x256 = torch.rand(1, 1, 256, 256)
    recon256, mu256, _ = model256(x256)
    assert recon256.shape == (1, 1, 256, 256)
    assert mu256.shape == (1, 8, 16, 16)

    # MultiChannel VAE
    m_model = MultiChannelVAE(in_channels=3, input_size=128, use_resnet=False)
    x_multi = torch.rand(1, 3, 128, 128)
    recon_m, _, _ = m_model(x_multi)
    assert recon_m.shape == (1, 3, 128, 128)

    # Plain Autoencoder
    ae = PlainAutoencoder(in_channels=1, input_size=128)
    ae_recon, ae_z, _ = ae(x)
    assert ae_recon.shape == (2, 1, 128, 128)
    assert ae_z.shape == (2, 8, 16, 16)


def test_real_brain_mask_and_percentile_norm():
    from processing.preprocessing import extract_brain_mask
    img, _ = synthetic_sample("A")
    mask = extract_brain_mask(img)
    assert mask.any()
    assert mask.sum() > 200
    # Brain mask should exclude corner background
    assert not mask[0, 0] and not mask[0, -1]

    # Percentile normalization
    normed = normalize(img, mask=mask, method="percentile")
    assert 0 <= normed.min() <= normed.max() <= 1
    assert normed[~mask].sum() == 0

    # Z-score normalization
    z_normed = normalize(img, mask=mask, method="zscore")
    assert 0 <= z_normed.min() <= z_normed.max() <= 1


def test_threshold_calibration():
    from processing.anomaly import calibrate_threshold
    err1 = np.ones((128, 128), np.float32) * 0.05
    err2 = np.ones((128, 128), np.float32) * 0.08
    thresh = calibrate_threshold([err1, err2], percentile=99.0)
    assert 0.05 <= thresh <= 0.09


def test_evaluation_metrics_and_summary():
    from evaluation.metrics import (compute_dice, compute_iou, compute_pixel_auroc_and_auprc,
                                    compute_lesion_metrics, aggregate_metrics_table)
    true_mask = np.zeros((128, 128), bool)
    true_mask[30:50, 30:50] = True  # Area 400 pixels (medium lesion)

    pred_mask = np.zeros((128, 128), bool)
    pred_mask[35:55, 35:55] = True  # Overlapping prediction

    # Dice & IoU
    dice = compute_dice(pred_mask, true_mask)
    iou = compute_iou(pred_mask, true_mask)
    assert 0.4 < dice < 0.8
    assert 0.2 < iou < 0.6

    # Pixel AUROC & AUPRC
    heat = pred_mask.astype(np.float32) * 0.8
    auroc, auprc = compute_pixel_auroc_and_auprc(heat, true_mask)
    assert auroc > 0.75
    assert auprc > 0.35

    # Lesion metrics
    lesion_res = compute_lesion_metrics(pred_mask, true_mask)
    assert lesion_res["sensitivities"]["all"] == 1.0
    assert lesion_res["sensitivities"]["medium"] == 1.0
    assert lesion_res["false_positives"] == 0

    # Summary table
    records = [{
        "dice": dice, "iou": iou, "pixel_auroc": auroc, "pixel_auprc": auprc,
        "lesion_sensitivities": lesion_res["sensitivities"],
        "false_positives": 0, "image_score": 0.8, "has_lesion": True
    }]
    summary = aggregate_metrics_table(records)
    assert len(summary) >= 7
    assert any(r["Metric"] == "Pixel Dice" for r in summary)
    assert any(r["Subset"] == "Small (<100 px)" for r in summary)


def test_multislice_dataset_and_patient_split(tmp_path):
    from training.dataset import split_patients
    train_dir = tmp_path / "patients"
    train_dir.mkdir()
    # Create volumes for patient_01 and patient_02
    for p_idx in (1, 2):
        volume = np.zeros((32, 32, 10), dtype=np.float32)
        # Brain tissue in slices 2..7 (extent 2 to 7)
        for z in range(2, 8):
            volume[8:24, 8:24, z] = 0.5
        path = train_dir / f"patient_0{p_idx}_t1.nii.gz"
        nib.save(nib.Nifti1Image(volume, np.eye(4)), path)

    # Verify patient splitting
    train_p, val_p = split_patients(train_dir, val_ratio=0.5, seed=42)
    assert len(train_p) == 1 and len(val_p) == 1
    assert train_p.isdisjoint(val_p)

    # Multi-slice dataset should load multiple slices from the extent
    ds = HealthyMRIDataset(train_dir, slice_range=(0.3, 0.7), min_brain_pixels=20)
    assert len(ds) >= 2

