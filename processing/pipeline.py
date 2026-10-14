from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from config import (CHECKPOINT, IMAGE_SIZE, LATENT_DIM, FEATURE_DIM, PIPELINE_VERSION,
                    DEFAULT_CALIBRATED_THRESHOLD, DEFAULT_NORM_METHOD)
from models.vae import ConvVAE, SpatialVAE, MultiChannelVAE, PlainAutoencoder, PlainVAE
from models.feature_extractor import FeatureExtractor
from processing.preprocessing import preprocess, extract_brain_mask
from processing.fuzzy_enhancement import fuzzy_enhance
from processing.anomaly import (demo_reconstruction, anomaly_map, foreground_mask,
                                threshold_map, score_error, intensity_threshold_baseline)
from processing.localization import localize
from processing.multimodal import fuse_maps, fuse_multichannel_residuals


@dataclass
class ModelBundle:
    vae: object
    features: FeatureExtractor
    demo: bool
    status: str
    metadata: dict
    calibrated_threshold: float = DEFAULT_CALIBRATED_THRESHOLD
    model_type: str = "spatial_vae"
    image_size: int = IMAGE_SIZE
    use_fuzzy: bool = True
    use_resnet: bool = True


def load_models(checkpoint=CHECKPOINT, allow_download=False, forced_model_type=None, image_size=IMAGE_SIZE):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            cfg = payload.get("config", {})
            model_type = forced_model_type or payload.get("model_type", cfg.get("model_type", "dense_vae"))
            ckpt_size = cfg.get("image_size", image_size)
            use_resnet = payload.get("use_resnet", True)
            calibrated_thresh = payload.get("calibrated_threshold", DEFAULT_CALIBRATED_THRESHOLD)

            # Feature extractor
            if use_resnet and payload.get("feature_state"):
                feature = FeatureExtractor(use_pretrained=False)
                feature.load_state_dict(payload["feature_state"])
                feature.pretrained = bool(payload.get("pretrained_features", False))
                feature.status = ("ImageNet-pretrained ResNet18 loaded from checkpoint."
                                  if feature.pretrained else
                                  "Untrained frozen ResNet18 saved with checkpoint; transfer learning inactive.")
            else:
                feature = FeatureExtractor(allow_download=allow_download).eval() if use_resnet else None

            # Instantiate model architecture
            if model_type == "spatial_vae":
                vae = SpatialVAE(in_channels=1, input_size=ckpt_size, use_resnet=use_resnet)
            elif model_type == "multi_channel_vae":
                vae = MultiChannelVAE(in_channels=3, input_size=ckpt_size, use_resnet=use_resnet)
            elif model_type == "plain_ae":
                vae = PlainAutoencoder(in_channels=1, input_size=ckpt_size)
            elif model_type == "plain_vae":
                vae = PlainVAE(in_channels=1, input_size=ckpt_size)
            else:
                vae = ConvVAE()

            if payload.get("model_state"):
                vae.load_state_dict(payload["model_state"])

            return ModelBundle(
                vae=vae.eval(),
                features=feature,
                demo=False,
                status=f"Trained {model_type} checkpoint loaded",
                metadata={
                    "epochs_trained": payload.get("epochs_trained", 1),
                    "data_description": payload.get("data_description", "Reference data"),
                    "validation_loss": payload.get("validation_loss"),
                    "calibrated_threshold": calibrated_thresh,
                    "clinical_validation": "Not evaluated"
                },
                calibrated_threshold=calibrated_thresh,
                model_type=model_type,
                image_size=ckpt_size,
                use_fuzzy=payload.get("use_fuzzy", True),
                use_resnet=use_resnet
            )
        except Exception as exc:
            status = f"Checkpoint could not be loaded ({type(exc).__name__}: {exc}). DEMO MODE activated."
    else:
        status = "No VAE checkpoint found. Running in high-fidelity DEMO / ABLATION MODE."

    chosen_type = forced_model_type or "spatial_vae"
    features = FeatureExtractor(allow_download=allow_download).eval()
    return ModelBundle(
        vae=None,
        features=features,
        demo=True,
        status=status,
        metadata={"calibrated_threshold": DEFAULT_CALIBRATED_THRESHOLD},
        calibrated_threshold=DEFAULT_CALIBRATED_THRESHOLD,
        model_type=chosen_type,
        image_size=image_size,
        use_fuzzy=True,
        use_resnet=True
    )


def analyze_image(image, bundle, mode="Calibrated", threshold=0.2, min_area=18,
                  norm_method=DEFAULT_NORM_METHOD, use_fuzzy=True, size=None,
                  calibrated_threshold=None):
    img_size = size or bundle.image_size
    calib_thresh = calibrated_threshold if calibrated_threshold is not None else bundle.calibrated_threshold

    original, processed, brain_mask = preprocess(image, size=img_size, norm_method=norm_method, return_mask=True)
    _, enhanced = fuzzy_enhance(processed, enabled=use_fuzzy)

    # Simple non-neural baseline
    if bundle.model_type == "intensity_threshold":
        mask, heat = intensity_threshold_baseline(enhanced, brain_mask, min_area=min_area)
        reconstruction = processed.copy()
        error = np.abs(enhanced - reconstruction)
        regions = localize(mask, heat)
        score, mean_error, band = score_error(heat, brain_mask)
        selected_thresh = float(threshold)
    else:
        x = torch.from_numpy(enhanced)[None, None]
        with torch.no_grad():
            descriptors = bundle.features(x) if (bundle.features and bundle.use_resnet) else None
            if bundle.demo:
                reconstruction = demo_reconstruction(enhanced)
            else:
                out = bundle.vae(x, descriptors) if descriptors is not None else bundle.vae(x)
                recon_tensor = out[0]
                reconstruction = recon_tensor[0, 0].cpu().numpy()

        error, heat = anomaly_map(enhanced, reconstruction, mask=brain_mask)
        mask, selected_thresh = threshold_map(
            heat, brain_mask, mode=mode, threshold=threshold, min_area=min_area,
            calibrated_threshold=calib_thresh
        )
        regions = localize(mask, heat)
        score, mean_error, band = score_error(error, brain_mask)

    feat_norm = float(descriptors.norm()) if ('descriptors' in locals() and descriptors is not None) else 0.0
    feat_status = bundle.features.status if bundle.features else "ResNet feature conditioning disabled (ablation)"
    feat_pre = bundle.features.pretrained if bundle.features else False

    return {
        "original": original, "processed": processed, "enhanced": enhanced,
        "reconstruction": reconstruction, "error": error, "heat": heat, "mask": mask,
        "foreground": brain_mask, "regions": regions, "threshold": selected_thresh,
        "score": score, "mean_error": mean_error, "band": band, "demo": bundle.demo,
        "feature_norm": feat_norm, "feature_status": feat_status,
        "feature_pretrained": feat_pre, "model_status": bundle.status,
        "model_metadata": bundle.metadata, "threshold_mode": mode, "min_area": min_area,
        "calibrated_threshold": calib_thresh, "model_type": bundle.model_type,
        "norm_method": norm_method, "use_fuzzy": use_fuzzy, "image_size": img_size
    }


def analyze_multimodal(images, bundle, mode="Calibrated", threshold=0.2, min_area=18,
                       norm_method=DEFAULT_NORM_METHOD, use_fuzzy=True, size=None,
                       calibrated_threshold=None, fusion_method="weighted", fusion_weights=None):
    if not images:
        raise ValueError("Upload at least one modality.")

    img_size = size or bundle.image_size
    calib_thresh = calibrated_threshold if calibrated_threshold is not None else bundle.calibrated_threshold

    # Multi-channel VAE path (takes T1, T2, FLAIR simultaneously)
    if bundle.model_type == "multi_channel_vae" and not bundle.demo and len(images) >= 2:
        # Preprocess each channel
        channels, preprocessed, originals = [], [], []
        channel_names = [k for k in ("T1", "T2", "FLAIR") if k in images] or list(images.keys())[:3]
        for name in channel_names:
            orig, proc, mask = preprocess(images[name], size=img_size, norm_method=norm_method, return_mask=True)
            _, enh = fuzzy_enhance(proc, enabled=use_fuzzy)
            channels.append(enh)
            preprocessed.append(proc)
            originals.append(orig)

        # Pad to 3 channels if fewer
        while len(channels) < 3:
            channels.append(channels[0])
            channel_names.append(channel_names[0])

        x_multi = torch.from_numpy(np.stack(channels))[None]  # (1, 3, H, W)
        with torch.no_grad():
            recon_multi = bundle.vae(x_multi)[0][0].cpu().numpy()  # (3, H, W)

        raw_errors = np.abs(np.stack(channels) - recon_multi)
        fused_error = fuse_multichannel_residuals(raw_errors, channel_names, weights=fusion_weights)
        fused_orig = np.mean(np.stack(originals), axis=0)
        fused_proc = np.mean(np.stack(preprocessed), axis=0)
        fused_enh = np.mean(np.stack(channels), axis=0)
        fused_recon = np.mean(recon_multi, axis=0)
        brain_fg = extract_brain_mask(fused_proc)

        error, heat = anomaly_map(fused_enh, fused_recon, mask=brain_fg)
        mask, selected_thresh = threshold_map(
            heat, brain_fg, mode=mode, threshold=threshold, min_area=min_area,
            calibrated_threshold=calib_thresh
        )
        regions = localize(mask, heat)
        score, mean_error, band = score_error(error, brain_fg)

        # Compute individual modality views for dashboard
        individual = {}
        for i, name in enumerate(channel_names[:len(images)]):
            i_err, i_heat = anomaly_map(channels[i], recon_multi[i], mask=brain_fg)
            i_mask, _ = threshold_map(i_heat, brain_fg, mode=mode, threshold=threshold, min_area=min_area, calibrated_threshold=calib_thresh)
            individual[name] = {
                "original": originals[i], "processed": preprocessed[i], "enhanced": channels[i],
                "reconstruction": recon_multi[i], "error": i_err, "heat": i_heat, "mask": i_mask,
                "foreground": brain_fg
            }

        return {
            "original": fused_orig, "processed": fused_proc, "enhanced": fused_enh,
            "reconstruction": fused_recon, "error": error, "heat": heat, "mask": mask,
            "foreground": brain_fg, "regions": regions, "threshold": selected_thresh,
            "score": score, "mean_error": mean_error, "band": band, "demo": bundle.demo,
            "feature_norm": 0.0, "feature_status": "Multi-Channel Joint Spatial VAE",
            "feature_pretrained": False, "model_status": bundle.status,
            "model_metadata": bundle.metadata, "threshold_mode": mode, "min_area": min_area,
            "calibrated_threshold": calib_thresh, "model_type": bundle.model_type,
            "modalities": individual, "weights": {n: 1.0/len(channel_names) for n in channel_names},
            "fusion_method": "multi_channel_joint"
        }

    # Standard per-modality analysis followed by calibrated weighted fusion
    individual = {name: analyze_image(image, bundle, mode=mode, threshold=threshold,
                                      min_area=min_area, norm_method=norm_method,
                                      use_fuzzy=use_fuzzy, size=img_size,
                                      calibrated_threshold=calib_thresh)
                  for name, image in images.items()}
    result = dict(next(iter(individual.values())))
    for key in ("original", "processed", "enhanced", "reconstruction", "error", "heat"):
        result[key], weights = fuse_maps({name: r[key] for name, r in individual.items()},
                                         weights=fusion_weights, method=fusion_method)
    result["foreground"] = np.logical_or.reduce([r["foreground"] for r in individual.values()])
    result["mask"], result["threshold"] = threshold_map(
        result["heat"], result["foreground"], mode=mode, threshold=threshold,
        min_area=min_area, calibrated_threshold=calib_thresh
    )
    result["regions"] = localize(result["mask"], result["heat"])
    result["score"], result["mean_error"], result["band"] = score_error(result["error"], result["foreground"])
    result.update(modalities=individual, weights=weights, fusion_method=fusion_method)
    return result
