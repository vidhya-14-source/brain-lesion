"""Train VAE or baseline models on multi-slice healthy MRI data; no clinical claims."""
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import (CHECKPOINT, IMAGE_SIZE, LATENT_DIM, FEATURE_DIM, PIPELINE_VERSION,
                    SPATIAL_LATENT_CHANNELS, DEFAULT_CALIBRATED_PERCENTILE)
from models.vae import ConvVAE, SpatialVAE, MultiChannelVAE, PlainAutoencoder, PlainVAE, vae_loss, ae_loss
from models.feature_extractor import FeatureExtractor
from training.dataset import HealthyMRIDataset, split_patients
from processing.anomaly import calibrate_threshold, foreground_mask


def create_model(model_type, image_size=IMAGE_SIZE, use_resnet=True):
    if model_type == "spatial_vae":
        return SpatialVAE(in_channels=1, input_size=image_size, use_resnet=use_resnet)
    elif model_type == "dense_vae":
        return ConvVAE(latent_dim=LATENT_DIM, feature_dim=FEATURE_DIM)
    elif model_type == "plain_vae":
        return PlainVAE(in_channels=1, input_size=image_size)
    elif model_type == "plain_ae":
        return PlainAutoencoder(in_channels=1, input_size=image_size)
    elif model_type == "multi_channel_vae":
        return MultiChannelVAE(in_channels=3, input_size=image_size, use_resnet=use_resnet)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.beta < 0 or args.lr <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive; beta must be nonnegative.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)

    # Dataset creation with multi-slice coverage and patient splitting
    if args.val_data:
        train_data = HealthyMRIDataset(args.data, image_size=args.image_size, use_fuzzy=not args.no_fuzzy)
        val_data = HealthyMRIDataset(args.val_data, image_size=args.image_size, use_fuzzy=not args.no_fuzzy)
        if set(p.resolve() for p in train_data.files) & set(p.resolve() for p in val_data.files):
            raise ValueError("Training and validation folders overlap. Split by subject into separate folders.")
    elif args.split_patients:
        train_pids, val_pids = split_patients(args.data, val_ratio=0.2, seed=args.seed)
        train_data = HealthyMRIDataset(args.data, image_size=args.image_size, use_fuzzy=not args.no_fuzzy, patient_whitelist=train_pids)
        val_data = HealthyMRIDataset(args.data, image_size=args.image_size, use_fuzzy=not args.no_fuzzy, patient_whitelist=val_pids)
    else:
        train_data = HealthyMRIDataset(args.data, image_size=args.image_size, use_fuzzy=not args.no_fuzzy)
        val_data = None

    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0) if val_data else None

    use_resnet = not args.no_resnet and args.model_type not in ("plain_ae", "plain_vae")
    feature = FeatureExtractor(allow_download=args.pretrained, use_pretrained=not args.offline).to(device).eval() if use_resnet else None
    model = create_model(args.model_type, image_size=args.image_size, use_resnet=use_resnet).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best, history = float("inf"), []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model: {args.model_type} ({args.image_size}x{args.image_size}); Device: {device}", flush=True)
    print(f"Training slices: {len(train_data)}; Validation slices: {len(val_data) if val_data else 0}", flush=True)
    if feature:
        print(feature.status, flush=True)

    is_ae = (args.model_type == "plain_ae")

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3)
        for x in loader:
            x = x.to(device)
            descriptors = feature(x) if feature else None
            reconstruction, mu, logvar = model(x, descriptors) if feature else model(x)
            if is_ae:
                loss, mse, kl = ae_loss(reconstruction, x)
            else:
                loss, mse, kl = vae_loss(reconstruction, x, mu, logvar, args.beta)

            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; inspect data and learning rate.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += np.array([loss.item(), mse.item(), kl.item()]) * len(x)
        totals /= len(train_data)

        validation = None
        calibrated_thresh = 0.12
        if val_loader:
            model.eval()
            total = 0.0
            val_errors = []
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    desc = feature(x) if feature else None
                    recon, mu, logvar = model(x, desc) if feature else model(x)
                    l = ae_loss(recon, x)[0] if is_ae else vae_loss(recon, x, mu, logvar, args.beta)[0]
                    total += l.item() * len(x)
                    # Record reconstruction error for threshold calibration
                    err = torch.abs(x - recon).squeeze(1).cpu().numpy()
                    val_errors.extend([err[i] for i in range(len(err))])
            validation = total / len(val_data)
            # Calibrate threshold on healthy validation error distribution
            calibrated_thresh = calibrate_threshold(val_errors, percentile=DEFAULT_CALIBRATED_PERCENTILE)

        criterion = validation if validation is not None else float(totals[0])
        record = {"epoch": epoch, "loss": float(totals[0]), "reconstruction_loss": float(totals[1]),
                  "kl": float(totals[2]), "validation_loss": validation, "calibrated_threshold": calibrated_thresh}
        history.append(record)
        print(json.dumps(record), flush=True)

        if criterion < best:
            best = criterion
            payload = {
                "model_state": model.state_dict(),
                "model_type": args.model_type,
                "feature_state": feature.state_dict() if feature else None,
                "pretrained_features": feature.pretrained if feature else False,
                "use_resnet": use_resnet,
                "use_fuzzy": not args.no_fuzzy,
                "image_size": args.image_size,
                "epochs_trained": epoch,
                "config": {
                    "image_size": args.image_size, "latent_dim": LATENT_DIM,
                    "spatial_latent_channels": SPATIAL_LATENT_CHANNELS,
                    "feature_dim": FEATURE_DIM, "pipeline_version": PIPELINE_VERSION,
                    "model_type": args.model_type
                },
                "calibrated_threshold": calibrated_thresh,
                "validation_loss": validation, "training_loss": float(totals[0]),
                "data_description": args.data_description, "seed": args.seed, "beta": args.beta
            }
            temporary = output.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(output)
            print(f"Saved best checkpoint: {output} (Calibrated threshold: {calibrated_thresh:.4f})", flush=True)
        output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--split-patients", action="store_true", help="Split single dataset folder by patient ID (80/20)")
    parser.add_argument("--model-type", choices=("spatial_vae", "dense_vae", "plain_vae", "plain_ae", "multi_channel_vae"), default="spatial_vae")
    parser.add_argument("--image-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy enhancement (ablation)")
    parser.add_argument("--no-resnet", action="store_true", help="Disable ResNet visual conditioning (ablation)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretrained", action="store_true", help="Allow ImageNet weight download; otherwise use cached weights if available.")
    group.add_argument("--offline", action="store_true", help="Use deterministic untrained backbone even if cached weights exist.")
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    parser.add_argument("--data-description", default="User-supplied reference MRI data; health status and clinical performance not verified.")
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
