"""Batch evaluation script for Brain MRI anomaly detection.

Evaluates:
- Pixel-level Dice, IoU, AUROC, AUPRC
- Lesion-level sensitivity and false positives per image (FPPI)
- Image-level AUROC
- Mean +/- std exported to CSV, split by lesion size (Small, Medium, Large)
- Ablation switches and baseline comparisons
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from config import CHECKPOINT, IMAGE_SIZE, DEFAULT_CALIBRATED_THRESHOLD, DEFAULT_NORM_METHOD
from processing.pipeline import load_models, analyze_image, analyze_multimodal, ModelBundle
from evaluation.metrics import (compute_dice, compute_iou, compute_pixel_auroc_and_auprc,
                                compute_lesion_metrics, aggregate_metrics_table)
from utils.image_utils import read_image
from utils.synthetic import synthetic_sample


def generate_benchmark_testset(n_cases=12, image_size=128):
    """Generate reproducible labelled test suite with varying lesion sizes."""
    test_cases = []
    # Sample A: healthy (no lesions)
    for seed in (42, 101, 202, 303):
        img, mask = synthetic_sample("A", seed=seed, size=image_size)
        test_cases.append({"id": f"healthy_s{seed}", "image": img, "mask": mask, "has_lesion": False})

    # Sample B: small / medium lesions
    for seed in (42, 102, 204, 306):
        img, mask = synthetic_sample("B", seed=seed, size=image_size)
        test_cases.append({"id": f"lesion_b_s{seed}", "image": img, "mask": mask, "has_lesion": True})

    # Sample C: multiple lesions (medium + large)
    for seed in (42, 103, 205, 307):
        img, mask = synthetic_sample("C", seed=seed, size=image_size)
        test_cases.append({"id": f"lesion_c_s{seed}", "image": img, "mask": mask, "has_lesion": True})

    return test_cases


def evaluate_case(image, true_mask, bundle, mode="Calibrated", threshold=DEFAULT_CALIBRATED_THRESHOLD,
                  use_fuzzy=True, norm_method=DEFAULT_NORM_METHOD):
    """Run pipeline on one case and compute all metrics."""
    result = analyze_image(
        image, bundle, mode=mode, threshold=threshold, min_area=10,
        norm_method=norm_method, use_fuzzy=use_fuzzy
    )

    pred_mask = result["mask"]
    heat_map = result["heat"]
    brain_mask = result["foreground"]

    # Pixel metrics
    dice = compute_dice(pred_mask, true_mask)
    iou = compute_iou(pred_mask, true_mask)
    auroc, auprc = compute_pixel_auroc_and_auprc(heat_map, true_mask, brain_mask=brain_mask)

    # Lesion metrics
    lesion_res = compute_lesion_metrics(pred_mask, true_mask)

    # Image score
    image_score = float(heat_map[brain_mask].max()) if brain_mask.any() else float(heat_map.max())

    return {
        "dice": dice,
        "iou": iou,
        "pixel_auroc": auroc,
        "pixel_auprc": auprc,
        "lesion_sensitivities": lesion_res["sensitivities"],
        "lesion_counts": lesion_res["counts"],
        "false_positives": lesion_res["false_positives"],
        "num_true_lesions": lesion_res["num_true_lesions"],
        "num_pred_regions": lesion_res["num_pred_regions"],
        "image_score": image_score,
        "has_lesion": bool(true_mask.any()),
        "threshold_used": result["threshold"]
    }


def run_evaluation_suite(test_cases, bundle, mode="Calibrated", threshold=DEFAULT_CALIBRATED_THRESHOLD,
                         use_fuzzy=True, norm_method=DEFAULT_NORM_METHOD):
    """Run evaluation over a test collection and return summary rows and per-case records."""
    records = []
    for case in test_cases:
        res = evaluate_case(
            case["image"], case["mask"], bundle, mode=mode,
            threshold=threshold, use_fuzzy=use_fuzzy, norm_method=norm_method
        )
        res["id"] = case["id"]
        records.append(res)

    summary_rows = aggregate_metrics_table(records)
    return summary_rows, records


def run_ablation_benchmarks(test_cases, checkpoint_path=CHECKPOINT):
    """Run full ablation matrix comparing architectures and component toggles."""
    experiments = [
        {"name": "Spatial VAE (Proposed)", "model_type": "spatial_vae", "use_fuzzy": True, "use_resnet": True},
        {"name": "Dense VAE (Legacy 32-D)", "model_type": "dense_vae", "use_fuzzy": True, "use_resnet": True},
        {"name": "Plain VAE (No ResNet, No Fuzzy)", "model_type": "plain_vae", "use_fuzzy": False, "use_resnet": False},
        {"name": "Plain Autoencoder (Baseline)", "model_type": "plain_ae", "use_fuzzy": False, "use_resnet": False},
        {"name": "Intensity Thresholding (Baseline)", "model_type": "intensity_threshold", "use_fuzzy": False, "use_resnet": False},
        {"name": "Ablation: Fuzzy OFF", "model_type": "spatial_vae", "use_fuzzy": False, "use_resnet": True},
        {"name": "Ablation: ResNet OFF", "model_type": "spatial_vae", "use_fuzzy": True, "use_resnet": False},
    ]

    ablation_results = {}
    for exp in experiments:
        name = exp["name"]
        bundle = load_models(checkpoint=checkpoint_path, forced_model_type=exp["model_type"])
        bundle.use_resnet = exp["use_resnet"]
        bundle.use_fuzzy = exp["use_fuzzy"]
        bundle.model_type = exp["model_type"]

        summary, per_case = run_evaluation_suite(
            test_cases, bundle, mode="Calibrated", use_fuzzy=exp["use_fuzzy"]
        )
        ablation_results[name] = {"summary": summary, "per_case": per_case}

    return ablation_results


def export_csv_report(summary_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Subset", "Mean", "Std", "Mean ± Std", "N_Samples"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved evaluation report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="Directory containing test MRI images")
    parser.add_argument("--masks", type=Path, help="Directory containing corresponding lesion masks")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--model-type", choices=("spatial_vae", "dense_vae", "plain_vae", "plain_ae", "intensity_threshold"), default="spatial_vae")
    parser.add_argument("--output-csv", type=Path, default=Path("evaluation_results.csv"))
    parser.add_argument("--ablation", action="store_true", help="Run full ablation comparison against baselines")
    parser.add_argument("--threshold-mode", choices=("Calibrated", "Automatic", "Manual"), default="Calibrated")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CALIBRATED_THRESHOLD)
    parser.add_argument("--norm-method", choices=("percentile", "zscore", "minmax"), default="percentile")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy enhancement")
    args = parser.parse_args()

    # Load test cases
    if args.images and args.images.is_dir():
        image_files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".nii", ".nii.gz"))
        test_cases = []
        for img_path in image_files:
            img_arr, _ = read_image(img_path.read_bytes(), img_path.name)
            # Find matching mask
            mask_arr = np.zeros_like(img_arr, dtype=bool)
            if args.masks and args.masks.is_dir():
                mask_candidates = list(args.masks.glob(f"{img_path.stem}*"))
                if mask_candidates:
                    m_data, _ = read_image(mask_candidates[0].read_bytes(), mask_candidates[0].name)
                    mask_arr = m_data > 0.5
            test_cases.append({"id": img_path.name, "image": img_arr, "mask": mask_arr, "has_lesion": bool(mask_arr.any())})
        print(f"Loaded {len(test_cases)} user test cases from {args.images}")
    else:
        print("Using reproducible synthetic benchmark test suite...")
        test_cases = generate_benchmark_testset(n_cases=12)

    if args.ablation:
        print("\n================== RUNNING ABLATION BENCHMARK ==================")
        ablation_res = run_ablation_benchmarks(test_cases, checkpoint_path=args.checkpoint)
        print(f"{'Experiment':<35} | {'Pixel Dice':<16} | {'Lesion Sens':<16} | {'FPPI':<10} | {'Img AUROC':<10}")
        print("-" * 95)
        for name, data in ablation_res.items():
            s_dict = {row["Metric"] + "_" + row["Subset"]: row["Mean ± Std"] for row in data["summary"]}
            dice = s_dict.get("Pixel Dice_All Lesions", "N/A")
            sens = s_dict.get("Lesion Sensitivity_All Lesions", "N/A")
            fppi = s_dict.get("False Positives Per Image (FPPI)_Per Image", "N/A")
            img_auc = s_dict.get("Image-level AUROC_Dataset Level", "N/A")
            print(f"{name:<35} | {dice:<16} | {sens:<16} | {fppi:<10} | {img_auc:<10}")

        # Export ablation summary CSV
        ablation_csv = args.output_csv.with_name("ablation_comparison.csv")
        with open(ablation_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Experiment", "Pixel Dice (mean ± std)", "Pixel IoU", "Lesion Sens (All)",
                             "Lesion Sens (Small)", "Lesion Sens (Med)", "Lesion Sens (Large)",
                             "FPPI", "Image AUROC"])
            for name, data in ablation_res.items():
                s_dict = {row["Metric"] + "_" + row["Subset"]: row["Mean ± Std"] for row in data["summary"]}
                writer.writerow([
                    name,
                    s_dict.get("Pixel Dice_All Lesions", "N/A"),
                    s_dict.get("Pixel IoU_All Lesions", "N/A"),
                    s_dict.get("Lesion Sensitivity_All Lesions", "N/A"),
                    s_dict.get("Lesion Sensitivity_Small (<100 px)", "N/A"),
                    s_dict.get("Lesion Sensitivity_Medium (100-400 px)", "N/A"),
                    s_dict.get("Lesion Sensitivity_Large (>400 px)", "N/A"),
                    s_dict.get("False Positives Per Image (FPPI)_Per Image", "N/A"),
                    s_dict.get("Image-level AUROC_Dataset Level", "N/A")
                ])
        print(f"\nSaved ablation comparison: {ablation_csv}")
    else:
        bundle = load_models(checkpoint=args.checkpoint, forced_model_type=args.model_type)
        summary, per_case = run_evaluation_suite(
            test_cases, bundle, mode=args.threshold_mode, threshold=args.threshold,
            use_fuzzy=not args.no_fuzzy, norm_method=args.norm_method
        )
        print("\n================== EVALUATION RESULTS SUMMARY ==================")
        print(f"{'Metric':<35} | {'Subset':<25} | {'Mean ± Std':<18} | {'N':<6}")
        print("-" * 90)
        for row in summary:
            print(f"{row['Metric']:<35} | {row['Subset']:<25} | {row['Mean ± Std']:<18} | {row['N_Samples']:<6}")

        export_csv_report(summary, args.output_csv)


if __name__ == "__main__":
    main()
