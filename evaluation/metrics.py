import numpy as np
from skimage.measure import label, regionprops
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def compute_dice(pred_mask, true_mask, eps=1e-7):
    """Compute pixel-level Dice coefficient."""
    p = np.asarray(pred_mask, dtype=bool)
    t = np.asarray(true_mask, dtype=bool)
    intersection = np.logical_and(p, t).sum()
    total = p.sum() + t.sum()
    if total == 0:
        return 1.0  # Perfect match on healthy slice
    return float((2.0 * intersection + eps) / (total + eps))


def compute_iou(pred_mask, true_mask, eps=1e-7):
    """Compute pixel-level Intersection over Union (Jaccard index)."""
    p = np.asarray(pred_mask, dtype=bool)
    t = np.asarray(true_mask, dtype=bool)
    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def compute_pixel_auroc_and_auprc(heat_map, true_mask, brain_mask=None):
    """Compute pixel-level AUROC and AUPRC over brain pixels."""
    h = np.asarray(heat_map, dtype=np.float32)
    t = np.asarray(true_mask, dtype=bool)
    if brain_mask is not None:
        bg = np.asarray(brain_mask, dtype=bool)
        h_vals = h[bg]
        t_vals = t[bg]
    else:
        h_vals = h.ravel()
        t_vals = t.ravel()

    # Need at least one positive and one negative pixel
    n_pos = int(t_vals.sum())
    n_neg = int(t_vals.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan

    try:
        auroc = float(roc_auc_score(t_vals, h_vals))
    except Exception:
        auroc = np.nan

    try:
        precision, recall, _ = precision_recall_curve(t_vals, h_vals)
        auprc = float(auc(recall, precision))
    except Exception:
        auprc = np.nan

    return auroc, auprc


def categorize_lesion_size(area, small_thresh=100, medium_thresh=400):
    if area < small_thresh:
        return "small"
    elif area <= medium_thresh:
        return "medium"
    return "large"


def compute_lesion_metrics(pred_mask, true_mask, min_overlap=0.1,
                           small_thresh=100, medium_thresh=400):
    """Compute lesion-level sensitivity (overall and split by size) and false positives.
    
    A ground truth lesion is detected if its overlap with predicted mask >= min_overlap.
    A predicted candidate region is a False Positive if it does not overlap with any true lesion.
    """
    p = np.asarray(pred_mask, dtype=bool)
    t = np.asarray(true_mask, dtype=bool)

    # Connected components of ground truth lesions
    labeled_true, num_true = label(t, return_num=True, connectivity=2)
    true_props = regionprops(labeled_true)

    # Connected components of predicted candidate regions
    labeled_pred, num_pred = label(p, return_num=True, connectivity=2)
    pred_props = regionprops(labeled_pred)

    counts = {
        "all": {"detected": 0, "total": 0},
        "small": {"detected": 0, "total": 0},
        "medium": {"detected": 0, "total": 0},
        "large": {"detected": 0, "total": 0}
    }

    # Evaluate ground truth lesions
    for prop in true_props:
        area = prop.area
        category = categorize_lesion_size(area, small_thresh, medium_thresh)
        lesion_mask = (labeled_true == prop.label)
        overlap = np.logical_and(lesion_mask, p).sum() / float(area)
        detected = (overlap >= min_overlap) or (overlap > 0.0 and area < small_thresh)

        counts["all"]["total"] += 1
        counts[category]["total"] += 1
        if detected:
            counts["all"]["detected"] += 1
            counts[category]["detected"] += 1

    # Count false positive predicted regions
    false_positives = 0
    for prop in pred_props:
        pred_region = (labeled_pred == prop.label)
        if not np.logical_and(pred_region, t).any():
            false_positives += 1

    # Compute sensitivities
    sensitivities = {}
    for cat, d in counts.items():
        sensitivities[cat] = (d["detected"] / d["total"]) if d["total"] > 0 else np.nan

    return {
        "sensitivities": sensitivities,
        "counts": counts,
        "false_positives": false_positives,
        "num_true_lesions": num_true,
        "num_pred_regions": num_pred
    }


def aggregate_metrics_table(eval_records):
    """Compute mean ± std for all metrics overall and split by lesion size.
    
    Returns a structured dictionary and rows suitable for CSV export.
    """
    if not eval_records:
        return []

    dices = [r["dice"] for r in eval_records if np.isfinite(r.get("dice", np.nan))]
    ious = [r["iou"] for r in eval_records if np.isfinite(r.get("iou", np.nan))]
    aurocs = [r["pixel_auroc"] for r in eval_records if np.isfinite(r.get("pixel_auroc", np.nan))]
    auprcs = [r["pixel_auprc"] for r in eval_records if np.isfinite(r.get("pixel_auprc", np.nan))]
    fppis = [r["false_positives"] for r in eval_records]

    # Image-level AUROC
    image_scores = [r["image_score"] for r in eval_records]
    image_labels = [int(r["has_lesion"]) for r in eval_records]
    if len(set(image_labels)) > 1:
        try:
            img_auroc = float(roc_auc_score(image_labels, image_scores))
        except Exception:
            img_auroc = np.nan
    else:
        img_auroc = np.nan

    def stats(arr):
        if not arr:
            return 0.0, 0.0, "N/A"
        m = float(np.mean(arr))
        s = float(np.std(arr))
        return m, s, f"{m:.4f} ± {s:.4f}"

    rows = []

    # Pixel-level metrics
    m, s, text = stats(dices)
    rows.append({"Metric": "Pixel Dice", "Subset": "All Lesions", "Mean": m, "Std": s, "Mean ± Std": text, "N_Samples": len(dices)})

    m, s, text = stats(ious)
    rows.append({"Metric": "Pixel IoU", "Subset": "All Lesions", "Mean": m, "Std": s, "Mean ± Std": text, "N_Samples": len(ious)})

    m, s, text = stats(aurocs)
    rows.append({"Metric": "Pixel AUROC", "Subset": "All Lesions", "Mean": m, "Std": s, "Mean ± Std": text, "N_Samples": len(aurocs)})

    m, s, text = stats(auprcs)
    rows.append({"Metric": "Pixel AUPRC", "Subset": "All Lesions", "Mean": m, "Std": s, "Mean ± Std": text, "N_Samples": len(auprcs)})

    # Lesion-level Sensitivity split by size
    for cat, label_str in [("all", "All Lesions"), ("small", "Small (<100 px)"),
                           ("medium", "Medium (100-400 px)"), ("large", "Large (>400 px)")]:
        sens_vals = [r["lesion_sensitivities"][cat] for r in eval_records
                     if np.isfinite(r["lesion_sensitivities"].get(cat, np.nan))]
        m, s, text = stats(sens_vals)
        rows.append({"Metric": "Lesion Sensitivity", "Subset": label_str, "Mean": m, "Std": s, "Mean ± Std": text, "N_Samples": len(sens_vals)})

    # False Positives Per Image
    m, s, text = stats(fppis)
    rows.append({"Metric": "False Positives Per Image (FPPI)", "Subset": "Per Image", "Mean": m, "Std": s, "Mean ± Std": text, "N_Samples": len(fppis)})

    # Image-level AUROC
    rows.append({"Metric": "Image-level AUROC", "Subset": "Dataset Level", "Mean": img_auroc, "Std": 0.0, "Mean ± Std": f"{img_auroc:.4f}" if np.isfinite(img_auroc) else "N/A", "N_Samples": len(eval_records)})

    return rows
