from io import BytesIO
import json
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from config import (CHECKPOINT, MODALITIES, DISCLAIMER, DEMO_WARNING, DEMO_OUTPUT,
                    DEFAULT_CALIBRATED_THRESHOLD, DEFAULT_NORM_METHOD,
                    DEFAULT_MODALITY_WEIGHTS, DEFAULT_CALIBRATED_PERCENTILE)
from processing.pipeline import load_models, analyze_image, analyze_multimodal
from processing.anomaly import foreground_mask
from utils.image_utils import read_image, select_slice, png_bytes
from utils.synthetic import synthetic_sample
from utils.visualization import heatmap_rgb, overlay
from evaluation.evaluate import generate_benchmark_testset, run_evaluation_suite, run_ablation_benchmarks
from evaluation.metrics import compute_dice, compute_iou, compute_lesion_metrics

st.set_page_config(page_title="Brain MRI · Research Studio", page_icon="🧠", layout="wide")
st.markdown("""<style>
.block-container {max-width:1440px;padding-top:1.8rem;padding-bottom:3rem}
h1 {font-size:2.2rem!important;letter-spacing:-.045em;line-height:1.15!important}
h2,h3 {letter-spacing:-.025em}
.eyebrow {color:#55dfbf;letter-spacing:.18em;font-size:.75rem;font-weight:700}
.subtitle {color:#99adc7;font-size:1.05rem;margin-bottom:1.5rem}
[data-testid="stMetric"] {background:#141f32;border:1px solid #26364e;border-radius:14px;padding:16px}
[data-testid="stMetricLabel"] {color:#a9bdd5}
[data-testid="stMetricValue"] {color:#72e4cb}
.stTabs [data-baseweb="tab-list"] {gap:18px;border-bottom:1px solid #29364b}
.stTabs [data-baseweb="tab"] {height:50px}
[data-testid="stImage"] {border-radius:10px;overflow:hidden}
div.stButton>button[kind="primary"] {font-weight:700;border-radius:9px}
.badge-pill {display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;margin-right:6px}
.badge-green {background:#14382c;color:#55dfbf;border:1px solid #256650}
.badge-blue {background:#132840;color:#6bb2ff;border:1px solid #244b78}
.badge-amber {background:#3b2a12;color:#f5b041;border:1px solid #6b4c20}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading research models…")
def cached_models(signature, allow_download, forced_model_type, image_size):
    return load_models(checkpoint=signature[0], allow_download=allow_download,
                       forced_model_type=forced_model_type, image_size=image_size)


@st.cache_data(show_spinner=False, max_entries=8)
def decode_upload(data, name):
    return read_image(data, name)


def upload_input(label, key):
    upload = st.file_uploader(label, type=["png", "jpg", "jpeg", "nii", "gz"], key=key)
    if upload is None:
        return None, None, False
    try:
        array, metadata = decode_upload(upload.getvalue(), upload.name)
        if array.ndim == 3:
            count = array.shape[2]
            index = st.slider("Axial slice (zero-based)", 0, count - 1, count // 2, key=key + "_slice") if count > 1 else 0
            metadata = dict(metadata, slice_index=index)
            array = select_slice(array, index)
        st.image(array / max(float(np.max(np.abs(array))), 1e-8) if array.min() >= 0 else (array - array.min()) / max(float(np.ptp(array)), 1e-8), caption=upload.name, clamp=True, width=220)
        with st.expander("Input details"):
            st.json(metadata)
        return array, dict(metadata, filename=upload.name), False
    except ValueError as exc:
        st.error(str(exc))
        return None, None, True


def execute(images, provenance, multi=False, gt_mask=None):
    try:
        with st.status("Processing MRI research pipeline…", expanded=True) as status:
            st.write(f"Normalizing with {norm_method} method and extracting anatomical brain mask.")
            if use_fuzzy:
                st.write("Applying bounded fuzzy contrast enhancement.")
            else:
                st.write("Fuzzy enhancement disabled (ablation switch).")

            if multi:
                result = analyze_multimodal(
                    images, bundle, mode=mode, threshold=threshold, min_area=min_area,
                    norm_method=norm_method, use_fuzzy=use_fuzzy, size=img_size,
                    calibrated_threshold=calibrated_thresh, fusion_method=fusion_method
                )
            else:
                result = analyze_image(
                    images, bundle, mode=mode, threshold=threshold, min_area=min_area,
                    norm_method=norm_method, use_fuzzy=use_fuzzy, size=img_size,
                    calibrated_threshold=calibrated_thresh
                )

            st.write("Reconstruction, anomaly mapping, calibrated thresholding, and localization complete.")
            result["provenance"] = provenance
            if gt_mask is not None:
                result["ground_truth_mask"] = gt_mask
                # Compute live metrics
                result["live_dice"] = compute_dice(result["mask"], gt_mask)
                result["live_iou"] = compute_iou(result["mask"], gt_mask)
                result["live_lesion"] = compute_lesion_metrics(result["mask"], gt_mask)

            st.session_state.result = result
            status.update(label="Analysis complete · open the Results tab", state="complete", expanded=False)
    except Exception as exc:
        st.session_state.pop("result", None)
        st.error(f"Analysis could not complete: {exc}. Check input and configuration.")


def result_archive(result):
    fields = ("score", "band", "mean_error", "threshold", "threshold_mode", "min_area",
              "regions", "demo", "provenance", "feature_status", "feature_pretrained",
              "model_status", "model_metadata", "model_type", "norm_method", "use_fuzzy")
    report = {key: result.get(key) for key in fields if key in result}
    report.update(
        disclaimer=DISCLAIMER,
        score_calibration="Calibrated research score against healthy validation distribution",
        clinical_metrics="Research evaluation only"
    )
    if "weights" in result:
        report["fusion_weights"] = result["weights"]
    if "live_dice" in result:
        report["evaluation"] = {
            "dice": result["live_dice"],
            "iou": result["live_iou"],
            "lesion_sensitivities": result["live_lesion"]["sensitivities"],
            "false_positives": result["live_lesion"]["false_positives"]
        }
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("research_report.json", json.dumps(report, indent=2))
        for key in ("original", "processed", "enhanced", "reconstruction", "error", "heat", "mask"):
            if key in result:
                archive.writestr(key + ".png", png_bytes(result[key]))
        archive.writestr("overlay.png", png_bytes(overlay(result["original"], result["heat"], result["mask"], result["regions"])))
        numerical = BytesIO()
        np.savez_compressed(numerical, **{k: result[k] for k in ("error", "heat", "mask") if k in result})
        archive.writestr("arrays.npz", numerical.getvalue())
        for name, individual in result.get("modalities", {}).items():
            archive.writestr(f"{name}_anomaly.png", png_bytes(individual["heat"]))
    return output.getvalue()


st.markdown('<div class="eyebrow">NEUROIMAGING RESEARCH STUDIO · MULTI-MODAL BRAIN LESION DETECTOR</div>', unsafe_allow_html=True)
st.title("Spatial VAE Multi-Modal Brain Lesion Detector")
st.markdown('<div class="subtitle">Fully Convolutional Spatial Latent Anomaly Detection, Contrast Calibration, and Ablation Studio</div>', unsafe_allow_html=True)
st.info(DISCLAIMER)

with st.sidebar:
    st.markdown("### Model & Ablation Controls")

    # Architecture selection
    model_type_label = st.selectbox(
        "Model Architecture",
        [
            "Spatial VAE (Fully Conv 16x16 Latent)",
            "Dense VAE (Legacy 32-D Bottleneck)",
            "Plain VAE (No ResNet, No Fuzzy)",
            "Plain Autoencoder (Baseline)",
            "Intensity Thresholding (Baseline)"
        ],
        index=0,
        help="Spatial VAE localizes anomalies without corrupting global latents."
    )
    model_type_map = {
        "Spatial VAE (Fully Conv 16x16 Latent)": "spatial_vae",
        "Dense VAE (Legacy 32-D Bottleneck)": "dense_vae",
        "Plain VAE (No ResNet, No Fuzzy)": "plain_vae",
        "Plain Autoencoder (Baseline)": "plain_ae",
        "Intensity Thresholding (Baseline)": "intensity_threshold"
    }
    selected_model_type = model_type_map[model_type_label]

    # Image Resolution
    img_size = st.radio("Resolution canvas", [128, 256], index=0, horizontal=True,
                        help="256x256 retains high detail for smaller lesions.")

    # Ablation toggles
    st.markdown("#### Component Switches")
    use_fuzzy = st.checkbox("Fuzzy Contrast Enhancement", value=True,
                            disabled=selected_model_type in ("plain_vae", "plain_ae", "intensity_threshold"))
    use_resnet = st.checkbox("ResNet18 Feature Conditioning", value=True,
                             disabled=selected_model_type in ("plain_vae", "plain_ae", "intensity_threshold"))
    allow_download = st.checkbox("Allow ImageNet Weights Download", value=False)

    # Brain Normalization
    st.markdown("#### Intensity Normalization")
    norm_method = st.selectbox(
        "Normalization within Brain",
        ["percentile", "zscore", "minmax"],
        index=0,
        help="Percentile (1st-99th) is robust to bright bone or artifact spikes."
    )

    # Thresholding Calibration
    st.markdown("#### Thresholding Calibration")
    mode = st.radio("Threshold method", ["Calibrated", "Automatic", "Manual"], key="threshold_mode")
    calibrated_thresh = st.slider("Calibrated Validation Cutoff", 0.02, 0.40, DEFAULT_CALIBRATED_THRESHOLD, 0.01,
                                  disabled=mode != "Calibrated",
                                  help="Empirical 99th percentile of reconstruction error on healthy brains.")
    threshold = st.slider("Manual anomaly threshold", 0.0, 1.0, 0.20, 0.01,
                          disabled=mode != "Manual", key="manual_threshold")
    min_area = st.slider("Minimum candidate area (pixels)", 1, 200, 12)

    # Multi-modal fusion method
    st.markdown("#### Multi-Modal Fusion")
    fusion_method_label = st.selectbox(
        "Fusion Strategy",
        ["Calibrated Weighted Fusion (FLAIR 0.5, T2 0.3, T1 0.2)",
         "Multi-Channel Joint VAE",
         "Equal-Weight Average (1/N)"],
        index=0
    )
    fusion_method = "weighted" if "Weighted" in fusion_method_label else "multi_channel" if "Multi-Channel" in fusion_method_label else "equal"

    if st.button("Reload checkpoint / models"):
        cached_models.clear()
        st.session_state.pop("result", None)

signature = (str(CHECKPOINT), CHECKPOINT.stat().st_mtime_ns if CHECKPOINT.exists() else 0)
bundle = cached_models(signature, allow_download, selected_model_type, img_size)
bundle.use_fuzzy = use_fuzzy
bundle.use_resnet = use_resnet
bundle.model_type = selected_model_type
bundle.calibrated_threshold = calibrated_thresh

# Status banner
col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
with col_s1:
    if bundle.demo:
        st.warning(f"HIGH-FIDELITY DEMO & ABLATION MODE · Architecture: {bundle.model_type.upper()}")
        st.caption(bundle.status)
    else:
        st.success(f"{bundle.status} · Resolution: {img_size}x{img_size} · Calibrated Thresh: {bundle.calibrated_threshold:.3f}")
with col_s2:
    st.markdown(f'<span class="badge-pill badge-green">Fuzzy: {"ON" if use_fuzzy else "OFF"}</span>'
                f'<span class="badge-pill badge-blue">ResNet: {"ON" if use_resnet else "OFF"}</span>',
                unsafe_allow_html=True)
with col_s3:
    st.markdown(f'<span class="badge-pill badge-amber">Norm: {norm_method.title()}</span>', unsafe_allow_html=True)

# Main Application Tabs
analyze_tab, multi_tab, results_tab, eval_tab, model_tab, about_tab = st.tabs([
    "Analyze MRI", "Multi-Modal Analysis", "Results", "Evaluation & Ablations", "Model Information", "About Project"
])

with analyze_tab:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Inspect a Single MRI Slice")
        st.write("Upload an axial slice or explore standardized reproducible test samples.")
        source = st.radio("Input source", ["Use Demo Sample", "Upload MRI"], horizontal=True)
        gt_mask = None
        if source == "Use Demo Sample":
            sample = st.selectbox("Synthetic sample",
                                  ["A — symmetric reference", "B — one bright artificial region", "C — two artificial regions"],
                                  key="sample_choice")
            data, gt_mask = synthetic_sample(sample[0], size=img_size)
            provenance = {"source": "Synthetic test image", "sample": sample[0], "seed": 42}
            st.caption(f"Synthetic test image · sample {sample[0]} · size {img_size}x{img_size}")
            st.image(data, clamp=True, width=280)
        else:
            data, provenance, _ = upload_input("MRI image or NIfTI volume", "single_upload")

        if st.button("Run single-image analysis", type="primary", disabled=data is None, key="run_single"):
            execute(data, provenance, gt_mask=gt_mask)

    with right:
        with st.container(border=True):
            st.markdown("### Why Spatial VAE Localizes Better")
            st.write("• **Legacy 32-D VAE:** Global compression forces the entire image through a 32-number vector, causing errors to erupt across the entire healthy brain.")
            st.write("• **Spatial 16x16 Latent VAE:** Preserves 2D coordinates so local abnormalities only affect their own neighborhood.")
        with st.container(border=True):
            st.markdown("### Brain Masking & Calibration")
            st.write("• **Real Brain Mask:** Skull-stripping heuristic discards bright bones and scalp artifacts.")
            st.write(f"• **Calibrated Threshold:** Fixed at **{calibrated_thresh:.3f}** based on healthy validation data, avoiding per-image distortion.")

with multi_tab:
    st.subheader("Multi-Modal Sequence Fusion")
    st.write("Combine corresponding T1, T2, and FLAIR slices with validation-calibrated weights or joint spatial VAE.")
    multi_demo = st.checkbox("Use synthetic multi-modal software test", value=False)
    images, sources = {}, {}
    invalid = False
    gt_multi_mask = None
    if multi_demo:
        base, gt_multi_mask = synthetic_sample("C", size=img_size)
        images = {
            "FLAIR": np.clip(base * 1.15, 0, 1).astype(np.float32),
            "T2": np.sqrt(base).astype(np.float32),
            "T1": np.clip(base * 0.85, 0, 1).astype(np.float32)
        }
        sources = {"source": "Synthetic multi-modal test suite", "sample": "C", "seed": 42}
        st.caption("Generated pre-aligned FLAIR, T2, and T1 modalities with sample C ground truth.")
    else:
        for column, name in zip(st.columns(4), MODALITIES):
            with column:
                image, metadata, bad = upload_input(name + " MRI", "upload_" + name)
                invalid |= bad
                if image is not None:
                    images[name], sources[name] = image, metadata

    aligned = st.checkbox("I have checked that these inputs depict corresponding, pre-aligned slices", value=multi_demo)
    if st.button("Run multi-modal analysis", type="primary", disabled=not images or invalid or (len(images) > 1 and not aligned), key="run_multi"):
        execute(images, sources, multi=True, gt_mask=gt_multi_mask)

with results_tab:
    if "result" not in st.session_state:
        st.info("Run an analysis from 'Analyze MRI' or 'Multi-Modal Analysis' to inspect results.")
    else:
        result = st.session_state.result
        st.subheader("Inspection & Localization Results")
        st.caption(f"Model: {result.get('model_type', 'spatial_vae').upper()} · Threshold Mode: {result['threshold_mode']} ({result['threshold']:.3f}) · Normalization: {result.get('norm_method', 'percentile')}")

        # Metrics bar
        cols = st.columns(5)
        cols[0].metric("AI Anomaly Score", f'{result["score"]:.2f} / 100')
        cols[1].metric("Applied Threshold", f'{result["threshold"]:.3f}')
        cols[2].metric("Candidate Regions", len(result["regions"]))
        cols[3].metric("Largest Lesion (px)", max((r["area_pixels"] for r in result["regions"]), default=0))
        cols[4].metric("Mean Brain Error", f'{result["mean_error"]:.4f}')

        # Ground truth validation metrics if available
        if "live_dice" in result:
            st.markdown("#### Ground Truth Evaluation on this Case")
            g_cols = st.columns(4)
            g_cols[0].metric("Pixel Dice", f"{result['live_dice']:.4f}")
            g_cols[1].metric("Pixel IoU", f"{result['live_iou']:.4f}")
            sens = result["live_lesion"]["sensitivities"]["all"]
            g_cols[2].metric("Lesion Sensitivity", f"{sens:.2%}" if np.isfinite(sens) else "N/A")
            g_cols[3].metric("False Positives", f"{result['live_lesion']['false_positives']}")

        # 8 Image Panels
        panels = [
            ("1. Original Slice", result["original"]),
            ("2. Preprocessed (Brain Norm)", result["processed"]),
            ("3. Fuzzy Contrast Enhanced", result["enhanced"]),
            ("4. Reconstruction", result["reconstruction"]),
            ("5. Difference Map", result["error"]),
            ("6. Anomaly Heatmap", heatmap_rgb(result["heat"])),
            ("7. Binary Candidate Mask", result["mask"].astype(np.float32)),
            ("8. Candidate Overlay", overlay(result["original"], result["heat"], result["mask"], result["regions"]))
        ]
        for start in (0, 4):
            for col, (label_txt, img) in zip(st.columns(4), panels[start:start + 4]):
                with col:
                    st.image(img, caption=label_txt, use_container_width=True, clamp=True)

        if "modalities" in result:
            st.markdown("#### Per-Modality Anomaly Decomposition")
            st.caption(f"Fusion Strategy: {result.get('fusion_method', 'weighted')} · Weights: {json.dumps(result.get('weights', {}))}")
            m_cols = st.columns(len(result["modalities"]))
            for col, (m_name, ind) in zip(m_cols, result["modalities"].items()):
                with col:
                    st.image(ind["enhanced"], caption=m_name + " Processed", use_container_width=True)
                    st.image(heatmap_rgb(ind["heat"]), caption=m_name + " Anomaly", use_container_width=True)

        if result["regions"]:
            st.markdown("#### Detected Suspicious Lesion Coordinates")
            st.dataframe(pd.DataFrame(result["regions"]), use_container_width=True, hide_index=True)

        st.download_button("Download Research Archive · ZIP", result_archive(result),
                           "brain_mri_research_results.zip", "application/zip")

with eval_tab:
    st.subheader("Benchmark Evaluation & Ablation Studio")
    st.write("Batch evaluation across labelled test slices reporting pixel-level Dice, IoU, AUROC, AUPRC, lesion-level sensitivity (split by size), and false positives per image.")

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        if st.button("Run Benchmark Evaluation", type="primary"):
            testset = generate_benchmark_testset(n_cases=12, image_size=img_size)
            summary_rows, per_case = run_evaluation_suite(
                testset, bundle, mode=mode, threshold=calibrated_thresh,
                use_fuzzy=use_fuzzy, norm_method=norm_method
            )
            st.session_state.eval_summary = summary_rows
            st.session_state.eval_per_case = per_case

    with eval_col2:
        if st.button("Run Full Ablation Matrix (All Baselines)"):
            testset = generate_benchmark_testset(n_cases=12, image_size=img_size)
            with st.spinner("Evaluating all models and ablation configurations…"):
                ablation_data = run_ablation_benchmarks(testset)
                st.session_state.ablation_matrix = ablation_data

    if "eval_summary" in st.session_state:
        st.markdown("### Evaluation Metrics (Mean ± Std)")
        df_summary = pd.DataFrame(st.session_state.eval_summary)
        st.dataframe(df_summary[["Metric", "Subset", "Mean ± Std", "N_Samples"]],
                     use_container_width=True, hide_index=True)

        csv_data = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button("Download Evaluation Summary CSV", csv_data,
                           "evaluation_summary.csv", "text/csv")

    if "ablation_matrix" in st.session_state:
        st.markdown("### Ablation Comparison: Proposed vs Baselines")
        rows = []
        for exp_name, exp_res in st.session_state.ablation_matrix.items():
            s_map = {r["Metric"] + " (" + r["Subset"] + ")": r["Mean ± Std"] for r in exp_res["summary"]}
            rows.append({
                "Configuration": exp_name,
                "Pixel Dice": s_map.get("Pixel Dice (All Lesions)", "N/A"),
                "Pixel IoU": s_map.get("Pixel IoU (All Lesions)", "N/A"),
                "Lesion Sens (All)": s_map.get("Lesion Sensitivity (All Lesions)", "N/A"),
                "Lesion Sens (Small)": s_map.get("Lesion Sensitivity (Small (<100 px))", "N/A"),
                "Lesion Sens (Med)": s_map.get("Lesion Sensitivity (Medium (100-400 px))", "N/A"),
                "Lesion Sens (Large)": s_map.get("Lesion Sensitivity (Large (>400 px))", "N/A"),
                "FPPI": s_map.get("False Positives Per Image (FPPI) (Per Image)", "N/A"),
                "Image AUROC": s_map.get("Image-level AUROC (Dataset Level)", "N/A")
            })
        df_ablation = pd.DataFrame(rows)
        st.dataframe(df_ablation, use_container_width=True, hide_index=True)
        st.download_button("Download Ablation Comparison CSV", df_ablation.to_csv(index=False).encode("utf-8"),
                           "ablation_comparison.csv", "text/csv")

with model_tab:
    st.subheader("Model Architecture & Technical Specification")
    st.markdown(r"""
    **1. Fully Convolutional Spatial VAE (16×16×C Latent Map)**
    - Solves global error explosion: local lesions affect only their spatial receptive field.
    - Convolutions downsample from 128×128 (or 256×256) to a 16×16 spatial grid.
    - Reparameterization samples $z(x, y) \sim \mathcal{N}(\mu(x, y), \sigma^2(x, y))$.
    
    **2. Real Brain Extraction & Outlier-Robust Normalization**
    - Morphological skull-stripping heuristic discards dura, skull, and scalp artifacts.
    - 1st to 99th percentile normalization prevents intensity spikes from compressing soft tissue dynamic range.
    
    **3. Threshold Calibration on Healthy Validation Data**
    - Replaces ad-hoc per-image rules with the 99th percentile error of healthy validation brains.
    - Ensures clean slices don't produce false lesions, while large lesions don't inflate the threshold.
    
    **4. Real Multi-Modal Fusion**
    - Calibrated weights prioritize FLAIR (0.50) and T2 (0.30) for hyperintensity detection, supported by T1 (0.20).
    - Multi-Channel VAE accepts T1, T2, and FLAIR simultaneously.
    """)
    st.json({
        "active_architecture": bundle.model_type,
        "latent_dimensions": "16x16x8 spatial map" if "spatial" in bundle.model_type else "32-D dense",
        "norm_method": norm_method,
        "calibrated_threshold": calibrated_thresh,
        "image_resolution": f"{img_size}x{img_size}"
    })

with about_tab:
    st.subheader("About the Research Prototype")
    st.markdown("""
    This system implements unsupervised anomaly detection for Brain MRI by integrating:
    1. Spatial latent Variational Autoencoders
    2. Fuzzy contrast enhancement with entropy adaptation
    3. Multi-modal sequence fusion
    4. Outlier-robust brain normalization and threshold calibration
    
    **Evaluation Benchmark:**
    Evaluates pixel Dice/IoU/AUROC/AUPRC, lesion-level sensitivity by size, and false positives per image (FPPI).
    """)
    st.info(DISCLAIMER)
