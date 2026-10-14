# Spatial VAE-Enhanced Multi-Modal Brain Lesion Detector

**Using Fully Convolutional Spatial Latent Variational Autoencoders, Outlier-Robust Normalization, Validation Calibration, and Ablation Benchmarking**

A local Streamlit research prototype for inspecting suspicious regions in brain MRI slices. It runs on CPU, starts without a checkpoint, includes reproducible synthetic software samples, and provides a spatial convolutional VAE training and evaluation path.

> Research prototype only. This system is not a medical diagnosis and must not replace evaluation by a qualified medical professional.

## Key Improvements & Architectural Updates

1. **Evaluation & Benchmark Suite (`evaluation/`):**
   - Batch evaluation CLI script (`python -m evaluation.evaluate`) running over labelled test sets.
   - Computes pixel-level **Dice, IoU, AUROC, and AUPRC**.
   - Computes lesion-level **sensitivity** (overall and split by size: Small `<100 px`, Medium `100-400 px`, Large `>400 px`) and **False Positives Per Image (FPPI)**.
   - Computes **image-level AUROC**.
   - Exports summary tables (`Mean ± Std`) and per-case breakdowns to CSV.

2. **Multi-Slice Training & Patient Splitting (`training/dataset.py`):**
   - Slices through the middle 30%–70% of brain extent across 3D volumes rather than relying solely on a single center slice.
   - Automatically filters out empty background slices.
   - Guarantees clean subject/patient-level splits (`split_patients`) with zero slice cross-contamination.

3. **Fully Convolutional Spatial VAE (`models/vae.py`):**
   - Replaces the 1D 32-number bottleneck with a **16×16×C spatial latent map**.
   - Constrains reconstruction errors to local abnormal neighborhoods, preventing lesions from corrupting global brain reconstruction.
   - Supports 128×128 and 256×256 resolutions to prevent small lesions from shrinking to a few pixels.

4. **Real Brain Masking & Validation Threshold Calibration (`processing/`):**
   - Replaces `image > 0.03` with an anatomical morphological brain mask (skull stripping).
   - Replaces outlier-sensitive min-max normalization with percentile (1st–99th) and z-score normalization within the brain.
   - Replaces per-image threshold heuristic with a fixed calibrated threshold derived from the empirical 99th/95th percentile of healthy validation reconstruction errors.

5. **Multi-Modal Sequence Fusion (`processing/multimodal.py`):**
   - Calibrated weighted fusion (FLAIR 0.50, T2 0.30, T1 0.20) prioritizing hyperintensity sensitivity.
   - Joint Multi-Channel Spatial VAE taking T1, T2, and FLAIR simultaneously.
   - Legacy equal-weight fusion available for direct comparison.

6. **Ablation Switches & Baseline Models:**
   - Config flags to toggle: Fuzzy Enhancement On/Off, ResNet Conditioning On/Off, Pretrained On/Off, Fusion On/Off.
   - Baseline models included: Plain Autoencoder, Plain VAE, and Simple Intensity Thresholding.
   - Batch ablation CLI runner (`python -m evaluation.evaluate --ablation`) generating comparative ablation matrices.

---

## Quick start — Windows / Python 3.11, 3.12, or 3.13

From this project folder in PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Batch Evaluation & Benchmarking

Run the evaluation benchmark on the test suite:

```powershell
# Standard evaluation on synthetic or user test set
python -m evaluation.evaluate --output-csv evaluation_summary.csv

# Run full ablation benchmark comparing Spatial VAE vs Baselines
python -m evaluation.evaluate --ablation --output-csv ablation_comparison.csv

# Evaluate on custom user MRI slices and masks
python -m evaluation.evaluate --images path/to/images --masks path/to/masks --output-csv user_results.csv
```

---

## Training on Multi-Slice Data

```powershell
# Train Spatial VAE on all brain-containing axial slices (30-70% extent) with patient-level splitting:
python training/train_vae.py --data healthy_data --split-patients --epochs 20 --image-size 128

# Train with specific ablation options:
python training/train_vae.py --data healthy_data --val-data healthy_val --model-type spatial_vae --no-fuzzy --no-resnet
```

---

## Project layout

```text
app.py                     Streamlit dashboard, ablation controls & evaluation tab
config.py                  Shared dimensions, calibrated thresholds, and ablation flags
evaluation/                Batch evaluation script (evaluate.py) & metrics (metrics.py)
models/                    Spatial VAE, Multi-Channel VAE, ConvVAE, and Baselines
processing/                Preprocessing, real brain mask, calibrated thresholding, fusion
training/                  Multi-slice dataset with patient splitting and training CLI
utils/                     Image I/O, synthetic fixtures, visualization
sample_data/               Synthetic software test fixtures (seed 42)
tests/                     Comprehensive unit, integration, and UI test suite
```

## Testing

```powershell
pytest -q
python -m compileall -q app.py config.py models processing utils training evaluation tests
```
