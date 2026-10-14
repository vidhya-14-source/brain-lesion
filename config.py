from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 128
LATENT_DIM = 32
FEATURE_DIM = 512
PIPELINE_VERSION = "fuzzy128-v2-spatial"
CHECKPOINT = ROOT / "checkpoints" / "vae_best.pth"
MODALITIES = ("T1", "T2", "FLAIR", "DWI")

# Supported image resolutions
SUPPORTED_SIZES = (128, 256)

# Spatial VAE configuration (16x16 spatial latent map)
SPATIAL_LATENT_CHANNELS = 8
SPATIAL_LATENT_SIZE = 16

# Intensity normalization within brain mask
NORM_METHODS = ("percentile", "zscore", "minmax")
DEFAULT_NORM_METHOD = "percentile"

# Calibration on healthy validation errors
THRESHOLD_MODES = ("Calibrated", "Automatic", "Manual")
DEFAULT_THRESHOLD_MODE = "Calibrated"
DEFAULT_CALIBRATED_PERCENTILE = 99.0
DEFAULT_CALIBRATED_THRESHOLD = 0.12

# Ablation defaults and model types
MODEL_TYPES = ("spatial_vae", "dense_vae", "plain_vae", "plain_ae", "intensity_threshold")
DEFAULT_MODEL_TYPE = "spatial_vae"
DEFAULT_USE_FUZZY = True
DEFAULT_USE_RESNET = True
DEFAULT_PRETRAINED = False
DEFAULT_USE_FUSION = True
FUSION_METHODS = ("weighted", "multi_channel", "equal")
DEFAULT_FUSION_METHOD = "weighted"

# Default modality validation weights (FLAIR most sensitive to hyperintensities)
DEFAULT_MODALITY_WEIGHTS = {"FLAIR": 0.50, "T2": 0.30, "T1": 0.20, "DWI": 0.30}

# Disclaimers
DISCLAIMER = "Research prototype only. This system is not a medical diagnosis and must not replace evaluation by a qualified medical professional."
DEMO_WARNING = "DEMO MODE — outputs are generated for software demonstration and are not clinically validated."
DEMO_OUTPUT = "Illustrative demo output — not produced by a validated trained VAE."
