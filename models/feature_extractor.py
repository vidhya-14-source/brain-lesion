from pathlib import Path
import warnings
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class FeatureExtractor(nn.Module):
    """Frozen 512-D ResNet18 descriptors; cached pretrained weights preferred."""
    def __init__(self, allow_download=False, use_pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        cached = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        self.pretrained = False
        self.status = "Untrained ResNet18 initialization; transfer learning is inactive."
        # Keep fallback reproducible without resetting the caller's RNG.
        with torch.random.fork_rng():
            torch.manual_seed(42)
            network = resnet18(weights=None)
        if use_pretrained and (allow_download or cached.exists()):
            try:
                state = weights.get_state_dict(progress=False, check_hash=True)
                network.load_state_dict(state)
                self.pretrained = True
                self.status = "ImageNet-pretrained ResNet18; general visual features, not MRI diagnostic training."
            except Exception as exc:
                self.status += f" Pretrained weights unavailable ({type(exc).__name__})."
                warnings.warn(self.status, RuntimeWarning)
        network.fc = nn.Identity()
        self.network = network.eval()
        self.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        self.network.eval()
        rgb = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        return self.network((rgb - self.mean) / self.std)
