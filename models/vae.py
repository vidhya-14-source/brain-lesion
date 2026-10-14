import torch
from torch import nn
from config import LATENT_DIM, FEATURE_DIM, SPATIAL_LATENT_CHANNELS, SPATIAL_LATENT_SIZE


class ConvVAE(nn.Module):
    """Legacy 128x128 convolutional VAE with 32-D dense bottleneck conditioned on ResNet18."""
    def __init__(self, latent_dim=LATENT_DIM, feature_dim=FEATURE_DIM, in_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Flatten())
        self.mu = nn.Linear(128 * 8 * 8 + feature_dim, latent_dim)
        self.logvar = nn.Linear(128 * 8 * 8 + feature_dim, latent_dim)
        self.project = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, 2, 1), nn.Sigmoid())

    def encode(self, x, features=None):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"ConvVAE requires {self.in_channels} input channels.")
        if features is None:
            features = x.new_zeros((x.shape[0], self.feature_dim))
        h = torch.cat([self.encoder(x), features], dim=1)
        return self.mu(h), self.logvar(h).clamp(-12, 12)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(self.project(z).view(-1, 128, 8, 8))

    def forward(self, x, features=None):
        mu, logvar = self.encode(x, features)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar


class SpatialVAE(nn.Module):
    """Fully convolutional VAE with a spatial latent map (e.g. 16x16xC).
    
    Preserves local spatial information to prevent localized lesions from
    corrupting the entire global latent representation. Supports 128x128 and 256x256.
    """
    def __init__(self, in_channels=1, out_channels=None, latent_channels=SPATIAL_LATENT_CHANNELS,
                 use_resnet=True, feature_dim=FEATURE_DIM, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.latent_channels = latent_channels
        self.use_resnet = use_resnet
        self.feature_dim = feature_dim
        self.input_size = input_size

        # Encoder: downsample to 16x16
        # If input_size == 128: 128 -> 64 -> 32 -> 16 (stride 2 x 3)
        # If input_size == 256: 256 -> 128 -> 64 -> 32 -> 16 (stride 2 x 4)
        encoder_layers = [
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
        ]
        if input_size >= 256:
            encoder_layers.extend([
                nn.Conv2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            ])
        self.encoder = nn.Sequential(*encoder_layers)

        # Optional ResNet conditioning projection (spatial expansion)
        enc_out_channels = 128
        if use_resnet:
            self.resnet_proj = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(inplace=True)
            )
            latent_in_channels = enc_out_channels + 32
        else:
            self.resnet_proj = None
            latent_in_channels = enc_out_channels

        # Fully convolutional spatial latent heads (16x16 spatial map)
        self.conv_mu = nn.Conv2d(latent_in_channels, latent_channels, kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv2d(latent_in_channels, latent_channels, kernel_size=3, padding=1)

        # Decoder: upsample from 16x16 back to input_size
        self.decoder_input = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        decoder_layers = []
        if input_size >= 256:
            decoder_layers.extend([
                nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ])
        decoder_layers.extend([
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, self.out_channels, 4, 2, 1), nn.Sigmoid()
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, features=None):
        h = self.encoder(x)
        # Ensure spatial dimensions are 16x16 regardless of minor rounding
        if h.shape[2:] != (SPATIAL_LATENT_SIZE, SPATIAL_LATENT_SIZE):
            h = nn.functional.adaptive_avg_pool2d(h, (SPATIAL_LATENT_SIZE, SPATIAL_LATENT_SIZE))

        if self.use_resnet:
            if features is None:
                features = x.new_zeros((x.shape[0], self.feature_dim))
            f_proj = self.resnet_proj(features)  # (B, 32)
            f_spatial = f_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.shape[2], h.shape[3])
            h = torch.cat([h, f_spatial], dim=1)

        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h).clamp(-12, 12)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        h = self.decoder_input(z)
        out = self.decoder(h)
        return out

    def forward(self, x, features=None):
        mu, logvar = self.encode(x, features)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar


class MultiChannelVAE(SpatialVAE):
    """Multi-channel Spatial VAE taking T1, T2, FLAIR together."""
    def __init__(self, in_channels=3, latent_channels=SPATIAL_LATENT_CHANNELS,
                 use_resnet=True, feature_dim=FEATURE_DIM, input_size=128):
        super().__init__(in_channels=in_channels, out_channels=in_channels,
                         latent_channels=latent_channels, use_resnet=use_resnet,
                         feature_dim=feature_dim, input_size=input_size)


class PlainAutoencoder(nn.Module):
    """Deterministic convolutional Autoencoder baseline (no KL divergence, no latent sampling)."""
    def __init__(self, in_channels=1, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        encoder_layers = [
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True)
        ]
        if input_size >= 256:
            encoder_layers.extend([nn.Conv2d(128, 128, 4, 2, 1), nn.ReLU(inplace=True)])
        self.encoder = nn.Sequential(*encoder_layers)
        self.bottleneck = nn.Conv2d(128, SPATIAL_LATENT_CHANNELS, 3, 1, 1)

        decoder_layers = [nn.Conv2d(SPATIAL_LATENT_CHANNELS, 128, 3, 1, 1), nn.ReLU(inplace=True)]
        if input_size >= 256:
            decoder_layers.extend([nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(inplace=True)])
        decoder_layers.extend([
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1), nn.Sigmoid()
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, features=None):
        z = self.bottleneck(self.encoder(x))
        recon = self.decoder(z)
        # Dummy mu and logvar for interface compatibility
        return recon, z, torch.zeros_like(z)


class PlainVAE(SpatialVAE):
    """Plain VAE baseline (no ResNet conditioning, no fuzzy enhancement)."""
    def __init__(self, in_channels=1, input_size=128):
        super().__init__(in_channels=in_channels, use_resnet=False, input_size=input_size)


def vae_loss(reconstruction, target, mu, logvar, beta=0.001):
    """Unified VAE loss with spatial or 1D latent KL divergence."""
    reconstruction_loss = torch.mean((reconstruction - target) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    return reconstruction_loss + beta * kl, reconstruction_loss, kl


def ae_loss(reconstruction, target):
    """Deterministic autoencoder reconstruction loss."""
    reconstruction_loss = torch.mean((reconstruction - target) ** 2)
    return reconstruction_loss, reconstruction_loss, torch.tensor(0.0, device=reconstruction.device)
