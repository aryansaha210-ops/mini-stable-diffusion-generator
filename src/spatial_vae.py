import torch
import torch.nn as nn


# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_CHANNELS = 4


# ============================================
# SPATIAL VAE
# ============================================

class SpatialVAE(nn.Module):

    def __init__(
        self,
        latent_channels=4
    ):

        super().__init__()

        # ====================================
        # ENCODER
        # 32x32 -> 16x16 -> 8x8
        # ====================================

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

        # ====================================
        # LATENT
        # ====================================

        self.mu = nn.Conv2d(
            128,
            latent_channels,
            kernel_size=3,
            padding=1
        )

        self.logvar = nn.Conv2d(
            128,
            latent_channels,
            kernel_size=3,
            padding=1
        )

        # ====================================
        # DECODER
        # 8x8 -> 16x16 -> 32x32
        # ====================================

        self.decoder = nn.Sequential(

            nn.Conv2d(
                latent_channels,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1
            ),

            nn.Tanh()
        )


    # ========================================
    # ENCODE
    # ========================================

    def encode(self, x):

        features = self.encoder(x)

        mu = self.mu(
            features
        )

        logvar = self.logvar(
            features
        )

        return mu, logvar


    # ========================================
    # REPARAMETERIZATION
    # ========================================

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        noise = torch.randn_like(
            std
        )

        return (
            mu
            +
            noise * std
        )


    # ========================================
    # DECODE
    # ========================================

    def decode(self, z):

        return self.decoder(z)


    # ========================================
    # FORWARD
    # ========================================

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        return (
            reconstruction,
            mu,
            logvar
        )


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = SpatialVAE(
        LATENT_CHANNELS
    ).to(DEVICE)

    test_image = torch.randn(
        4,
        1,
        32,
        32,
        device=DEVICE
    )

    reconstruction, mu, logvar = model(
        test_image
    )

    print()
    print(
        "Input shape:"
    )

    print(
        test_image.shape
    )

    print()
    print(
        "Latent shape:"
    )

    print(
        mu.shape
    )

    print()
    print(
        "Reconstruction shape:"
    )

    print(
        reconstruction.shape
    )

    print()
    print(
        "Spatial VAE test successful!"
    )