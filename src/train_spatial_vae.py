import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================
# CONFIGURATION
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

LATENT_CHANNELS = 4

# Very small KL weight to prevent posterior collapse
KL_WEIGHT = 0.00001


print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


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
        #
        # 32x32
        #   ↓
        # 16x16
        #   ↓
        # 8x8
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
        # LATENT DISTRIBUTION
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
        #
        # 8x8
        #   ↓
        # 16x16
        #   ↓
        # 32x32
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
    # ENCODER
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

        z = (
            mu
            +
            noise * std
        )

        return z


    # ========================================
    # DECODER
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
# DATASET
# ============================================

transform = transforms.Compose([

    transforms.Resize(
        (32, 32)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================
# MODEL
# ============================================

model = SpatialVAE(
    LATENT_CHANNELS
).to(DEVICE)


# ============================================
# OPTIMIZER
# ============================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# ============================================
# TRAINING
# ============================================

print()
print(
    "Starting spatial VAE training..."
)


for epoch in range(EPOCHS):

    total_loss = 0.0

    total_reconstruction = 0.0

    total_kl = 0.0


    for images, _ in loader:

        images = images.to(
            DEVICE
        )


        # ====================================
        # FORWARD PASS
        # ====================================

        reconstruction, mu, logvar = model(
            images
        )


        # ====================================
        # RECONSTRUCTION LOSS
        # ====================================

        reconstruction_loss = torch.mean(
            (
                reconstruction
                -
                images
            ) ** 2
        )


        # ====================================
        # KL DIVERGENCE
        # ====================================

        kl_loss = -0.5 * torch.mean(
            1
            +
            logvar
            -
            mu.pow(2)
            -
            logvar.exp()
        )


        # ====================================
        # TOTAL LOSS
        # ====================================

        loss = (
            reconstruction_loss
            +
            KL_WEIGHT * kl_loss
        )


        # ====================================
        # BACKPROPAGATION
        # ====================================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ====================================
        # RECORD LOSSES
        # ====================================

        total_loss += loss.item()

        total_reconstruction += (
            reconstruction_loss.item()
        )

        total_kl += (
            kl_loss.item()
        )


    # ========================================
    # AVERAGE LOSSES
    # ========================================

    average_loss = (
        total_loss
        /
        len(loader)
    )

    average_reconstruction = (
        total_reconstruction
        /
        len(loader)
    )

    average_kl = (
        total_kl
        /
        len(loader)
    )


    # ========================================
    # PRINT RESULTS
    # ========================================

    print(
        f"Epoch {epoch + 1} completed "
        f"| Loss: {average_loss:.4f} "
        f"| Reconstruction: "
        f"{average_reconstruction:.4f} "
        f"| KL: {average_kl:.4f}"
    )


    # ========================================
    # SAVE CHECKPOINT
    # ========================================

    torch.save(
        model.state_dict(),
        f"checkpoints/"
        f"spatial_vae_epoch_"
        f"{epoch + 1}.pth"
    )


# ============================================
# COMPLETE
# ============================================

print()

print(
    "===================================="
)

print(
    "SPATIAL VAE TRAINING COMPLETED"
)

print(
    "===================================="
)