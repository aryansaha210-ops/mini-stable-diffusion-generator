import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "checkpoints/spatial_vae_epoch_10.pth"


# ============================================
# SPATIAL VAE
# ============================================

class SpatialVAE(nn.Module):

    def __init__(self, latent_channels=4):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

        # Latent
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

        # Decoder
        self.decoder = nn.Sequential(

            nn.Conv2d(
                latent_channels,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
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


    def encode(self, x):

        features = self.encoder(x)

        mu = self.mu(features)

        logvar = self.logvar(features)

        return mu, logvar


    def decode(self, z):

        return self.decoder(z)


# ============================================
# LOAD MODEL
# ============================================

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


model = SpatialVAE().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

print()
print("Spatial VAE loaded successfully!")


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
    train=False,
    download=True,
    transform=transform
)


# ============================================
# SELECT 8 IMAGES
# ============================================

images = []

labels = []

for i in range(8):

    image, label = dataset[i]

    images.append(image)

    labels.append(label)


images = torch.stack(images).to(DEVICE)


# ============================================
# ENCODE
# ============================================

with torch.no_grad():

    mu, logvar = model.encode(
        images
    )

    # Use mean latent for stable reconstruction
    reconstruction = model.decode(
        mu
    )


# ============================================
# PRINT SHAPES
# ============================================

print()
print("Input shape:")
print(images.shape)

print()
print("Latent shape:")
print(mu.shape)

print()
print("Reconstruction shape:")
print(reconstruction.shape)


# ============================================
# CONVERT FOR DISPLAY
# ============================================

original_images = (
    images.cpu()
    .squeeze(1)
    .numpy()
)

reconstructed_images = (
    reconstruction.cpu()
    .squeeze(1)
    .numpy()
)


# Convert [-1, 1] → [0, 1]

original_images = (
    original_images + 1
) / 2

reconstructed_images = (
    reconstructed_images + 1
) / 2


# ============================================
# CREATE COMPARISON
# ============================================

fig, axes = plt.subplots(
    2,
    8,
    figsize=(12, 4)
)


for i in range(8):

    # Original
    axes[0, i].imshow(
        original_images[i],
        cmap="gray"
    )

    axes[0, i].set_title(
        f"Original {labels[i]}"
    )

    axes[0, i].axis("off")


    # Reconstruction
    axes[1, i].imshow(
        reconstructed_images[i],
        cmap="gray"
    )

    axes[1, i].set_title(
        "Reconstructed"
    )

    axes[1, i].axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/spatial_vae_reconstruction.png",
    dpi=150
)

plt.close()


# ============================================
# COMPLETE
# ============================================

print()
print(
    "===================================="
)

print(
    "SPATIAL VAE TEST COMPLETED"
)

print(
    "===================================="
)

print()
print(
    "Saved to:"
)

print(
    "outputs/spatial_vae_reconstruction.png"
)