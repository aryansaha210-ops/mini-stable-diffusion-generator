import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128

NUM_BATCHES = 20

VAE_CHECKPOINT = (
    "checkpoints/spatial_vae_epoch_10.pth"
)


# ============================================================
# SPATIAL VAE
# EXACT TRAINING ARCHITECTURE
# ============================================================

class SpatialVAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                3,
                1,
                1
            ),

            nn.ReLU()
        )

        self.mu = nn.Conv2d(
            128,
            4,
            kernel_size=3,
            padding=1
        )

        self.logvar = nn.Conv2d(
            128,
            4,
            kernel_size=3,
            padding=1
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(
                4,
                128,
                3,
                1,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                1,
                3,
                1,
                1
            ),

            nn.Tanh()
        )

    def encode(self, x):

        h = self.encoder(x)

        mu = self.mu(h)

        logvar = self.logvar(h)

        return mu, logvar

    def decode(self, z):

        return self.decoder(z)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("VAE LATENT DISTRIBUTION DIAGNOSTIC")
print("=" * 60)

print()

print(
    "Device:",
    DEVICE
)

if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print()
print("Loading SpatialVAE...")

vae = SpatialVAE().to(
    DEVICE
)

checkpoint = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE
)

vae.load_state_dict(
    checkpoint
)

vae.eval()

print("SpatialVAE loaded.")


# ============================================================
# DATASET
#
# EXACT SAME PREPROCESSING USED FOR TRAINING
# ============================================================

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


print()
print("Loading MNIST...")

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    "Dataset size:",
    len(dataset)
)


# ============================================================
# COLLECT LATENT STATISTICS
# ============================================================

all_means = []

all_stds = []

all_mins = []

all_maxs = []

all_latents = []

total_images = 0


print()
print(
    "Analyzing VAE latent distribution..."
)

print(
    "Batches:",
    NUM_BATCHES
)

print()


with torch.no_grad():

    for batch_index, (images, labels) in enumerate(loader):

        if batch_index >= NUM_BATCHES:
            break

        images = images.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Encode image
        # ----------------------------------------------------

        mu, logvar = vae.encode(
            images
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # The diffusion model was trained on MU directly.
        # ----------------------------------------------------

        latents = mu

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_mean = (
            latents.mean().item()
        )

        batch_std = (
            latents.std().item()
        )

        batch_min = (
            latents.min().item()
        )

        batch_max = (
            latents.max().item()
        )

        all_means.append(
            batch_mean
        )

        all_stds.append(
            batch_std
        )

        all_mins.append(
            batch_min
        )

        all_maxs.append(
            batch_max
        )

        all_latents.append(
            latents.detach().cpu()
        )

        total_images += (
            images.shape[0]
        )

        print(
            f"Batch {batch_index + 1:2d} | "
            f"mean={batch_mean:.6f} | "
            f"std={batch_std:.6f} | "
            f"min={batch_min:.6f} | "
            f"max={batch_max:.6f}"
        )


# ============================================================
# COMBINE LATENTS
# ============================================================

latents = torch.cat(
    all_latents,
    dim=0
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

global_mean = (
    latents.mean().item()
)

global_std = (
    latents.std().item()
)

global_min = (
    latents.min().item()
)

global_max = (
    latents.max().item()
)


# ============================================================
# CHANNEL STATISTICS
# ============================================================

channel_mean = (
    latents.mean(
        dim=(0, 2, 3)
    )
)

channel_std = (
    latents.std(
        dim=(0, 2, 3)
    )
)

channel_min = (
    latents.amin(
        dim=(0, 2, 3)
    )
)

channel_max = (
    latents.amax(
        dim=(0, 2, 3)
    )
)


# ============================================================
# RANDOM NORMAL COMPARISON
# ============================================================

random_latent = torch.randn(
    latents.shape
)

random_mean = (
    random_latent.mean().item()
)

random_std = (
    random_latent.std().item()
)

random_min = (
    random_latent.min().item()
)

random_max = (
    random_latent.max().item()
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print()

print(
    "Images analyzed:",
    total_images
)

print(
    "Latent shape:",
    tuple(latents.shape)
)

print()

print("REAL VAE LATENTS")
print("-" * 60)

print(
    "Mean:",
    global_mean
)

print(
    "Std:",
    global_std
)

print(
    "Min:",
    global_min
)

print(
    "Max:",
    global_max
)

print()

print("PER-CHANNEL STATISTICS")
print("-" * 60)

for channel in range(4):

    print(
        f"Channel {channel}: "
        f"mean={channel_mean[channel].item():.6f} | "
        f"std={channel_std[channel].item():.6f} | "
        f"min={channel_min[channel].item():.6f} | "
        f"max={channel_max[channel].item():.6f}"
    )


print()

print("STANDARD NORMAL N(0,1)")
print("-" * 60)

print(
    "Mean:",
    random_mean
)

print(
    "Std:",
    random_std
)

print(
    "Min:",
    random_min
)

print(
    "Max:",
    random_max
)


# ============================================================
# COMPARISON
# ============================================================

print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)

print()

print(
    f"Real latent mean : {global_mean:.6f}"
)

print(
    f"Random mean      : {random_mean:.6f}"
)

print()

print(
    f"Real latent std  : {global_std:.6f}"
)

print(
    f"Random std       : {random_std:.6f}"
)

print()


if abs(global_mean) < 0.2:

    print(
        "Mean check: REAL LATENTS ARE "
        "ROUGHLY CENTERED AROUND ZERO."
    )

else:

    print(
        "Mean check: REAL LATENTS HAVE "
        "A NOTICEABLE MEAN SHIFT."
    )


if abs(global_std - 1.0) < 0.2:

    print(
        "Std check: REAL LATENTS HAVE "
        "A SCALE CLOSE TO N(0,1)."
    )

else:

    print(
        "Std check: REAL LATENTS HAVE "
        "A DIFFERENT SCALE FROM N(0,1)."
    )


print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)