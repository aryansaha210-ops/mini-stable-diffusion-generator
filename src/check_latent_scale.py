import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from spatial_vae import SpatialVAE


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = r".\checkpoints\spatial_vae_epoch_10.pth"

BATCH_SIZE = 64


# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("LATENT SCALE DIAGNOSTIC")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD VAE
# ============================================================

print("\nLoading SpatialVAE...")

vae = SpatialVAE().to(DEVICE)

checkpoint = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model" in checkpoint:
    vae.load_state_dict(checkpoint["model"])
else:
    vae.load_state_dict(checkpoint)

vae.eval()

print("SpatialVAE loaded successfully.")


# ============================================================
# LOAD MNIST
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# COLLECT LATENTS
# ============================================================

print("\nEncoding MNIST images...")

all_latents = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        mu, logvar = vae.encode(images)

        # Use mean latent instead of random sampling
        latents = mu

        all_latents.append(latents.cpu())


latents = torch.cat(all_latents, dim=0)


# ============================================================
# LATENT STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("LATENT STATISTICS")
print("=" * 60)

print("Shape:", tuple(latents.shape))

print("Mean:", latents.mean().item())

print("Std:", latents.std().item())

print("Min:", latents.min().item())

print("Max:", latents.max().item())

print("Absolute mean:", latents.abs().mean().item())


# ============================================================
# RANDOM NOISE STATISTICS
# ============================================================

noise = torch.randn_like(latents)

print("\n" + "=" * 60)
print("RANDOM NOISE STATISTICS")
print("=" * 60)

print("Mean:", noise.mean().item())

print("Std:", noise.std().item())

print("Min:", noise.min().item())

print("Max:", noise.max().item())


# ============================================================
# SCALE RATIO
# ============================================================

latent_std = latents.std().item()
noise_std = noise.std().item()

print("\n" + "=" * 60)
print("SCALE ANALYSIS")
print("=" * 60)

print("Real latent std :", latent_std)
print("Noise std       :", noise_std)

if latent_std > 0:

    scale_ratio = noise_std / latent_std

    print("Noise / latent ratio:", scale_ratio)

    print("\nRecommended latent scaling factor:")

    print(scale_ratio)

else:

    print("ERROR: latent standard deviation is zero.")


# ============================================================
# TEST RECONSTRUCTION
# ============================================================

print("\n" + "=" * 60)
print("RECONSTRUCTION TEST")
print("=" * 60)

images, labels = next(iter(loader))

images = images.to(DEVICE)

with torch.no_grad():

    mu, logvar = vae.encode(images)

    reconstructed = vae.decode(mu)

    mse = torch.mean(
        (images - reconstructed) ** 2
    ).item()

print("Reconstruction MSE:", mse)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)