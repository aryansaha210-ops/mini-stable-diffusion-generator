import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import os


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = (
    "checkpoints/spatial_vae_epoch_10.pth"
)

OUTPUT_DIR = (
    "outputs/vae_reconstruction"
)


# ============================================================
# SPATIAL VAE
# EXACT TRAINED ARCHITECTURE
# ============================================================

class SpatialVAE(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Latent parameters
        # ----------------------------------------------------

        self.mu = nn.Conv2d(
            128,
            4,
            3,
            padding=1
        )

        self.logvar = nn.Conv2d(
            128,
            4,
            3,
            padding=1
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

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
# START
# ============================================================

print("=" * 60)
print("SPATIAL VAE RECONSTRUCTION TEST")
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


# ============================================================
# LOAD VAE
# ============================================================

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
# SAME PREPROCESSING AS TRAINING
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

print(
    "Dataset size:",
    len(dataset)
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# RECONSTRUCT 10 DIGITS
# ============================================================

print()
print("Testing VAE reconstruction...")
print()


total_mse = 0.0


with torch.no_grad():

    for index in range(10):

        image, label = dataset[index]

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image_batch = image.unsqueeze(
            0
        ).to(
            DEVICE
        )

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        mu, logvar = vae.encode(
            image_batch
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # The diffusion model uses MU directly.
        #
        # For reconstruction, we also decode MU directly.
        # This removes random sampling from the test.
        # ----------------------------------------------------

        reconstruction = vae.decode(
            mu
        )

        # ----------------------------------------------------
        # Calculate reconstruction MSE
        # ----------------------------------------------------

        mse = torch.mean(
            (
                image_batch
                -
                reconstruction
            ) ** 2
        ).item()

        total_mse += mse

        # ----------------------------------------------------
        # Convert original
        # [-1,1] -> [0,1]
        # ----------------------------------------------------

        original = (
            image_batch[0, 0]
            .cpu()
            + 1.0
        ) / 2.0

        original = torch.clamp(
            original,
            0.0,
            1.0
        )

        original = (
            original.numpy()
            * 255.0
        ).astype(
            "uint8"
        )

        # ----------------------------------------------------
        # Convert reconstruction
        # [-1,1] -> [0,1]
        # ----------------------------------------------------

        reconstructed = (
            reconstruction[0, 0]
            .cpu()
            + 1.0
        ) / 2.0

        reconstructed = torch.clamp(
            reconstructed,
            0.0,
            1.0
        )

        reconstructed = (
            reconstructed.numpy()
            * 255.0
        ).astype(
            "uint8"
        )

        # ----------------------------------------------------
        # Create side-by-side image
        # ----------------------------------------------------

        original_image = Image.fromarray(
            original,
            mode="L"
        )

        reconstructed_image = Image.fromarray(
            reconstructed,
            mode="L"
        )

        comparison = Image.new(
            "L",
            (
                64,
                32
            )
        )

        comparison.paste(
            original_image,
            (
                0,
                0
            )
        )

        comparison.paste(
            reconstructed_image,
            (
                32,
                0
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            f"digit_{index}_label_{label}.png"
        )

        comparison.save(
            output_path
        )

        print(
            f"Digit {index} | "
            f"Label={label} | "
            f"MSE={mse:.6f} | "
            f"Saved: {output_path}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

average_mse = (
    total_mse / 10.0
)

print()
print("=" * 60)
print("VAE RECONSTRUCTION RESULTS")
print("=" * 60)

print()

print(
    "Average reconstruction MSE:",
    f"{average_mse:.6f}"
)

print()

print(
    "Original image is on the LEFT."
)

print(
    "VAE reconstruction is on the RIGHT."
)

print()

print(
    "Output directory:"
)

print(
    "D:\\StableDiffusionProject\\"
    + OUTPUT_DIR
)

print()
print("=" * 60)
print("VAE RECONSTRUCTION TEST COMPLETE")
print("=" * 60)