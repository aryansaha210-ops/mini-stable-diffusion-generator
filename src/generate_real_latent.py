import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIMESTEPS = 300

LATENT_CHANNELS = 4

NUM_IMAGES = 16

CHECKPOINT = (
    "checkpoints/"
    "real_latent_diffusion_epoch_5.pth"
)

VAE_CHECKPOINT = (
    "checkpoints/"
    "spatial_vae_epoch_10.pth"
)


# ============================================
# SPATIAL VAE
# ============================================

class SpatialVAE(nn.Module):

    def __init__(self, latent_channels=4):

        super().__init__()

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
                padding=1
            ),
            nn.ReLU()
        )

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
                32, 1,
                kernel_size=3,
                padding=1
            ),
            nn.Tanh()
        )

    def decode(self, z):

        return self.decoder(z)


# ============================================
# RESIDUAL BLOCK
# ============================================

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim=128
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1
        )

        self.time_projection = nn.Linear(
            time_dim,
            out_channels
        )

        self.activation = nn.SiLU()

        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                1
            )

        else:

            self.skip = nn.Identity()

    def forward(
        self,
        x,
        time_embedding
    ):

        h = self.conv1(x)

        h = self.activation(h)

        time_condition = self.time_projection(
            time_embedding
        )

        time_condition = time_condition[
            :, :, None, None
        ]

        h = h + time_condition

        h = self.conv2(h)

        h = self.activation(h)

        return h + self.skip(x)


# ============================================
# TIME EMBEDDING
# ============================================

class TimeEmbedding(nn.Module):

    def __init__(self, dim=128):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(1, dim),

            nn.SiLU(),

            nn.Linear(dim, dim)
        )

    def forward(self, timestep):

        timestep = timestep.float()

        timestep = timestep.unsqueeze(1)

        timestep = timestep / TIMESTEPS

        return self.network(timestep)


# ============================================
# DIFFUSION U-NET
# ============================================

class LatentDiffusionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.time_embedding = TimeEmbedding(128)

        self.input = nn.Conv2d(
            4,
            64,
            3,
            padding=1
        )

        self.res1 = ResBlock(
            64,
            128
        )

        self.down = nn.Conv2d(
            128,
            128,
            3,
            stride=2,
            padding=1
        )

        self.middle1 = ResBlock(
            128,
            128
        )

        self.middle2 = ResBlock(
            128,
            128
        )

        self.up = nn.ConvTranspose2d(
            128,
            128,
            4,
            stride=2,
            padding=1
        )

        self.res2 = ResBlock(
            256,
            128
        )

        self.res3 = ResBlock(
            128,
            64
        )

        self.output = nn.Conv2d(
            64,
            4,
            3,
            padding=1
        )

    def forward(
        self,
        x,
        timestep
    ):

        time_embedding = self.time_embedding(
            timestep
        )

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_embedding
        )

        x2 = self.down(x1)

        x2 = self.middle1(
            x2,
            time_embedding
        )

        x2 = self.middle2(
            x2,
            time_embedding
        )

        x3 = self.up(x2)

        x3 = torch.cat(
            [x3, x1],
            dim=1
        )

        x3 = self.res2(
            x3,
            time_embedding
        )

        x3 = self.res3(
            x3,
            time_embedding
        )

        return self.output(x3)


# ============================================
# DEVICE
# ============================================

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================
# LOAD VAE
# ============================================

vae = SpatialVAE().to(DEVICE)

vae.load_state_dict(
    torch.load(
        VAE_CHECKPOINT,
        map_location=DEVICE
    )
)

vae.eval()

print()
print(
    "Spatial VAE loaded successfully!"
)


# ============================================
# LOAD DIFFUSION MODEL
# ============================================

model = LatentDiffusionUNet().to(
    DEVICE
)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

print(
    "Real latent diffusion model "
    "loaded successfully!"
)


# ============================================
# DIFFUSION SCHEDULE
# ============================================

betas = torch.linspace(
    1e-4,
    0.02,
    TIMESTEPS,
    device=DEVICE
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0
)


# ============================================
# START FROM RANDOM LATENT
# ============================================

latent = torch.randn(
    NUM_IMAGES,
    4,
    8,
    8,
    device=DEVICE
)


# ============================================
# REVERSE DIFFUSION
# ============================================

print()
print(
    "Starting real latent reverse diffusion..."
)

with torch.no_grad():

    for t in reversed(
        range(TIMESTEPS)
    ):

        timestep = torch.full(
            (NUM_IMAGES,),
            t,
            device=DEVICE,
            dtype=torch.long
        )

        predicted_noise = model(
            latent,
            timestep
        )

        beta = betas[t]

        alpha = alphas[t]

        alpha_bar = alpha_bars[t]

        # ------------------------------------
        # DDPM reverse step
        # ------------------------------------

        if t > 0:

            random_noise = torch.randn_like(
                latent
            )

        else:

            random_noise = torch.zeros_like(
                latent
            )

        latent = (
            (1 / torch.sqrt(alpha))
            *
            (
                latent
                -
                (
                    (1 - alpha)
                    /
                    torch.sqrt(
                        1 - alpha_bar
                    )
                )
                *
                predicted_noise
            )
            +
            torch.sqrt(beta)
            *
            random_noise
        )

        if t % 50 == 0:

            print(
                f"Diffusion step: {t}"
            )


# ============================================
# DECODE LATENTS
# ============================================

print()
print(
    "Decoding generated latents..."
)

with torch.no_grad():

    generated_images = vae.decode(
        latent
    )


# ============================================
# CONVERT TO DISPLAY RANGE
# ============================================

generated_images = (
    generated_images.clamp(-1, 1)
    + 1
) / 2


generated_images = (
    generated_images
    .cpu()
    .squeeze(1)
    .numpy()
)


# ============================================
# DISPLAY
# ============================================

fig, axes = plt.subplots(
    4,
    4,
    figsize=(8, 8)
)


for i, ax in enumerate(
    axes.flat
):

    ax.imshow(
        generated_images[i],
        cmap="gray"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/real_latent_generated.png",
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
    "REAL LATENT GENERATION COMPLETED"
)

print(
    "===================================="
)

print()
print(
    "Generated image saved to:"
)

print(
    "outputs/real_latent_generated.png"
)