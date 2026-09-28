import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIMESTEPS = 300

LATENT_CHANNELS = 4

NUM_CLASSES = 10

CONDITION_DIM = 128

NUM_IMAGES = 16


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
# TIME EMBEDDING
# ============================================

class TimeEmbedding(nn.Module):

    def __init__(self, dim=128):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                1,
                dim
            ),

            nn.SiLU(),

            nn.Linear(
                dim,
                dim
            )
        )


    def forward(self, timestep):

        timestep = timestep.float()

        timestep = timestep.unsqueeze(1)

        timestep = timestep / TIMESTEPS

        return self.network(
            timestep
        )


# ============================================
# RESIDUAL BLOCK
# ============================================

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        condition_dim=128
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.time_projection = nn.Linear(
            128,
            out_channels
        )

        self.condition_projection = nn.Linear(
            condition_dim,
            out_channels
        )

        self.activation = nn.SiLU()

        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.skip = nn.Identity()


    def forward(
        self,
        x,
        time_embedding,
        condition
    ):

        h = self.conv1(x)

        h = self.activation(h)

        # Time conditioning

        time_condition = self.time_projection(
            time_embedding
        )

        time_condition = time_condition[
            :, :, None, None
        ]

        # Class conditioning

        class_condition = self.condition_projection(
            condition
        )

        class_condition = class_condition[
            :, :, None, None
        ]

        h = (
            h
            +
            time_condition
            +
            class_condition
        )

        h = self.conv2(h)

        h = self.activation(h)

        return h + self.skip(x)


# ============================================
# CONDITIONAL LATENT DIFFUSION U-NET
# ============================================

class ConditionalLatentDiffusionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.time_embedding = TimeEmbedding(
            128
        )

        self.class_embedding = nn.Embedding(
            NUM_CLASSES,
            CONDITION_DIM
        )

        self.input = nn.Conv2d(
            4,
            64,
            kernel_size=3,
            padding=1
        )

        self.res1 = ResBlock(
            64,
            128
        )

        self.down = nn.Conv2d(
            128,
            128,
            kernel_size=3,
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
            kernel_size=4,
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
            kernel_size=3,
            padding=1
        )


    def forward(
        self,
        x,
        timestep,
        labels
    ):

        time_embedding = self.time_embedding(
            timestep
        )

        class_condition = self.class_embedding(
            labels
        )

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_embedding,
            class_condition
        )

        x2 = self.down(x1)

        x2 = self.middle1(
            x2,
            time_embedding,
            class_condition
        )

        x2 = self.middle2(
            x2,
            time_embedding,
            class_condition
        )

        x3 = self.up(x2)

        x3 = torch.cat(
            [x3, x1],
            dim=1
        )

        x3 = self.res2(
            x3,
            time_embedding,
            class_condition
        )

        x3 = self.res3(
            x3,
            time_embedding,
            class_condition
        )

        return self.output(x3)


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
# LOAD VAE
# ============================================

vae = SpatialVAE(
    LATENT_CHANNELS
).to(DEVICE)

vae.load_state_dict(
    torch.load(
        "checkpoints/spatial_vae_epoch_10.pth",
        map_location=DEVICE
    )
)

vae.eval()

for parameter in vae.parameters():
    parameter.requires_grad = False


print()
print(
    "Spatial VAE loaded successfully!"
)


# ============================================
# LOAD CONDITIONAL MODEL
# ============================================

model = ConditionalLatentDiffusionUNet().to(
    DEVICE
)

model.load_state_dict(
    torch.load(
        "checkpoints/"
        "conditional_real_latent_epoch_5.pth",
        map_location=DEVICE
    )
)

model.eval()


print(
    "Conditional real latent diffusion "
    "model loaded successfully!"
)


# ============================================
# ASK USER FOR DIGIT
# ============================================

print()
print(
    "===================================="
)

print(
    "CONDITIONAL REAL LATENT GENERATION"
)

print(
    "===================================="
)

digit = int(
    input(
        "Enter digit to generate (0-9): "
    )
)

if digit < 0 or digit > 9:

    raise ValueError(
        "Digit must be between 0 and 9."
    )


# ============================================
# INITIAL RANDOM LATENT
# ============================================

latents = torch.randn(
    NUM_IMAGES,
    LATENT_CHANNELS,
    8,
    8,
    device=DEVICE
)

labels = torch.full(
    (NUM_IMAGES,),
    digit,
    dtype=torch.long,
    device=DEVICE
)


# ============================================
# REVERSE DIFFUSION
# ============================================

print()
print(
    "Starting conditional reverse diffusion..."
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
            latents,
            timestep,
            labels
        )

        alpha = alphas[t]

        alpha_bar = alpha_bars[t]

        beta = betas[t]

        latents = (
            1.0
            / torch.sqrt(alpha)
        ) * (
            latents
            -
            (
                (1.0 - alpha)
                /
                torch.sqrt(
                    1.0 - alpha_bar
                )
            )
            * predicted_noise
        )

        if t > 0:

            noise = torch.randn_like(
                latents
            )

            latents = (
                latents
                +
                torch.sqrt(beta)
                * noise
            )

        if t % 50 == 0:

            print(
                f"Diffusion step: {t}"
            )


# ============================================
# DECODE
# ============================================

print()
print(
    "Decoding generated latents..."
)

with torch.no_grad():

    generated_images = vae.decode(
        latents
    )

    generated_images = (
        generated_images
        + 1
    ) / 2

    generated_images = generated_images.clamp(
        0,
        1
    )


# ============================================
# SAVE IMAGE GRID
# ============================================

plt.figure(
    figsize=(8, 8)
)

for i in range(NUM_IMAGES):

    plt.subplot(
        4,
        4,
        i + 1
    )

    plt.imshow(
        generated_images[i, 0]
        .cpu()
        .numpy(),
        cmap="gray"
    )

    plt.axis("off")

    plt.title(
        f"Digit {digit}"
    )


plt.tight_layout()

output_path = (
    "outputs/"
    "conditional_real_latent_generated.png"
)

plt.savefig(
    output_path,
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
    "CONDITIONAL GENERATION COMPLETED"
)

print(
    "===================================="
)

print()

print(
    f"Generated digit: {digit}"
)

print(
    f"Images generated: {NUM_IMAGES}"
)

print()

print(
    "Generated image saved to:"
)

print(
    output_path
)