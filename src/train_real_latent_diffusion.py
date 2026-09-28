import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 5
LR = 2e-4

TIMESTEPS = 300

LATENT_CHANNELS = 4
LATENT_SIZE = 8


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
                stride=1,
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


    def encode(self, x):

        features = self.encoder(x)

        mu = self.mu(features)

        logvar = self.logvar(features)

        return mu, logvar


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
            time_dim,
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
# REAL LATENT DIFFUSION U-NET
# ============================================

class LatentDiffusionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.time_embedding = TimeEmbedding(
            128
        )

        # ------------------------------------
        # Input
        # 4 x 8 x 8
        # ------------------------------------

        self.input = nn.Conv2d(
            4,
            64,
            kernel_size=3,
            padding=1
        )

        # ------------------------------------
        # Encoder
        # ------------------------------------

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

        # 4x4 latent

        # ------------------------------------
        # Middle
        # ------------------------------------

        self.middle1 = ResBlock(
            128,
            128
        )

        self.middle2 = ResBlock(
            128,
            128
        )

        # ------------------------------------
        # Upsampling
        # ------------------------------------

        self.up = nn.ConvTranspose2d(
            128,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # ------------------------------------
        # Decoder
        # ------------------------------------

        self.res2 = ResBlock(
            128 + 128,
            128
        )

        self.res3 = ResBlock(
            128,
            64
        )

        # ------------------------------------
        # Output
        # ------------------------------------

        self.output = nn.Conv2d(
            64,
            4,
            kernel_size=3,
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

        # ====================================
        # Encoder
        # ====================================

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_embedding
        )

        x2 = self.down(x1)

        # ====================================
        # Middle
        # ====================================

        x2 = self.middle1(
            x2,
            time_embedding
        )

        x2 = self.middle2(
            x2,
            time_embedding
        )

        # ====================================
        # Decoder
        # ====================================

        x3 = self.up(x2)

        # Skip connection
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

        # ====================================
        # Predict noise
        # ====================================

        noise = self.output(x3)

        return noise


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
# LOAD SPATIAL VAE
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


# Freeze VAE

for parameter in vae.parameters():

    parameter.requires_grad = False


print()
print(
    "Spatial VAE loaded successfully!"
)


# ============================================
# DIFFUSION MODEL
# ============================================

model = LatentDiffusionUNet().to(
    DEVICE
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
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
# TRAINING
# ============================================

print()
print(
    "Starting real latent diffusion training..."
)


for epoch in range(EPOCHS):

    total_loss = 0.0


    for images, _ in loader:

        images = images.to(
            DEVICE
        )

        batch_size = images.shape[0]


        # ====================================
        # ENCODE IMAGE
        # ====================================

        with torch.no_grad():

            mu, _ = vae.encode(
                images
            )

            # Use mean latent
            latents = mu


        # ====================================
        # RANDOM TIMESTEPS
        # ====================================

        timestep = torch.randint(
            0,
            TIMESTEPS,
            (
                batch_size,
            ),
            device=DEVICE
        )


        # ====================================
        # RANDOM NOISE
        # ====================================

        noise = torch.randn_like(
            latents
        )


        # ====================================
        # FORWARD DIFFUSION
        # ====================================

        alpha_bar = alpha_bars[
            timestep
        ]

        alpha_bar = alpha_bar[
            :, None, None, None
        ]

        noisy_latent = (
            torch.sqrt(alpha_bar)
            * latents
            +
            torch.sqrt(
                1.0 - alpha_bar
            )
            * noise
        )


        # ====================================
        # PREDICT NOISE
        # ====================================

        predicted_noise = model(
            noisy_latent,
            timestep
        )


        # ====================================
        # LOSS
        # ====================================

        loss = torch.mean(
            (
                predicted_noise
                -
                noise
            ) ** 2
        )


        # ====================================
        # BACKPROPAGATION
        # ====================================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


    # ========================================
    # EPOCH RESULT
    # ========================================

    average_loss = (
        total_loss
        /
        len(loader)
    )


    print(
        f"Epoch {epoch + 1} completed "
        f"| Average Loss: "
        f"{average_loss:.4f}"
    )


    # ========================================
    # SAVE CHECKPOINT
    # ========================================

    torch.save(
        model.state_dict(),
        f"checkpoints/"
        f"real_latent_diffusion_epoch_"
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
    "REAL LATENT DIFFUSION TRAINING DONE"
)

print(
    "===================================="
)